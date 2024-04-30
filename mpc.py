from pydrake.solvers import MathematicalProgram, Solve, BoundingBoxConstraint, SnoptSolver
from pydrake.trajectories import PiecewisePolynomial
from pydrake.planning import DirectCollocation, DirectCollocationConstraint, AddDirectCollocationConstraint
from pydrake.systems.framework import BasicVector, LeafSystem_, LeafSystem
from pydrake.systems.scalar_conversion import TemplateSystem

from pydrake.all import Value, eq
from params import Params

import numpy as np
from copy import copy
import logging as log

# TODO: Move direct collocation to the ctor, allowing for faster MPC


class LinearModelPredictiveController(LeafSystem):
    def __init__(self, plant: LeafSystem, params=Params(), log_level=log.INFO):
        LeafSystem.__init__(self)

        log.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
        self._plant = plant
        self._params = params
        self._context = plant.CreateDefaultContext()

        self.state_input_port = self.DeclareVectorInputPort(
            "state", self._params.num_states)  # 2 for number of states
        self.goal_input_port = self.DeclareVectorInputPort(
            "goal", self._params.num_states)

        self.state_trajectory = self.DeclareAbstractState(Value(PiecewisePolynomial.FirstOrderHold(
            [0., 1.], [[0., 0.]] * self._params.num_states)))  # * 2 for 2 states
        self.input_trajectory = self.DeclareAbstractState(Value(PiecewisePolynomial.FirstOrderHold(
            [0., 1.], [[0., 0.]] * self._params.num_inputs)))  # * 1 for 1 input
        self.time_offset = self.DeclareAbstractState(Value(BasicVector([0.])))

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=self._params.sample_time, offset_sec=0.0, update=self._Update)

        # Torque == Action
        self.action_output_port = self.DeclareVectorOutputPort(
            "action", BasicVector(self._params.num_inputs), self._DoCalcAction)

        num_decision = self._params.N * \
            (self._params.num_states + self._params.num_inputs)

        self.trajectory_output_port = self.DeclareVectorOutputPort(
            "trajectory", BasicVector(num_decision), self._DoCalcTrajectory)

    def _Update(self, context, state):
        initial_state = self.state_input_port.Eval(context)
        goal_state = self.goal_input_port.Eval(context)

        N = self._params.N
        dt = self._params.time_horizon / (N - 1)

        sim_time = context.get_time()

        dircol = DirectCollocation(
            self._plant, self._context, num_time_samples=N, minimum_time_step=dt, maximum_time_step=dt)
        dircol.AddEqualTimeIntervalsConstraints()

        prog = dircol.prog()

        prog.AddBoundingBoxConstraint(
            initial_state, initial_state, dircol.initial_state())

        prog.AddBoundingBoxConstraint(
            goal_state, goal_state, dircol.final_state())

        force_limit = self._params.wheel_speed_limit
        speed_limit = np.zeros(self._params.num_inputs) + force_limit
        dircol.AddConstraintToAllKnotPoints(BoundingBoxConstraint(
            lb=-speed_limit, ub=speed_limit), vars=dircol.input())

        # Velocity constraints
        velocity_magnitude = dircol.state()[0]**2 + dircol.state()[1]**2
        velocity_limit = self._params.max_velocity
        dircol.AddConstraintToAllKnotPoints(
            velocity_limit**2 >= velocity_magnitude)

        if sim_time != 0:
            initial_state_trajectory = copy(
                context.get_abstract_state(self.state_trajectory).get_value())

            initial_input_trajectory = copy(
                context.get_abstract_state(self.input_trajectory).get_value())

            dircol.SetInitialTrajectory(
                initial_input_trajectory, initial_state_trajectory)

        else:
            state_trajectory_guess = PiecewisePolynomial.FirstOrderHold(
                [0., dt * (N-1)], np.column_stack((initial_state, goal_state)))
            dircol.SetInitialTrajectory(
                PiecewisePolynomial(), state_trajectory_guess)

        # TODO: Cost function
        error = dircol.state() - goal_state
        u = dircol.input()
        Q = np.diag([1., 1., 1.])
        R = np.diag([1., 1., 1., 1.])
        dircol.AddRunningCost(error.T @ Q @ error + u.T @ R @ u)

        # Note: Gurobi would be the best solver for this problem, but it is not available in the current environment
        solver = SnoptSolver()
        solver_id = solver.solver_id()

        prog.SetSolverOption(solver_id, "Iterations Limits", 1e5)
        prog.SetSolverOption(solver_id, "Major Iterations Limit", 200)
        prog.SetSolverOption(solver_id, "Major Feasibility Tolerance", 5e-6)
        prog.SetSolverOption(solver_id, "Major Optimality Tolerance", 1e-4)
        prog.SetSolverOption(solver_id, "Superbasics limit", 2000)
        prog.SetSolverOption(solver_id, "Linesearch tolerance", 0.9)

        result = solver.Solve(prog)

        if not result.is_success():
            infeasible = result.GetInfeasibleConstraints(prog)
            log.debug("Infeasible constraints:")
            for i in infeasible:
                log.debug(i)

        input_trajectory = dircol.ReconstructInputTrajectory(result)
        # times = dircol.GetSampleTimes(result)

        state_trajectory = dircol.ReconstructStateTrajectory(result)

        state.get_mutable_abstract_state(int(self.state_trajectory)).set_value(
            state_trajectory)
        state.get_mutable_abstract_state(
            int(self.input_trajectory)).set_value(input_trajectory)
        state.get_mutable_abstract_state(
            int(self.time_offset)).set_value(sim_time)

    def _DoCalcAction(self, context, output):
        sim_time = context.get_time()

        time_offset = context.get_abstract_state(
            self.time_offset).get_value().value()

        input_trajectory = context.get_abstract_state(
            self.input_trajectory).get_value()

        value = input_trajectory.value(sim_time - time_offset)

        output.SetFromVector(value)

    def _DoCalcTrajectory(self, context, output):
        input_trajectory = context.get_abstract_state(
            self.input_trajectory).get_value()

        state_trajectory = context.get_abstract_state(
            self.state_trajectory).get_value()

        times = np.linspace(state_trajectory.start_time(
        ), state_trajectory.end_time(), self._params.N)

        # TODO: Recalculate times to not depend on equal time intervals
        input_values = input_trajectory.vector_values(times)

        state_values = state_trajectory.vector_values(times)

        decision = np.concatenate(
            (state_values.flatten(), input_values.flatten()), axis=0)

        output.SetFromVector(decision)
