from mpc import LinearModelPredictiveController
from params import Params
from OmniLeafSystem import OmniLeafSystem_

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput
from pydrake.systems.analysis import Simulator

import numpy as np
import matplotlib.pyplot as plt
import logging as log


def main():
    params = Params()

    builder = DiagramBuilder()

    double_integrator = builder.AddSystem(OmniLeafSystem_[float]())
    # plant_context = double_integrator.CreateDefaultContext()

    controller = builder.AddSystem(
        LinearModelPredictiveController(OmniLeafSystem_[float](), params=params))
    controller_context = controller.CreateDefaultContext()

    builder.Connect(controller.GetOutputPort("action"),
                    double_integrator.GetInputPort("wheel"))
    builder.Connect(double_integrator.GetOutputPort(
        "state"), controller.GetInputPort("state"))

    state_logger = LogVectorOutput(
        double_integrator.GetOutputPort("state"), builder)
    action_logger = LogVectorOutput(
        controller.GetOutputPort("action"), builder)
    trajectory_logger = LogVectorOutput(controller.GetOutputPort(
        "trajectory"), builder, publish_period=params.sample_time)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    diagram.ForcedPublish(context)

    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()
    simulator.set_target_realtime_rate(1.0)

    initial_state = np.array([0., 0., 0.])
    final_state = np.array([5., 5., np.pi/2])

    plant_context = double_integrator.GetMyContextFromRoot(sim_context)
    plant_context.SetContinuousState(initial_state)

    controller_context.get_time()

    controller.GetInputPort("goal").FixValue(
        controller.GetMyContextFromRoot(sim_context), final_state)

    # simulator.Initialize()
    simulator.AdvanceTo(simulator.get_context().get_time() + params.Tf)

    state_log = state_logger.FindLog(sim_context)
    action_log = action_logger.FindLog(sim_context)
    trajectory_log = trajectory_logger.FindLog(sim_context)

    # Use log to plot
    times = state_log.sample_times()
    states = state_log.data()  # [x, y, theta]
    wheel_velocities = action_log.data()  # Wheel velocities
    trajectory = trajectory_log.data()
    trajectory_sample_time = trajectory_log.sample_times()

    fig, ax = plt.subplots(4, 2)
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("x")
    ax[0, 0].plot(times, states[0, :])
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("y")
    ax[1, 0].plot(times, states[1, :])
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 0].set_ylabel("theta")
    ax[2, 0].plot(times, states[2, :])
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].set_ylabel("v1")
    ax[0, 1].plot(times, wheel_velocities[0, :])
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].set_ylabel("v2")
    ax[1, 1].plot(times, wheel_velocities[1, :])
    ax[2, 1].set_xlabel("Time (s)")
    ax[2, 1].set_ylabel("v3")
    ax[2, 1].plot(times, wheel_velocities[2, :])
    ax[3, 1].set_xlabel("Time (s)")
    ax[3, 1].set_ylabel("v4")
    ax[3, 1].plot(times, wheel_velocities[3, :])
    ax[3, 0].set_visible(False)

    # TODO: Plot the trajectories
    plt.show()


if __name__ == "__main__":
    main()
