from mpc import LinearModelPredictiveController
from params import Params
from OmniLeafSystem import OmniLeafSystem_

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput
from pydrake.systems.analysis import Simulator

from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

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
    final_state = np.array([3., 5., np.pi/2])

    plant_context = double_integrator.GetMyContextFromRoot(sim_context)
    plant_context.SetContinuousState(initial_state)

    controller_context.get_time()

    controller.GetInputPort("goal").FixValue(controller.GetMyContextFromRoot(sim_context), final_state)

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

    n = wheel_velocities[0].shape[0]
    velocity = np.zeros((3, n))

    for i in range(n):
        v1 = wheel_velocities[0, i]
        v2 = wheel_velocities[1, i]
        v3 = wheel_velocities[2, i]
        theta = states[2, i]
        
        v_heading = np.sqrt(3) * (v3 - v1) / 3
        v_normal = (v3 + v1) / 3 - (2/3) * v2
        omega = (1 / (3 * params.d)) * (v1 + v2 + v3)

        x_dot = v_heading * np.cos(theta) + v_normal * np.sin(theta)
        y_dot = v_heading * np.sin(theta) - v_normal * np.cos(theta)
        theta_dot = omega
        
        velocity[0, i] = x_dot
        velocity[1, i] = y_dot
        velocity[2, i] = theta_dot

    fig, ax = plt.subplots(3, 1)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("x")
    ax[0].plot(times, velocity[0, :])
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("y")
    ax[1].plot(times, velocity[1, :])
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("theta")
    ax[2].plot(times, velocity[2, :])

    plt.show()

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

    robot_radius = 0.5  
    arrow_length = 0.3  
    rounding_radius = 0.1  

    square_vertices = np.array([
        [-robot_radius, -robot_radius + rounding_radius],
        [-robot_radius + rounding_radius, -robot_radius],
        [robot_radius - rounding_radius, -robot_radius],
        [robot_radius, -robot_radius + rounding_radius],
        [robot_radius, robot_radius - rounding_radius],
        [robot_radius - rounding_radius, robot_radius],
        [-robot_radius + rounding_radius, robot_radius],
        [-robot_radius, robot_radius - rounding_radius]
    ])
    
    
    pose = np.array([0, 0, np.pi / 2])  

    v_x = 0.0 
    v_y = 0.0 
    omega = 0.0  

    dt = 0.1 
    duration = 10 

    fig, ax = plt.subplots()
    ax.set_xlim([-4, 8])
    ax.set_ylim([-4, 8])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Omnibot Simulation')
    line_robot, = ax.plot([], [], 'b-') 
    line_arrow, = ax.plot([], [], 'r-', lw=2)  

    arrow_head = FancyArrowPatch(posA=(0, 0), posB=(0, 0), arrowstyle='->', color='r', lw=2)
    ax.add_patch(arrow_head)

    dt = 0.1

    def init():
        line_robot.set_data([], [])
        line_arrow.set_data([], [])
        arrow_head.set_positions(posA=(0, 0), posB=(0, 0))
        return line_robot, line_arrow

    def animate(frame):

        v_x = states[0, frame]
        v_y = states[1, frame]
        v_t = states[2, frame]
                        
        pose[0] = v_x
        pose[1] = v_y
        pose[2] = -v_t
     
        print(pose)

        rotation_matrix = np.array([
            [np.cos(pose[2]), -np.sin(pose[2])],
            [np.sin(pose[2]), np.cos(pose[2])]
        ])
        rotated_vertices = square_vertices.dot(rotation_matrix)
        translated_vertices = rotated_vertices + pose[:2]

        arrow_start = pose[:2]
        arrow_end = arrow_start + arrow_length * np.array([np.sin(pose[2]), np.cos(pose[2])])
        line_arrow.set_data([arrow_start[0], arrow_end[0]], [arrow_start[1], arrow_end[1]])
        arrow_head.set_positions(posA=arrow_start, posB=arrow_end)

        line_robot.set_data(*zip(*np.vstack((translated_vertices, translated_vertices[0]))))

        return line_robot, line_arrow

    s = states[1].shape
    ani = FuncAnimation(fig, animate, init_func=init, frames=int(s[0]), blit=True, interval=dt*1000, repeat=False)

    plt.show()


if __name__ == "__main__":
    main()
