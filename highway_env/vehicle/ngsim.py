from __future__ import division, print_function
import numpy as np
from highway_env.vehicle.kinematics import Vehicle


class NGSIMVehicle(Vehicle):
    """
    Create a human-like (InverseRL) driving agent.
    """

    def __init__(self, v_id, road, position, heading=0, speed=0,
                 v_length=None, v_width=None, ngsim_traj=None, human=True):
        super(NGSIMVehicle, self).__init__(v_id, road, position, heading, speed)

        self.ngsim_traj_data = ngsim_traj
        self.human = human

        if v_length is not None and v_width is not None:
            self.LENGTH = v_length  # Vehicle length [m]
            self.WIDTH = v_width  # Vehicle width [m]

        self.position_history = np.array(self.position)
        self.heading_history = []
        self.speed_history = []
        self.crash_history = []
        self.sim_step = 0

    def act(self, action=None):
        super().act(action)

    def step(self, dt=0):
        self.sim_step += 1
        self.heading_history.append(self.heading)
        self.speed_history.append(self.speed)
        self.crash_history.append(self.crashed)

        if self.human:
            self.position = self.ngsim_traj_data[self.sim_step][:2]
            self.speed = self.ngsim_traj_data[self.sim_step][2]
        if not self.human:
            if self.action is None:
                RuntimeError('You should give an action to an Ego car')
            else:
                super(NGSIMVehicle, self).step(dt)

        self.position_history = np.append(self.position_history, self.position, axis=0)

    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj_data[:self.sim_step + 1, :2]
        ego_traj = self.position_history.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return FDE
