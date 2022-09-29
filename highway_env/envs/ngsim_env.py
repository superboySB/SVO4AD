from __future__ import division, print_function, absolute_import

from highway_env.envs.common.abstract import *
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, StraightLane
from highway_env.utils import near_split, lmap
from highway_env.vehicle.ngsim import NGSIMVehicle
from highway_env.data.trajectory import build_trajectory
from gym.envs.registration import register


# from NGSIM_env.envs.base import AbstractEnv
# from NGSIM_env.road.road import Road, RoadNetwork
# from NGSIM_env.vehicle.control import NGSIMVehicle
# from NGSIM_env.road.lane import LineType, StraightLane
# from NGSIM_env.utils.utils import *


class NGSIMEnvMASVO(AbstractEnv):
    """
    A highway driving environment with NGSIM data.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "scene": 'us-101',
            "veh_list": [80, 81],
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "screen_width": 800,
            "screen_height": 300,
            "show_trajectories": True,
            "weights": [4.486, 1.233, -2.201, -2.932, -4.665, -3.079, -10.0, -5.346],  # for v_id: 1206
            # "weights": [0.959, -1.093, -1.158, -1.144, -0.802, -0.388, -10.0, -1.028],  # for v_id: 1740
            "reset_step": 0,
            "duration": 500,
        })
        return config

    def process_raw_trajectory(self, trajectory):
        trajectory = np.array(trajectory)
        for i in range(trajectory.shape[0]):
            x = trajectory[i][0] - 6
            y = trajectory[i][1]
            speed = trajectory[i][2]
            trajectory[i][0] = y / 3.281
            trajectory[i][1] = x / 3.281
            trajectory[i][2] = speed / 3.281

        return trajectory

    def _reset(self):
        """
        Reset the environment at a given time (scene) and specify whether use human target
        """
        self.ego_id = self.config["veh_list"][0]
        self.trajectory_set = build_trajectory(self.config["scene"], 0, self.config["veh_list"][0])
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']
        self.duration = len(self.ego_trajectory) - 3
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        print(f"{self.ego_id}'s surrounding: {self.surrounding_vehicles}")
        if self.config["vehicles_count"] == 666:
            self.config["veh_list"] = copy.deepcopy(self.surrounding_vehicles)
            self.config["vehicles_count"] = len(self.surrounding_vehicles) - 1
        self.surrounding_vehicles.pop(0)

        self._create_road()
        self._create_vehicles()

    def _create_road(self):
        """
        Create a road composed of NGSIM road network
        """
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        if self.config["scene"] == 'us-101':
            length = 2150 / 3.281  # m
            width = 12 / 3.281  # m
            ends = [0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('merge_in', 's2',
                         StraightLane([480 / 3.281, 5.5 * width], [ends[1], 5 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(6):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_out lanes
            net.add_lane('s3', 'merge_out',
                         StraightLane([ends[2], 5 * width], [1550 / 3.281, 7 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        elif self.config["scene"] == 'i-80':
            length = 1700 / 3.281
            lanes = 6
            width = 12 / 3.281
            ends = [0, 600 / 3.281, 700 / 3.281, 900 / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s1', 's2',
                         StraightLane([380 / 3.281, 7.1 * width], [ends[1], 6 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s2', 's3',
                         StraightLane([ends[1], 6 * width], [ends[2], 6 * width], width=width, line_types=[s, c]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lane
            net.add_lane('s3', 's4',
                         StraightLane([ends[2], 6 * width], [ends[3], 5 * width], width=width, line_types=[n, c]))

            # forth section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[3], lane * width]
                end = [ends[4], lane * width]
                net.add_lane('s4', 's5', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)
        ego_trajectory = whole_trajectory[self.config["reset_step"]:]
        self.duration = min(len(self.ego_trajectory) - 3, self.config["duration"])

        v_id = 0
        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = NGSIMVehicle(self.ego_id,
                                              self.road, ego_trajectory[0][:2],
                                              speed=ego_trajectory[0][2],
                                              v_length=self.trajectory_set['ego']['length'] / 3.281,
                                              v_width=self.trajectory_set['ego']['width'] / 3.281,
                                              ngsim_traj=ego_trajectory,
                                              human=True)
            v_id += 1
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                veh_id = self.config["veh_list"][v_id]
                other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[
                                   self.config["reset_step"]:]
                vehicle = NGSIMVehicle(veh_id,
                                       self.road, other_trajectory[0][:2],
                                       speed=other_trajectory[0][2],
                                       v_length=self.trajectory_set[veh_id]['length'] / 3.281,
                                       v_width=self.trajectory_set[veh_id]['width'] / 3.281,
                                       ngsim_traj=other_trajectory,
                                       human=True)
                self.road.vehicles.append(vehicle)
                v_id += 1

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or go off road or the time is out.
        """
        return self.vehicle.crashed or self.time >= self.duration or self.vehicle.position[0] >= 2150 / 3.281

    def step(self, actions=None):
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self.road.act()
        self.road.step()
        self.time += 1
        self._automatic_rendering()
        self.enable_auto_render = False

        obs = self.observation_type.observe()
        reward_list = ngsim_ma_reward_list(self.time, self.config, self.road)

        terminal = self._is_terminal()
        info = self._info(obs, actions)

        return obs, reward_list[0], terminal, info

    def _info(self, obs, actions) -> dict:
        info = {
            "speed": [vehicle.speed for vehicle in self.road.vehicles],
            "crashed": [vehicle.crashed for vehicle in self.road.vehicles],
            "actions": actions,
            "reward_list": ngsim_ma_reward_list(self.time, self.config, self.road),
            "time": self.time
        }
        return info


def vehicle_features(time, ego_vehicle, road):
    """
    Hand-crafted features
    :return: the array of the defined features
    """
    # ego motion x->v->a->jerk
    if time >= 3:
        ego_longitudial_positions = ego_vehicle.position_history.reshape(-1, 2)[time - 3:, 0]  # 直向
        ego_longitudial_speeds = (ego_longitudial_positions[1:] - ego_longitudial_positions[:-1]) / 0.1
        ego_longitudial_accs = (ego_longitudial_speeds[1:] - ego_longitudial_speeds[:-1]) / 0.1
        ego_longitudial_jerks = (ego_longitudial_accs[1:] - ego_longitudial_accs[:-1]) / 0.1

        ego_lateral_positions = ego_vehicle.position_history.reshape(-1, 2)[time - 3:, 1]  # 侧向
        ego_lateral_speeds = (ego_lateral_positions[1:] - ego_lateral_positions[:-1]) / 0.1
        ego_lateral_accs = (ego_lateral_speeds[1:] - ego_lateral_speeds[:-1]) / 0.1
    else:
        ego_longitudial_speeds = [0]
        ego_longitudial_accs = [0]
        ego_longitudial_jerks = [0]
        ego_lateral_speeds = [0]
        ego_lateral_accs = [0]

    # travel efficiency
    ego_speed = abs(ego_longitudial_speeds[-1])

    # comfort
    ego_longitudial_acc = ego_longitudial_accs[-1]
    ego_lateral_acc = ego_lateral_accs[-1]
    ego_longitudial_jerk = ego_longitudial_jerks[-1]

    # time headway front (THWF) and time headway behind (THWB)
    THWFs = [100]
    THWBs = [100]
    for v in road.vehicles:
        if v.position[0] > ego_vehicle.position[0] and abs(
                v.position[1] - ego_vehicle.position[
                    1]) < ego_vehicle.WIDTH and ego_vehicle.speed >= 1:
            THWF = (v.position[0] - ego_vehicle.position[0]) / ego_vehicle.speed
            THWFs.append(THWF)
        elif v.position[0] < ego_vehicle.position[0] and abs(
                v.position[1] - ego_vehicle.position[1]) < ego_vehicle.WIDTH and v.speed >= 1:
            THWB = (ego_vehicle.position[0] - v.position[0]) / v.speed
            THWBs.append(THWB)

    THWF = np.exp(-min(THWFs))
    THWB = np.exp(-min(THWBs))

    # avoid collision
    collision = 1 if ego_vehicle.crashed else 0

    # interaction (social) impact
    social_impact = 0
    for i, v in enumerate(road.vehicles):
        if i == 0:
            continue
        if v.human is True and v.speed != 0:
            social_impact += np.abs(v.speed - v.speed_history[-1]) / 0.1 if v.speed - v.speed_history[-1] < 0 else 0

    # feature array
    features = np.array([ego_speed, abs(ego_longitudial_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk),
                         THWF, THWB, collision, social_impact])

    max_v = np.array([20.62959301, 3.04119561, 1.6431296, 11.28073619, 0.50698337, 0.28708416, 1, 12.6157628])

    return features / max_v


def ngsim_ma_reward_list(time, config, road):
    reward_list = []
    for vehicle in road.vehicles:
        reward = np.sum(config["weights"] * vehicle_features(time, vehicle, road))
        reward_list.append(reward)
    return np.array(reward_list)


register(
    id='ngsim-masvo-v0',
    entry_point='highway_env.envs:NGSIMEnvMASVO',
)
