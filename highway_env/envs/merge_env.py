import numpy as np
import copy
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.envs.common.action import DiscreteMetaAction


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.config["right_lane_reward"] * self.vehicle.lane_index[2] / 1 \
                 + self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.config["merging_speed_reward"] * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > 370

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road

        # ego_vehicle = self.action_type.vehicle_class(road,
        #                                              road.network.get_lane(("a", "b", 1)).position(30, 0),
        #                                              speed=30)
        # road.vehicles.append(ego_vehicle)
        #
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))
        #
        # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)


        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=30))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle


class MergeEnvSVO(MergeEnv):
    """
    A variant of merge-v0 with svo prediction:
        - comming soon
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "vehicles_count": 5,
            "duration": 30,  # [s]
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "reward_speed_range": [20, 30],
        })
        return cfg

    def trial_step(self, actions):
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        road_copy = copy.deepcopy(self.road)
        for v_id, vehicle in enumerate(road_copy.vehicles):
            vehicle.act(DiscreteMetaAction.ACTIONS_ALL[actions[v_id]])

        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            road_copy.step(1 / self.config["simulation_frequency"])

        self.enable_auto_render = False
        reward_list = merge_ma_reward_list(self.config, road_copy, actions[0])
        del road_copy

        return reward_list

    def _info(self, obs, action) -> dict:
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "reward_list": merge_ma_reward_list(self.config, self.road, action),
            "action": action,
        }
        return info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               self.vehicle.position[0] > 370


def merge_ma_reward_list(config, road, ego_action):
    reward_list = []
    for v_id, vehicle in enumerate(road.vehicles):

        scaled_speed = utils.lmap(vehicle.speed, config["reward_speed_range"], [0, 1])
        reward = config["collision_reward"] * vehicle.crashed \
                 + config["right_lane_reward"] * vehicle.lane_index[2] / 1 \
                 + config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
            reward += config["merging_speed_reward"] * \
                      (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        # TODO: 有点阴间，并且目前IDM vehicle无法得到meta discrete action
        if v_id == 0:
            action_reward = {0: config["lane_change_reward"],
                             1: 0,
                             2: config["lane_change_reward"],
                             3: 0,
                             4: 0}
            reward_list.append(action_reward[ego_action] + reward)
        else:
            reward_list.append(reward)

    return np.array(reward_list)


register(
    id='merge-v0',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='merge-svo-v0',
    entry_point='highway_env.envs:MergeEnvSVO',
)

register(
    id='merge-masvo-v0',
    entry_point='highway_env.envs:MergeEnvMASVO',
)
