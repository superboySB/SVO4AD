import numpy as np
from gym.envs.registration import register

from highway_env.utils import lmap
from highway_env.envs.common.abstract import *
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs.common.action import DiscreteMetaAction


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
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
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        v_id = 0
        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                v_id,
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            v_id += 1
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(v_id, self.road,
                                                            spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                v_id += 1

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class HighwayEnvSVO(HighwayEnv):
    """
    A variant of highway-v0 with svo prediction:
        - comming soon
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "vehicles_count": 5,
            "duration": 30,  # [s]
            "collision_reward": -2,
            "off_road_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
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
            road_copy.act()
            road_copy.step(1 / self.config["simulation_frequency"])

        self.enable_auto_render = False
        reward_list = highway_ma_reward_list(self.config, road_copy)
        del road_copy

        return reward_list

    def _info(self, obs, action) -> dict:
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "reward_list": highway_ma_reward_list(self.config, self.road),
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass
        return info


class HighwayEnvMASVO(HighwayEnv):
    """
    A variant of highway-v0 with svo prediction:
        - comming soon
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "policy_frequency": 1,
            "vehicles_count": 5,
            "duration": 30,  # [s]
            "collision_reward": -1,
            "right_lane_reward": 1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0,
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
            road_copy.act()
            road_copy.step(1 / self.config["simulation_frequency"])

        self.enable_auto_render = False
        reward_list = highway_ma_reward_list(self.config, road_copy)
        del road_copy

        return reward_list

    def step(self, actions):
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            if not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                for v_id, vehicle in enumerate(self.road.vehicles):
                    vehicle.act(DiscreteMetaAction.ACTIONS_ALL[actions[v_id]])

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

        obs = self.observation_type.observe()
        reward_list = highway_ma_reward_list(self.config, self.road)
        terminal = self._is_terminal()
        info = self._info(obs, actions)

        return obs, reward_list[0], terminal, info

    def _info(self, obs, action) -> dict:
        info = {
            "speed": [vehicle.speed for vehicle in self.road.vehicles],
            "crashed": [vehicle.crashed for vehicle in self.road.vehicles],
            "reward_list": highway_ma_reward_list(self.config, self.road),
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass
        return info

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        v_id = 0
        self.controlled_vehicles = []
        for others in other_per_controlled:
            # TODO: general
            # controlled_vehicle = self.action_type.vehicle_class.create_random(
            #     v_id,
            #     self.road,
            #     speed=25,
            #     lane_id=0,
            #     spacing=self.config["ego_spacing"]
            # )
            # v_id += 1
            # self.controlled_vehicles.append(controlled_vehicle)
            # self.road.vehicles.append(controlled_vehicle)
            # for _ in range(others):
            #     vehicle = self.action_type.vehicle_class.create_random(v_id, self.road, spacing=1 / self.config["vehicles_density"])
            #     self.road.vehicles.append(vehicle)
            #     v_id += 1

            # TODO: for demo
            setting_dict = {
                0: {"lane_id": 0, "position": 150, "speed": 15},
                1: {"lane_id": 1, "position": 155, "speed": 14},    # 165, 13
                2: {"lane_id": 0, "position": 160, "speed": 10},
            }
            controlled_vehicle = ControlledVehicle.create_according_to(
                v_id,
                self.road,
                speed=setting_dict[v_id]["speed"],
                lane_id=setting_dict[v_id]["lane_id"],
                position=setting_dict[v_id]["position"],
            )
            v_id += 1
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            for _ in range(others):
                vehicle = ControlledVehicle.create_according_to(v_id, self.road,
                                                                lane_id=setting_dict[v_id]["lane_id"],
                                                                speed=setting_dict[v_id]["speed"],
                                                                position=setting_dict[v_id]["position"])
                self.road.vehicles.append(vehicle)
                v_id += 1


def highway_ma_reward_list(config, road):
    reward_list = []
    for vehicle in road.vehicles:
        neighbours = road.network.all_side_lanes(vehicle.lane_index)
        num_lane = max(len(neighbours) - 1, 1)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        scaled_speed = lmap(vehicle.speed, config["reward_speed_range"], [0, 1])

        # TODO: 定制各个车的朝向
        if vehicle.id != 0:
            lane = -lane

        reward = \
            + config["collision_reward"] * vehicle.crashed \
            + config["right_lane_reward"] * lane / num_lane \
            + config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = 0 if not vehicle.on_road else reward
        # reward = utils.lmap(reward,
        #                     [config["collision_reward"],
        #                      config["high_speed_reward"] + config["right_lane_reward"]],
        #                     [0, 1])
        reward_list.append(reward)
    return np.array(reward_list)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id='highway-svo-v0',
    entry_point='highway_env.envs:HighwayEnvSVO',
)

register(
    id='highway-masvo-v0',
    entry_point='highway_env.envs:HighwayEnvMASVO',
)
