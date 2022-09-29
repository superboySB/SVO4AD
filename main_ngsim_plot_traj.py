import gym
from gym.wrappers import Monitor
import highway_env
from highway_env.data.settings import vehicle_i80, vehicles_us101
from multiprocessing import Process
import numpy as np
import time

start_time = 0
duration = 10000

save_dir = "us101_traj"  # "i80_traj", "us101_traj"
scene = 'us-101'    # 'i-80', 'us-101'
veh_id_array = np.array(vehicles_us101)  # vehicle_i80, vehicles_us101
visualize = True

# save_dir = "us101_traj"
# scene = 'us-101'
# veh_id_array = np.array(vehicles_us101)
# visualize = True

# 跑的太慢了，加入多进程优化
num_process = 64
num_vehicles = veh_id_array.shape[0]


def main(sub_veh_id_array):
    for veh_id in sub_veh_id_array:
        env = gym.make("ngsim-masvo-v0")
        env = Monitor(env, directory=f"{save_dir}/{veh_id}", video_callable=lambda e: True, force=True)
        env.unwrapped.set_monitor(env)
        env.configure({
            "offscreen_rendering": True,
            "veh_list": [veh_id],
            "vehicles_count": 666,
            "reset_step": start_time,
            "duration": duration,  # [step]
            "scene": scene,  # us-101, i-80
        })

        done = False
        obs = env.reset()

        step = 0
        while not done:
            # 1. calculate best action
            actions = None

            # 2. step
            obs, reward, done, info = env.step(actions)
            reward_list = info["reward_list"]
            # print(info)
            step += 1

            if visualize:
                env.render()

        time.sleep(60)
        env.close()


# 备选环境：
# 1. ngsim-masvo-v0
if __name__ == "__main__":
    print(f"total num: {num_vehicles}, num process: {num_process}")
    sub_veh_id_arrays = np.array_split(veh_id_array, num_process)
    for pid, sub_veh_id_array in enumerate(sub_veh_id_arrays):
        print(f"Process_{pid}: {sub_veh_id_array}\n")
    process = [Process(target=main, args=(sub_veh_id_array,)) for sub_veh_id_array in sub_veh_id_arrays]
    [p.start() for p in process]
    [p.join() for p in process]
