import gym
from gym.wrappers import Monitor
import highway_env
from social.iter_br import iterative_best_response_discrete
from social.svo import HistogramSVO
from utils.visualize import plot_svo
import numpy as np

# 20211230 weight1740
# [1210, 1192, 1196, 1198, 1200, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1211, 1213]
# start 60 step 200
# save_dir = "us101_1210_1207"
# scene='us-101'
# veh_id = [1210, 1207]
# rollout_step = 5

# 20220105 weight1206
start_time = 350
duration = 200
save_dir = "i80_45_46"
scene = 'i-80'
veh_id = [45, 46]
rollout_step = 20

# 20220106 weight1206
# start_time = 30
# duration = 200
# save_dir = "i80_202_210_213"
# scene = 'i-80'
# veh_id = [202, 210, 213]
# rollout_step = 20

# 20220111 weight1206
# start_time = 100
# duration = 500
# save_dir = "i80_2083_2111_2103"  # 好像有红灯
# scene = 'i-80'
# veh_id = [2083, 2111, 2103]
# rollout_step = 20

# start_time = 170
# duration = 250
# save_dir = "i80_2109_2115_2067"
# scene = 'i-80'
# veh_id = [2109, 2115, 2067]
# rollout_step = 5

# start_time = 50
# duration = 200
# save_dir = "i80_2321_2333_2314"
# scene = 'i-80'
# veh_id = [2321, 2333, 2314]
# rollout_step = 5

# start_time = 50
# duration = 100
# save_dir = "i80_2148_2155_2136"
# scene = 'i-80'
# veh_id = [2148, 2155, 2136]
# rollout_step = 5


num_vehicles = len(veh_id)
num_scenes = 1
visualize = True


def main():
    env = gym.make("ngsim-masvo-v0")
    env = Monitor(env, directory=f"{save_dir}/videos", video_callable=lambda e: True, force=True)
    env.unwrapped.set_monitor(env)
    env.configure({
        "offscreen_rendering": True,
        "veh_list": veh_id,
        "vehicles_count": num_vehicles - 1,
        "reset_step": start_time,
        "duration": duration,  # [step]
        "scene": scene,  # us-101, i-80
    })

    for scene_id in range(num_scenes):
        done = False
        obs = env.reset()

        # initialize SVO and HF
        hf_list = [HistogramSVO(rollout_step) for _ in range(num_vehicles)]

        step = 0
        reward_history = np.zeros([rollout_step, num_vehicles])
        while not done:
            # 1. calculate best action
            actions = None

            # 2. step
            obs, reward, done, info = env.step(actions)
            reward_list = info["reward_list"]
            reward_history[0:-1] = reward_history[1:]
            reward_history[-1] = reward_list
            step += 1
            print(info)

            if visualize:
                env.render()
                if done:
                    plot_svo(hf_list, step, save_dir=f"{save_dir}/videos/svo_scene_{scene_id}.png", veh_id_list=veh_id)

            if step >= rollout_step:
                # 3. update HF by posteriori_probabilities
                reward_history_traj = reward_history.sum(axis=0)
                for tmp_ego_index in range(num_vehicles):
                    num_hf_discrete = hf_list[tmp_ego_index].n_discrete
                    utility_exp_list = np.ones([num_hf_discrete, ]) / num_hf_discrete
                    for i, svo_hist in enumerate(hf_list[tmp_ego_index].histograms):
                        r_ego, r_others = reward_history_traj[tmp_ego_index], reward_history_traj[
                            reward_history_traj != tmp_ego_index]
                        utility = np.sum(np.cos(svo_hist) * r_ego + np.sin(svo_hist) * r_others) / (num_vehicles - 1)
                        utility_exp_list[i] = np.exp(utility)
                    post_probs = utility_exp_list / (np.sum(utility_exp_list) + 1e-6)
                    hf_list[tmp_ego_index].update(post_probs)

        del hf_list

    env.close()


# 备选环境：
# 1. ngsim-masvo-v0
if __name__ == "__main__":
    main()
