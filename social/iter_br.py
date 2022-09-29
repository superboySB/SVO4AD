import numpy as np
import scipy.optimize as opt
import copy


def iterative_best_response_discrete(env, hf_list, given_svo=None, agent_id=None):
    n_agent = len(hf_list)
    u_prev = np.zeros((n_agent,))
    u_k = np.ones((n_agent,))
    svo_k = [hf.sample_svo for hf in hf_list]
    if given_svo is not None:
        if agent_id is not None:
            svo_k[agent_id] = given_svo[agent_id]
        else:
            svo_k = given_svo

    for _ in range(100):
        if np.linalg.norm(u_k - u_prev) < 1e-6:
            break
        u_k_ = np.zeros_like(u_k)
        for v_id in range(n_agent):
            best_utility = -1e10
            tmp_actions = u_k.copy()
            for action in range(env.action_space.n):
                tmp_actions[v_id] = action
                rewards = env.trial_step(tmp_actions)  # TODO: still need dynamic model
                # print(rewards)
                r_ego, r_others = rewards[v_id], rewards[rewards != v_id]
                tmp_utility = np.sum(np.cos(svo_k[v_id]) * r_ego + np.sin(svo_k[v_id]) * r_others) / (
                        n_agent - 1)
                if tmp_utility > best_utility:  # TODO: 是＞还是≥
                    u_k_[v_id] = action
                    best_utility = tmp_utility
        u_prev = u_k
        u_k = u_k_

    # print(u_k, "\n")
    return u_k
