import matplotlib.pyplot as plt
import numpy as np


def plot_svo(hf_list, n_steps, save_dir=None,veh_id_list=None):
    plt.figure(figsize=(20, 15))
    plt.xlabel("step", fontsize=32)
    plt.ylabel("SVO", fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
               labels=['$-\pi$', '$-3\pi/4$', '$-\pi/2$', '$-\pi/4$', '$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$'],
               fontsize=32)

    x = np.linspace(0, n_steps - 1, n_steps)
    for v_id, hf in enumerate(hf_list):
        plot_data = np.array(hf.svo_error_history)
        y, error = plot_data[:n_steps, 0], plot_data[:n_steps, 1]
        if veh_id_list is None:
            name=f'Veh {v_id}'
        else:
            name=f'Veh {veh_id_list[v_id]}'
        color = np.random.rand(3, )  # 'b'
        plt.plot(x, y, c=color, label=name, linewidth=2)
        plt.fill_between(x, y + error, y - error, color=color, alpha=0.2)

    plt.legend(loc='best', fontsize=25, ncol=2)
    plt.savefig(save_dir)
    plt.close()
