import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append('../../')
from evaluation.eval_random_simulation.utility import gen_test_env_poly_list_env

def plot_success_rate(rose, spiral, saw):
    name_list = ['-', 'with\npredictor', '-', 'with\npredictor', '-', 'with\npredictor']

    rose_success = np.concatenate((rose[0, :], [0, 0, 0, 0]), axis=0)
    rose_collision = np.concatenate((rose[1, :], [0, 0, 0, 0]), axis=0)
    rose_timeout = np.concatenate((rose[2, :], [0, 0, 0, 0]), axis=0)
    plt.bar(range(len(name_list)), rose_success, width=0.3, color='deepskyblue', zorder=100)
    plt.bar(range(len(name_list)), rose_timeout, width=0.3, bottom=rose_success, color='limegreen', zorder=100)
    plt.bar(range(len(name_list)), rose_collision, width=0.3, bottom=rose_success+rose_timeout, color='red', zorder=100)

    spiral_success = np.concatenate(([0, 0], spiral[0, :], [0, 0]), axis=0)
    spiral_collision = np.concatenate(([0, 0], spiral[1, :], [0, 0]), axis=0)
    spiral_timeout = np.concatenate(([0, 0], spiral[2, :], [0, 0]), axis=0)
    plt.bar(range(len(name_list)), spiral_success, width=0.3, color='dodgerblue', zorder=100, label='Success')
    plt.bar(range(len(name_list)), spiral_timeout, width=0.3, bottom=spiral_success, color='limegreen', zorder=100)
    plt.bar(range(len(name_list)), spiral_collision, width=0.3, bottom=spiral_success+spiral_timeout, color='red', zorder=100)

    saw_success = np.concatenate(([0, 0, 0, 0], saw[0, :]), axis=0)
    saw_collision = np.concatenate(([0, 0, 0, 0], saw[1, :]), axis=0)
    saw_timeout = np.concatenate(([0, 0, 0, 0], saw[2, :]), axis=0)
    plt.bar(range(len(name_list)), saw_success, width=0.3, color='royalblue', zorder=100)
    plt.bar(range(len(name_list)), saw_timeout, width=0.3, bottom=saw_success, label='Timeout', color='limegreen', zorder=100)
    plt.bar(range(len(name_list)), saw_collision, width=0.3, bottom=saw_success+saw_timeout, tick_label=name_list, label='Collision', color='red', zorder=100)

    plt.ylim((0, 1.005))
    plt.grid(axis='y', linestyle=':', color='grey', zorder=0)
    plt.legend(bbox_to_anchor=(1, 1), loc='lower right', ncol=3)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

def plot_robot_goal_paths(data1, data2, poly_list, env_size=((-10, 10), (-10, 10))):
    """
    Plot Robot Path and Goal Path from experiment
    :param path: robot path
    :param goal_track: goal's moving trajectory
    :param final_state: final states
    :param poly_list: obstacle poly list
    """
    path1 = data1["path"]
    goal_track1 = data1["goal_track"]
    path2 = data2["path"]
    goal_track2 = data2["goal_track"]
    final_state2 = data2["final_state"]
    '''
    ROSE: k = 5, 11, 17
    SPIRAL: k = 1, 8, 13
    SAW: k = 2, 14, 16
    '''
    k = 19
    robot_p1 = path1[k]
    robot_p1_x = [robot_p1[num][0] for num in range(len(robot_p1))]
    robot_p1_y = [robot_p1[num][1] for num in range(len(robot_p1))]
    goal_p1 = goal_track1[k]
    del(goal_p1[0])
    goal_p1_x = [goal_p1[num][0] for num in range(len(goal_p1))]
    goal_p1_y = [goal_p1[num][1] for num in range(len(goal_p1))]

    robot_p2 = path2[k]
    robot_p2_x = [robot_p2[num][0] for num in range(len(robot_p2))]
    robot_p2_y = [robot_p2[num][1] for num in range(len(robot_p2))]
    goal_p2 = goal_track2[k]
    del(goal_p2[0])
    goal_p2_x = [goal_p2[num][0] for num in range(len(goal_p2))]
    goal_p2_y = [goal_p2[num][1] for num in range(len(goal_p2))]

    if final_state2[k] != 1:
        print("Wrong Final State Value ...")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        # plot obstacles
        for obs in poly_list:
            x = [obs[num][0] for num in range(len(obs))]
            y = [obs[num][1] for num in range(len(obs))]
            x.append(obs[0][0])
            y.append(obs[0][1])
            ax[0].plot(x, y, 'k-')
            ax[1].plot(x, y, 'k-')
        # plot paths
        ax[0].plot(robot_p1_x, robot_p1_y, color='#4169E1', linestyle='-', lw=1)
        ax[0].scatter(robot_p1_x[0], robot_p1_y[0], color='b', s=90)
        ax[0].scatter(robot_p1_x[-1], robot_p1_y[-1], color='b', s=90)
        ax[0].plot(goal_p1_x, goal_p1_y, color='r', linestyle='-', lw=1)
        ax[0].scatter(goal_p1_x[0], goal_p1_y[0], color='r', s=90)
        target = patches.RegularPolygon((goal_p1_x[-1], goal_p1_y[-1]), 4, 0.2 * np.sqrt(2), color='red')
        ax[0].add_patch(target)
        ax[0].set_xlim(env_size[0])
        ax[0].set_ylim(env_size[1])
        ax[0].set_aspect('equal', 'box')
        ax[0].set_title("Successful Navigation Routes Without Predictor")

        ax[1].plot(robot_p2_x, robot_p2_y, color='#4169E1', linestyle='-', lw=1)
        ax[1].scatter(robot_p2_x[0], robot_p2_y[0], color='b', s=90)
        ax[1].scatter(robot_p2_x[-1], robot_p2_y[-1], color='b', s=90)
        ax[1].plot(goal_p2_x, goal_p2_y, color='r', linestyle='-', lw=1)
        ax[1].scatter(goal_p2_x[0], goal_p2_y[0], color='r', s=90)
        target = patches.RegularPolygon((goal_p2_x[-1], goal_p2_y[-1]), 4, 0.2 * np.sqrt(2), color='red')
        ax[1].add_patch(target)
        ax[1].set_xlim(env_size[0])
        ax[1].set_ylim(env_size[1])
        ax[1].set_aspect('equal', 'box')
        ax[1].set_title("Successful Navigation Routes With Predictor")

if __name__ == "__main__":
    # plot success rate bar
    rose_rate_data = np.array([[0.825, 0.865],
                               [0.07, 0.09],
                               [0.105, 0.045]])
    spiral_rate_data = np.array([[0.67, 0.865],
                                 [0.08, 0.055],
                                 [0.25, 0.08]])
    saw_rate_data = np.array([[0.785, 0.83],
                              [0.145, 0.115],
                              [0.07, 0.055]])
    # plot_success_rate(rose_rate_data, spiral_rate_data, saw_rate_data)

    # plot navigation routes
    # trajectory_name = 'rose'
    trajectory_name = 'spiral'
    # trajectory_name = 'saw'
    FILE_NAME1 = trajectory_name + '_ddpg_0_19.p'
    FILE_NAME2 = trajectory_name + '_predict_ddpg_0_199.p'

    run_data1 = pickle.load(open('../record_data/' + FILE_NAME1, 'rb'))
    run_data2 = pickle.load(open('../record_data/' + FILE_NAME2, 'rb'))
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    plot_robot_goal_paths(run_data1, run_data2, raw_poly_list)
    plt.show()