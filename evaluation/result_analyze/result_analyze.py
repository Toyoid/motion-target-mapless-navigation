import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
import sys
sys.path.append('../../')
from evaluation.xxx_eval_simulation.utils import gen_test_env_poly_list_env


def analyze_run(data):
    """
    Analyze success rate, path distance, path time, and path avg spd
    :param data: run_data
    :return: state_list, path_dis, path_time, path_spd
    """
    run_num = len(data["final_state"])
    state_list = [0, 0, 0]
    path_dis = np.zeros(run_num)
    path_time = np.zeros(run_num)
    path_spd = np.zeros(run_num)
    for r in range(run_num):
        if data["final_state"][r] == 1 or data["final_state"][r] == 3:
            tmp_overll_path_dis = 0
            for d in range(len(data["robot_path"][r]) - 1):
                rob_pos = data["robot_path"][r][d]
                next_rob_pos = data["robot_path"][r][d + 1]
                tmp_dis = np.sqrt((next_rob_pos[0] - rob_pos[0]) ** 2 + (next_rob_pos[1] - rob_pos[1]) ** 2)
                tmp_overll_path_dis += tmp_dis
            path_dis[r] = tmp_overll_path_dis
            path_time[r] = data["time"][r]
            path_spd[r] = path_dis[r] / path_time[r]
        if data["final_state"][r] == 1:
            state_list[0] += 1
        elif data["final_state"][r] == 2:
            state_list[1] += 1
        elif data["final_state"][r] == 3:
            print(r)
            state_list[2] += 1
        else:
            print("FINAL STATE TYPE ERROR ...")
    return state_list, path_dis, path_time, path_spd


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
    """
    robot_path1 = data1["robot_path"]
    target_path1 = data1["target_path"]
    final_state1 = data1["final_state"]
    robot_path2 = data2["robot_path"]
    target_path2 = data2["target_path"]
    final_state2 = data2["final_state"]

    # good comparison(greedy r2): k = 11, 55, 144, 188, 190
    # good comparison(param3): k = 7, 11, 33(!), 35, 38, 46(!), 49(!), 159(!)
    k = 190  # 14 144 190
    robot_p1 = robot_path1[k]
    robot_p1_x = [robot_p1[num][0] for num in range(len(robot_p1))]
    robot_p1_y = [robot_p1[num][1] for num in range(len(robot_p1))]
    target_p1 = target_path1[k]
    del(target_p1[0])
    target_p1_x = [target_p1[num][0] for num in range(len(target_p1))]
    target_p1_y = [target_p1[num][1] for num in range(len(target_p1))]

    robot_p2 = robot_path2[k]
    robot_p2_x = [robot_p2[num][0] for num in range(len(robot_p2))]
    robot_p2_y = [robot_p2[num][1] for num in range(len(robot_p2))]
    target_p2 = target_path2[k]
    del(target_p2[0])
    target_p2_x = [target_p2[num][0] for num in range(len(target_p2))]
    target_p2_y = [target_p2[num][1] for num in range(len(target_p2))]

    if final_state2[k] == -1:
        print("Wrong Final State Value ...")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        # plot obstacles
        for i, obs in enumerate(poly_list):
            if i > 0:
                p = gpd.GeoSeries(obs)
                p.plot(color='k', ax=ax[0])
                p.plot(color='k', ax=ax[1])
            else:
                ax[0].plot(*obs.xy, 'k-')
                ax[1].plot(*obs.xy, 'k-')

        # plot paths
        ax[0].plot(robot_p1_x, robot_p1_y, color='#4169E1', linestyle='-', lw=1)
        ax[0].scatter(robot_p1_x[0], robot_p1_y[0], color='b', s=90)
        ax[0].scatter(robot_p1_x[-1], robot_p1_y[-1], color='b', s=90)
        ax[0].plot(target_p1_x, target_p1_y, color='r', linestyle='-', lw=1)
        ax[0].scatter(target_p1_x[0], target_p1_y[0], color='r', s=90)
        target = patches.RegularPolygon((target_p1_x[-1], target_p1_y[-1]), 4, 0.2 * np.sqrt(2), color='red')
        ax[0].add_patch(target)
        ax[0].set_xlim(env_size[0])
        ax[0].set_ylim(env_size[1])
        ax[0].set_aspect('equal', 'box')
        ax[0].set_title("Greedy Meta-controller Navigation Routes")
        # ax[0].set_title("DDPG Navigation Routes")

        ax[1].plot(robot_p2_x, robot_p2_y, color='#4169E1', linestyle='-', lw=1)
        ax[1].scatter(robot_p2_x[0], robot_p2_y[0], color='b', s=90)
        ax[1].scatter(robot_p2_x[-1], robot_p2_y[-1], color='b', s=90)
        ax[1].plot(target_p2_x, target_p2_y, color='r', linestyle='-', lw=1)
        ax[1].scatter(target_p2_x[0], target_p2_y[0], color='r', s=90)
        target = patches.RegularPolygon((target_p2_x[-1], target_p2_y[-1]), 4, 0.2 * np.sqrt(2), color='red')
        ax[1].add_patch(target)
        ax[1].set_xlim(env_size[0])
        ax[1].set_ylim(env_size[1])
        ax[1].set_aspect('equal', 'box')
        ax[1].set_title("HiDDPG Navigation Routes")

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
    # MODEL_NAME1 = 'greedy_meta_ddpg_r2'
    MODEL_NAME1 = 'ddpg_move_policy'
    # MODEL_NAME1 = 'xxx_randomtrain_ctrlstep5_distreward'
    MODEL_NAME2 = 'xxx_hyperparam2_r2eval'
    # MODEL_NAME2 = 'xxx_hyperparam3_r2eval'
    # MODEL_NAME2 = 'xxx_hyperparam3_try_againkf'
    FILE_NAME1 = MODEL_NAME1 + '_0_199.p'
    FILE_NAME2 = MODEL_NAME2 + '_0_199.p'

    run_data1 = pickle.load(open('../record_data/' + FILE_NAME1, 'rb'))
    termination, path_dist, path_time, path_spd = analyze_run(run_data1)
    print(MODEL_NAME1 + " random simulation results:")
    print("Success: ", termination[0], " Collision: ", termination[1], " Overtime: ", termination[2])
    print("Average Path Distance of Success and Overtime Routes: ", np.mean(path_dist[path_dist > 0]), ' m')
    print("Average Path Time of Success and Overtime Routes: ", np.mean(path_time[path_dist > 0]), ' s')
    print("Average Path Speed of Success and Overtime Routes: ", np.mean(path_spd[path_dist > 0]), ' m/s')
    print('\n')

    run_data2 = pickle.load(open('../record_data/' + FILE_NAME2, 'rb'))
    termination, path_dist, path_time, path_spd = analyze_run(run_data2)
    print(MODEL_NAME2 + " random simulation results:")
    print("Success: ", termination[0], " Collision: ", termination[1], " Overtime: ", termination[2])
    print("Average Path Distance of Success and Overtime Routes: ", np.mean(path_dist[path_dist > 0]), ' m')
    print("Average Path Time of Success and Overtime Routes: ", np.mean(path_time[path_dist > 0]), ' s')
    print("Average Path Speed of Success and Overtime Routes: ", np.mean(path_spd[path_dist > 0]), ' m/s')
    poly_list, _ = gen_test_env_poly_list_env()
    plot_robot_goal_paths(run_data1, run_data2, poly_list)
    plt.show()