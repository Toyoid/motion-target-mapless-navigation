import math
import numpy as np

def trajectory_encoder(traj_predict, robot_pose,
                       goal_dis_min_dis=0.3, goal_dis_scale=1.0, goal_dir_range=math.pi):
    '''
    Compuate relative distance and direction between robot and
    predicted trajectory of motion target
    '''
    # relative distance
    trajectory = np.array(traj_predict)
    delta_x = trajectory[:, 0] - robot_pose[0]
    delta_y = trajectory[:, 1] - robot_pose[1]
    dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
    # scale distance
    scaled_dist = np.where(dist != 0, dist, goal_dis_min_dis)
    scaled_dist = goal_dis_min_dis / scaled_dist
    scaled_dist = np.where(scaled_dist <= 1, scaled_dist * goal_dis_scale, goal_dis_scale)
    # relative direction
    ego_direction = np.arctan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    theta = ego_direction - robot_direction
    direction = np.arctan2(np.sin(theta), np.cos(theta))
    # scale direction
    scaled_dir = direction / goal_dir_range
    # encoded_trajectory = np.column_stack((scaled_dir, scaled_dist))
    encoded_trajectory = np.concatenate((scaled_dir, scaled_dist), axis=0)
    return encoded_trajectory


def robot_2_goal_dis_dir(goal_pos, robot_pose):
    """
    Compute Relative Distance and Direction between robot and goal
    :param robot_pose: robot pose
    :param goal_pos: goal position
    :return: distance, direction
    """
    delta_x = goal_pos[0] - robot_pose[0]
    delta_y = goal_pos[1] - robot_pose[1]
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

    ego_direction = np.arctan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    theta = ego_direction - robot_direction
    direction = np.arctan2(np.sin(theta), np.cos(theta))
    return distance, direction


traj = [[0, 0], [-1, 1], [-2, 2], [-2, 0]]
robot_pose = [0, 0, math.pi * (7 / 2)]
encoded_traj = trajectory_encoder(traj, robot_pose)
# print(relative_traj)
print(encoded_traj, type(encoded_traj))
state = np.zeros(2 + 2 * 9)
print(np.concatenate((encoded_traj, state), axis=0))


dist, theta = robot_2_goal_dis_dir([2,0], robot_pose)
print(dist, type(dist))
print(theta, type(theta))

print('----------')
goal_dis_min_dis = 0.3
goal_dis_scale = 1
tmp_goal_dis = 1.4142
print(tmp_goal_dis)

if tmp_goal_dis == 0:
    tmp_goal_dis = goal_dis_scale
else:
    tmp_goal_dis = goal_dis_min_dis / tmp_goal_dis
    if tmp_goal_dis > 1:
        tmp_goal_dis = 1
    tmp_goal_dis = tmp_goal_dis * goal_dis_scale
print(tmp_goal_dis)