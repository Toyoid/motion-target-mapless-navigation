import rospy
import sys
sys.path.append('../../')
from evaluation.xxx_eval_simulation.ddpg_move_eval import DDPGMoveEval
from evaluation.xxx_eval_simulation.utils import *


def evaluate_xxx(pos_start=0, pos_end=199, model_name='ddpg', save_dir='../saved_model/',
                 is_save_result=False, use_cuda=True):
    """
    Evaluate XXX in Simulated Environment

    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    """
    rospy.init_node('navi_eval')
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    rand_paths_robot_list = pickle.load(open('eval_paths_robot_pose.p', 'rb'))
    robot_init_list = rand_paths_robot_list[0][:]
    target_paths_list = rand_paths_robot_list[1][:]

    net_dir = save_dir + model_name + '.pt'
    # net_dir = '../../training/save_model_weights/save_controller/DDPG_ctrl_actor_network_s23.pt'
    controller_net = load_actor_net(net_dir)
    evalution = DDPGMoveEval(controller_net, robot_init_list, target_paths_list, poly_list,
                               max_steps=1000, action_rand=0.01, use_cuda=use_cuda)
    data = evalution.run_ros()
    if is_save_result:
        pickle.dump(data,
                    open('../record_data/' + 'ddpg_move_policy_' + str(pos_start) + '_' + str(pos_end) + '.p', 'wb+'))
    print(str(model_name) + " Evaluation Finished ...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False

    MODEL_NAME = 'ddpg'

    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    evaluate_xxx(use_cuda=USE_CUDA,
                 model_name=MODEL_NAME,
                 is_save_result=SAVE_RESULT)
