# import gin
import os
import numpy as np
import pickle as pkl
import gym
import imageio
import importlib

# from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
# from metaworld.policies import (
#     SawyerAssemblyV2Policy,
#     SawyerBasketballV2Policy,
#     SawyerBinPickingV2Policy,
#     SawyerBoxCloseV2Policy,
#     SawyerButtonPressTopdownV2Policy,
#     SawyerCoffeeButtonV2Policy,
#     SawyerDialTurnV2Policy,
#     SawyerDoorOpenV2Policy,
#     SawyerDoorUnlockV2Policy,
#     SawyerDrawerCloseV2Policy,
# )
# from metaworld.policies import __all__
from metaworld.envs.mujoco.env_dict import MT50_V2, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import metaworld.policies


def create_env_and_policy(env_name, seed=None):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
    env = env_cls(seed=seed)

    # def initialize(env, seed=None):
    #     if seed is not None:
    #         st0 = np.random.get_state()
    #         np.random.seed(seed)
    #     # super(type(env), env).__init__()
    #     # env._partially_observable = True
    #     env._freeze_rand_vec = False
    #     # env._set_task_called = True
    #     env.reset()
    #     env._freeze_rand_vec = True
    #     if seed is not None:
    #         env.seed(seed)
    #         np.random.set_state(st0)

    # initialize(env, seed=seed)

    policy_name = 'Sawyer' + env_name.title().replace('-', '').replace('PegInsert', 'PegInsertion') + 'Policy'
    assert policy_name in metaworld.policies.__all__
    policy_cls = getattr(importlib.import_module("metaworld.policies"), policy_name)
    policy = policy_cls()

    return env, policy


def main():
    # training dataset
    num_trajs = 1000
    traj_len = 200
    # dataset = [
    #     [(0, 3), (1, 3), (2, 3), (3, 4), (5, 4), (9, 1), (8, 1), (7, 1), (6, 0)],
    #     [(2, 1), (1, 4), (4, 4), (7, 3), (8, 3)],
    #     [(7, 2), (4, 2), (1, 1), (0, 3), (1, 3)]
    # ]
    train_dataset = []

    # env = GridEnv()
    # assembly_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["assembly-v2-goal-observable"]
    # basketball_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["basketball-v2-goal-observable"]
    # bin_picking_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["bin-picking-v2-goal-observable"]
    # box_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["box-close-v2-goal-observable"]
    # button_press_topdown_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-topdown-v2-goal-observable"]
    # coffee_button_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-button-v2-goal-observable"]
    # dial_turn_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["dial-turn-v2-goal-observable"]
    # door_open_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
    # door_unlock_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-lock-v2-goal-observable"]
    # drawer_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]
    #
    # envs_cls = [assembly_cls, basketball_cls, bin_picking_cls, box_close_cls, button_press_topdown_cls,
    #             coffee_button_cls, dial_turn_cls, door_open_cls, door_unlock_cls, drawer_close_cls]
    # policies = [SawyerAssemblyV2Policy(), SawyerBasketballV2Policy(), SawyerBinPickingV2Policy(),
    #             SawyerBoxCloseV2Policy(), SawyerButtonPressTopdownV2Policy(), SawyerCoffeeButtonV2Policy(),
    #             SawyerDialTurnV2Policy(), SawyerDoorOpenV2Policy(), SawyerDoorUnlockV2Policy(),
    #             SawyerDrawerCloseV2Policy()]

    # random policy
    # policy = np.ones(env.num_actions) / env.num_actions
    imgs = []
    for traj_idx in range(num_trajs):
        env, policy = create_env_and_policy(list(MT50_V2.keys())[traj_idx % 50],
                                            seed=traj_idx)
        print("env: {}".format(env.__class__))
        print("traj: {}".format(traj_idx))

        traj = []
        # state = env.reset()
        obs = env.reset()
        for _ in range(traj_len):
            # action = np.random.choice(env.num_actions, p=policy)
            # action = env.action_space.sample()
            # if traj_idx % 2 == 0:
            #     action = env.action_space.sample()
            # else:
            #     action = policy.get_action(obs)
            action = policy.get_action(obs)
            # img = env.render(resolution=(32, 32), offscreen=True)

            # cameras: corner3, corner, corner2, topview, gripperPOV, behindGripper
            # H x W x C
            # img = env.render(resolution=(48, 48), offscreen=True, camera_name='topview')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='topview')
            img = env.render(resolution=(48, 48), offscreen=True, camera_name='corner')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='corner')

            # action = np.random.randint(low=0, high=5)
            # traj.append((obs, action))
            imgs.append(img)
            traj.append((img.reshape(-1), action))

            obs, reward, done, info = env.step(action)

        train_dataset.append(traj)

    # # Save Video
    # video_dir = os.path.abspath("./videos")
    # video_path = os.path.join(video_dir, "metaworld_50_tasks_video.mp4")
    # # imgs = np.asarray(imgs)
    # imageio.mimsave(video_path, imgs, fps=20)
    # print("Save video to: {}".format(video_path))

    dataset_dir = os.path.abspath("data")
    # dataset_path = os.path.join(dataset_dir, "dataset.pkl")
    dataset_path = os.path.join(dataset_dir, "metaworld_50_tasks_img_train.pkl")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_mixed_img.pkl")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_random_img.pkl")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(dataset_path, "wb+") as f:
        pkl.dump(train_dataset, f)
    print("Dataset saved to: {}".format(dataset_path))

    # validation dataset
    num_trajs = 250
    traj_len = 200
    val_dataset = []

    for traj_idx in range(num_trajs):
        env, policy = create_env_and_policy(list(MT50_V2.keys())[traj_idx % 50],
                                            seed=traj_idx)
        print("env: {}".format(env.__class__))
        print("traj: {}".format(traj_idx))

        traj = []
        obs = env.reset()
        for _ in range(traj_len):
            action = policy.get_action(obs)

            # cameras: corner3, corner, corner2, topview, gripperPOV, behindGripper
            # H x W x C
            # img = env.render(resolution=(48, 48), offscreen=True, camera_name='topview')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='topview')
            img = env.render(resolution=(48, 48), offscreen=True, camera_name='corner')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='corner')

            imgs.append(img)
            traj.append((img.reshape(-1), action))

            obs, reward, done, info = env.step(action)

        val_dataset.append(traj)

    dataset_dir = os.path.abspath("data")
    dataset_path = os.path.join(dataset_dir, "metaworld_50_tasks_img_val.pkl")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(dataset_path, "wb+") as f:
        pkl.dump(val_dataset, f)
    print("Dataset saved to: {}".format(dataset_path))


if __name__ == "__main__":
    main()
