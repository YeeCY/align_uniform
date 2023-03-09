# import gin
import os
import numpy as np
import pickle as pkl
import gym
import imageio

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
)


def main():
    num_trajs = 10
    traj_len = 200
    # dataset = [
    #     [(0, 3), (1, 3), (2, 3), (3, 4), (5, 4), (9, 1), (8, 1), (7, 1), (6, 0)],
    #     [(2, 1), (1, 4), (4, 4), (7, 3), (8, 3)],
    #     [(7, 2), (4, 2), (1, 1), (0, 3), (1, 3)]
    # ]
    dataset = []

    # env = GridEnv()
    assembly_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["assembly-v2-goal-observable"]
    basketball_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["basketball-v2-goal-observable"]
    bin_picking_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["bin-picking-v2-goal-observable"]
    box_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["box-close-v2-goal-observable"]
    button_press_topdown_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-topdown-v2-goal-observable"]
    coffee_button_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["coffee-button-v2-goal-observable"]
    dial_turn_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["dial-turn-v2-goal-observable"]
    door_open_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
    door_unlock_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-lock-v2-goal-observable"]
    drawer_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]

    envs_cls = [assembly_cls, basketball_cls, bin_picking_cls, box_close_cls, button_press_topdown_cls,
                coffee_button_cls, dial_turn_cls, door_open_cls, door_unlock_cls, drawer_close_cls]
    policies = [SawyerAssemblyV2Policy(), SawyerBasketballV2Policy(), SawyerBinPickingV2Policy(),
                SawyerBoxCloseV2Policy(), SawyerButtonPressTopdownV2Policy(), SawyerCoffeeButtonV2Policy(),
                SawyerDialTurnV2Policy(), SawyerDoorOpenV2Policy(), SawyerDoorUnlockV2Policy(),
                SawyerDrawerCloseV2Policy()]

    # random policy
    # policy = np.ones(env.num_actions) / env.num_actions
    # imgs = []
    for traj_idx in range(num_trajs):
        env_idx = traj_idx % 10
        env = envs_cls[env_idx](seed=traj_idx)
        policy = policies[env_idx]
        print("env: {}".format(env.__class__))
        print("traj: {}".format(traj_idx))

        traj = []
        # state = env.reset()
        obs = env.reset()

        # random_action_start_step = np.random.randint(1, traj_len)
        # random_action_start_steps = np.array([10, 20, 40, 60, 80, 100, 120, 200 ])
        random_action = (np.random.rand() >= 0.0)

        for t in range(traj_len):
            # action = np.random.choice(env.num_actions, p=policy)
            # action = env.action_space.sample()
            # if t < np.random.choice(random_action_start_steps):
            #     action = policy.get_action(obs)
            # else:
            #     action = env.action_space.sample()
            if random_action:
                action = env.action_space.sample()
            else:
                action = policy.get_action(obs)
            # action = policy.get_action(obs)
            # img = env.render(resolution=(32, 32), offscreen=True)

            # cameras: corner3, corner, corner2, topview, gripperPOV, behindGripper
            # H x W x C
            # img = env.render(resolution=(48, 48), offscreen=True, camera_name='topview')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='topview')
            img = env.render(resolution=(48, 48), offscreen=True, camera_name='corner')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='corner')

            # action = np.random.randint(low=0, high=5)
            # traj.append((obs, action))
            # imgs.append(img)
            traj.append((img.reshape(-1), action))

            obs, reward, done, info = env.step(action)

        dataset.append(traj)

    # Save Video
    # video_dir = os.path.abspath("./videos")
    # video_path = os.path.join(video_dir, "metaworld_10_tasks_random_action_video.mp4")
    # # imgs = np.asarray(imgs)
    # imageio.mimsave(video_path, imgs, fps=20)
    # print("Save video to: {}".format(video_path))

    dataset_dir = os.path.abspath("data")
    # dataset_path = os.path.join(dataset_dir, "dataset.pkl")
    dataset_path = os.path.join(dataset_dir, "metaworld_10_tasks_img_random.pkl")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_mixed_img.pkl")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_random_img.pkl")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(dataset_path, "wb+") as f:
        pkl.dump(dataset, f)
    print("Dataset saved to: {}".format(dataset_path))


if __name__ == "__main__":
    main()
