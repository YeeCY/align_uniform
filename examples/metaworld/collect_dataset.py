# import gin
import os
import numpy as np
import pickle as pkl
import gym
import imageio

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import SawyerDoorOpenV2Policy


def main():
    num_trajs = 200
    traj_len = 200
    # dataset = [
    #     [(0, 3), (1, 3), (2, 3), (3, 4), (5, 4), (9, 1), (8, 1), (7, 1), (6, 0)],
    #     [(2, 1), (1, 4), (4, 4), (7, 3), (8, 3)],
    #     [(7, 2), (4, 2), (1, 1), (0, 3), (1, 3)]
    # ]
    dataset = []

    # env = GridEnv()
    door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]

    env = door_open_goal_observable_cls()
    policy = SawyerDoorOpenV2Policy()

    # random policy
    # policy = np.ones(env.num_actions) / env.num_actions
    # imgs = []
    for traj_idx in range(num_trajs):
        print("traj: {}".format(traj_idx))

        traj = []
        # state = env.reset()
        obs = env.reset()
        for _ in range(traj_len):
            # action = np.random.choice(env.num_actions, p=policy)
            action = env.action_space.sample()
            # if traj_idx % 2 == 0:
            #     action = env.action_space.sample()
            # else:
            #     action = policy.get_action(obs)
            # img = env.render(resolution=(32, 32), offscreen=True)

            # H x W x C
            img = env.render(resolution=(48, 48), offscreen=True, camera_name='topview')
            # img = env.render(resolution=(480, 480), offscreen=True, camera_name='topview')

            # action = np.random.randint(low=0, high=5)
            # traj.append((obs, action))
            # imgs.append(img)
            traj.append((img.reshape(-1), action))

            obs, reward, done, info = env.step(action)

        dataset.append(traj)

    # Save Video
    # video_dir = os.path.abspath("./videos")
    # video_path = os.path.join(video_dir, "metaworld_door_open_v2_video.mp4")
    # # imgs = np.asarray(imgs)
    # imageio.mimsave(video_path, imgs, fps=20)
    # print("Save video to: {}".format(video_path))

    dataset_dir = os.path.abspath("./data")
    # dataset_path = os.path.join(dataset_dir, "dataset.pkl")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_img.pkl")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_mixed_img.pkl")
    dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_random_img.pkl")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(dataset_path, "wb+") as f:
        pkl.dump(dataset, f)
    print("Dataset saved to: {}".format(dataset_path))


if __name__ == "__main__":
    main()
