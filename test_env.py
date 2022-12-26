from bullet_envs.utils.make_vec_env import make_vec_env
import torch
import numpy as np

if __name__ == "__main__":
    env_id = "BulletDrawerState-v1"
    num_worker = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vec_env(env_id, num_worker, device, kwargs={"reward_type": "dense", "use_gpu_render": False, "obj_task_ratio": 1.0})
    done = [False] * num_worker
    random_moved_episode = 0
    total_episode = 0
    obs = env.reset()
    initial_obs = obs[0][-6:]
    print("initial obs", initial_obs)
    for i in range(500):
        action = torch.from_numpy(np.concatenate(
            [np.random.randint(0, 3, size=(num_worker, 1)), 
            np.random.uniform(-1, 1, size=(num_worker, 4))], axis=-1
        )).float().to(device)
        obs, reward, done, info = env.step(action)
        if done[0]:
            total_episode += 1
            print("terminal obs", info[0]["terminal_observation"][-6:])
            if np.linalg.norm(initial_obs[:3] - info[0]["terminal_observation"][-6: -3]) > 0.02:
                random_moved_episode += 1
            print("initial obs", obs[0][-6:])
            initial_obs = obs[0][-6:]
    print("moved episode", random_moved_episode, "total episode", total_episode)