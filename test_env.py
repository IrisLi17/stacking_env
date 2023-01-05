from bullet_envs.utils.make_vec_env import make_vec_env
import torch
import numpy as np

if __name__ == "__main__":
    env_id = "BulletDrawerState-v1"
    num_worker = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render = True
    env = make_vec_env(env_id, num_worker, device, kwargs={
        "reward_type": "dense", "use_gpu_render": render and torch.cuda.is_available(), "render_goal": render, "obj_task_ratio": 1.0
    })
    done = [False] * num_worker
    random_moved_episode = 0
    total_episode = 0
    if render:
        env.env_method("start_rec", "output", indices=0)
    obs = env.reset()
    initial_obs = obs[0][-8:]
    print("initial obs", initial_obs)
    for i in range(100):
        action = torch.from_numpy(np.concatenate(
            [np.random.randint(0, 3, size=(num_worker, 1)), 
            np.random.uniform(-1, 1, size=(num_worker, 4))], axis=-1
        )).float().to(device)
        obs, reward, done, info = env.step(action)
        print(obs)
        if done[0]:
            total_episode += 1
            print("terminal obs", info[0]["terminal_observation"][-6:])
            if np.linalg.norm(initial_obs[2:5].cpu() - info[0]["terminal_observation"][-6: -3]) > 0.02 and \
                np.linalg.norm(initial_obs[0].cpu() - info[0]["terminal_observation"][-8]) < 0.01:
                random_moved_episode += 1
            print("initial obs", obs[0][-6:])
            initial_obs = obs[0][-8:]
    if render:
        env.env_method("end_rec", indices=0)
    print("moved episode", random_moved_episode, "total episode", total_episode)