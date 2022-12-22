from bullet_envs.utils.make_vec_env import make_vec_env
import torch
import numpy as np
import pickle

def main():
    desired_timestep = 100_000
    num_workers = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_env_kwargs = dict(
        kwargs=dict(reward_type="sparse", view_mode="third"),
    )
    env = make_vec_env("BulletDrawer-v1", num_workers, device, log_dir=None, **create_env_kwargs)

    traj_cache = [dict(obs=[], action=[], reward=[], done=[]) for _ in range(num_workers)]
    total_timestep = 0
    all_demos = dict(obs=[], action=[], terminate_obs=[], boundary=[])
    obs = env.reset()
    done = [False for _ in range(num_workers)]
    for i in range(num_workers):
        traj_cache[i]["obs"].append(obs[i])
    while total_timestep < desired_timestep:
        action = env.env_method("oracle_agent")
        action = torch.from_numpy(np.array(action)).to(device)
        obs, reward, done, info = env.step(action)
        for i in range(num_workers):
            traj_cache[i]["obs"].append(obs[i])
            traj_cache[i]["action"].append(action[i])
            traj_cache[i]["reward"].append(reward[i])
            traj_cache[i]["done"].append(done[i])
            if done[i]:
                if info[i]["is_success"]:
                    # make sure is terminate at the first success. discard the episodes with length 1
                    if len(traj_cache[i]["action"]) > 1:
                        assert np.all(np.array(traj_cache[i]["done"][:-1]) == False)
                        assert np.all(np.array(traj_cache[i]["reward"][:-1]) == 0)
                        assert traj_cache[i]["reward"][-1] == 1
                        all_demos["obs"].append(torch.stack(traj_cache[i]["obs"][:-1], dim=0).detach().cpu().numpy())
                        all_demos["action"].append(torch.stack(traj_cache[i]["action"], dim=0).cpu().numpy())
                        all_demos["terminate_obs"].append(traj_cache[i]["obs"][-1].detach().cpu().numpy())
                        all_demos["boundary"].append(total_timestep)
                        total_timestep += len(traj_cache[i]["action"])
                        print(total_timestep)
                traj_cache[i] = dict(obs=[obs[i]], action=[], reward=[], done=[])
    all_demos["obs"] = np.concatenate(all_demos["obs"], axis=0)
    all_demos["action"] = np.concatenate(all_demos["action"], axis=0)
    all_demos["terminate_obs"] = np.stack(all_demos["terminate_obs"], axis=0)
    with open("warmup_dataset.pkl", "wb") as f:
        pickle.dump(all_demos, f)

if __name__ == "__main__":
    main()
