from bullet_envs.utils.make_vec_env import make_vec_env
import torch
import numpy as np
import pickle


def main():
    desired_timestep = 100_000
    num_workers = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    create_env_kwargs = dict(
        kwargs=dict(
            n_object=6, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
            n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", use_gpu_render=True,
        )
    )
    env = make_vec_env(
        "BulletPixelStack-v1", 
        num_workers, device, log_dir=None, **create_env_kwargs
    )
    print(env.observation_space, env.action_space)
    traj_cache = [dict(img=[], robot=[], state=[], action=[]) for _ in range(num_workers)]
    total_timestep = 0
    all_demos = dict(obs=[], action=[])
    all_tasks = []
    obs = env.reset()
    # print(obs[0, :768], obs[0, 768: 768 + 7], obs[0, 768 + 7: 768 * 2 + 7], obs[0, 768 * 2 + 7:].reshape(2, -1))
    # exit()
    # check obs
    done = [False for _ in range(num_workers)]
    for i in range(num_workers):
        traj_cache[i]["img"].append(obs[i][:768])
        traj_cache[i]["robot"].append(obs[i][768: 768 + 7])
        traj_cache[i]["state"].append(obs[i][768 * 2 + 7:].reshape(2, -1)[0])
    while total_timestep < desired_timestep:
        action = torch.cat(env.env_method("act"), dim=0).to(device)
        # print(action)
        obs, reward, done, info = env.step(action)
        for i in range(num_workers):
            img = obs[i][:768]
            robot_obs = obs[i][768: 768 + 7]
            state = obs[i][768 * 2 + 7:].reshape(2, -1)[0]
            traj_cache[i]["img"].append(img)
            traj_cache[i]["robot"].append(robot_obs)
            traj_cache[i]["state"].append(state)
            traj_cache[i]["action"].append(action[i])
            # traj_cache[i]["reward"].append(reward[i])
            # traj_cache[i]["done"].append(done[i])
            if done[i]:
                if info[i]["is_success"]:
                    terminate_obs = info[i]["terminal_observation"]
                    traj_cache[i]["img"][-1] = terminate_obs[:768]
                    traj_cache[i]["robot"][-1] = terminate_obs[768: 768 + 7]
                    traj_cache[i]["state"][-1] = terminate_obs[768 * 2 + 7:].reshape(2, -1)[0]
                    if not torch.all(traj_cache[i]["action"][0] == -1):
                        # TODO: store all tasks
                        for _s_idx in range(len(traj_cache[i]["img"]) - 1):
                            _e_idx = _s_idx + 1
                            _cur_img = traj_cache[i]["img"][_s_idx]
                            _cur_robot = traj_cache[i]["robot"][_s_idx]
                            _cur_state = traj_cache[i]["state"][_s_idx]
                            _goal_img = traj_cache[i]["img"][_e_idx]
                            _goal_state = traj_cache[i]["state"][_e_idx]
                            relabeled_obs = torch.cat([_cur_img, _cur_robot, _goal_img, _cur_state, _goal_state])
                            # assert np.all(np.array(traj_cache[i]["done"][:-1]) == False)
                            # assert np.all(np.array(traj_cache[i]["reward"][:-1]) == 0), (traj_cache[i]["reward"])
                            # assert traj_cache[i]["reward"][-1] == 1, (traj_cache[i]["reward"][-1])
                            all_demos["obs"].append(relabeled_obs.detach().cpu().numpy())
                            assert not torch.all(traj_cache[i]["action"][_s_idx] == -1)
                            all_demos["action"].append(traj_cache[i]["action"][_s_idx].cpu().numpy())
                            total_timestep += 1
                            if total_timestep % 100 == 0:
                                print(total_timestep)
                traj_cache[i] = dict(
                    img=[obs[i][:768]], robot=[obs[i][768: 768 + 7]], 
                    state=[obs[i][768 * 2 + 7:].reshape(2, -1)[0]], action=[])
    all_demos["obs"] = np.stack(all_demos["obs"], axis=0)
    all_demos["action"] = np.stack(all_demos["action"], axis=0)
    # all_demos["terminate_obs"] = np.stack(all_demos["terminate_obs"], axis=0)
    # all_demos["boundary"] = np.array(all_demos["boundary"])
    with open("warmup_dataset_stacking.pkl", "wb") as f:
        pickle.dump(all_demos, f)


if __name__ == "__main__":
    main()
