from bullet_envs.utils.make_vec_env import make_vec_env
import torch
import torch.nn as nn
import numpy as np
import pickle

def main():
    desired_timestep = 100_000
    num_workers = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_env_kwargs = dict(
        kwargs=dict(reward_type="sparse", use_gpu_render=False, obj_task_ratio=1.0),
    )
    env = make_vec_env("BulletDrawerState-v1", num_workers, device, log_dir=None, **create_env_kwargs)

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
                    if len(traj_cache[i]["action"]) > 1 and (not info[i]["is_goal_move_drawer"]):
                        assert np.all(np.array(traj_cache[i]["done"][:-1]) == False)
                        assert np.all(np.array(traj_cache[i]["reward"][:-1]) == 0)
                        assert traj_cache[i]["reward"][-1] == 1
                        all_demos["obs"].append(torch.stack(traj_cache[i]["obs"][:-1], dim=0).detach().cpu().numpy())
                        all_demos["action"].append(torch.stack(traj_cache[i]["action"], dim=0).cpu().numpy())
                        all_demos["terminate_obs"].append(info[i]["terminal_observation"].detach().cpu().numpy())
                        all_demos["boundary"].append(total_timestep)
                        total_timestep += len(traj_cache[i]["action"])
                        print(total_timestep)
                traj_cache[i] = dict(obs=[obs[i]], action=[], reward=[], done=[])
    all_demos["obs"] = np.concatenate(all_demos["obs"], axis=0)
    all_demos["action"] = np.concatenate(all_demos["action"], axis=0)
    all_demos["terminate_obs"] = np.stack(all_demos["terminate_obs"], axis=0)
    with open("warmup_dataset.pkl", "wb") as f:
        pickle.dump(all_demos, f)

def predict_privilege():
    class NNPredictor(nn.Module):
        def __init__(self, feat_dim, pred_dim) -> None:
            super().__init__()
            self.feat_dim = feat_dim
            self.pred_dim = pred_dim
            self.layer = nn.Sequential(
                nn.Linear(self.feat_dim, 256), nn.ReLU(),
                nn.Linear(256, self.pred_dim), 
            )
        
        def forward(self, im_feat):
            pred = self.layer.forward(im_feat)
            return pred
        
        def get_loss(self, im_feat, gt):
            pred = self.forward(im_feat)
            loss = torch.mean((pred - gt) ** 2)
            return loss
    nn_predictor = NNPredictor(768, 4)
    optimizer = torch.optim.Adam(nn_predictor.parameters())
    with open("warmup_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    obs = dataset["obs"]
    im_feat = obs[:, :768]
    state = obs[:, 768 * 2 + 14:]
    state = np.concatenate([state[:, 0:1], state[:, 2:5]], axis=-1)    
    goal_feat = dataset["terminate_obs"][:, 768 + 14: 768 * 2 + 14]
    goal_state = dataset["terminate_obs"][:, 768 * 2 + 14:]
    goal_state = np.concatenate([goal_state[:, 0:1], goal_state[:, 2:5]], axis=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_input = torch.from_numpy(np.concatenate([im_feat, goal_feat], axis=0)).float().to(device)
    train_gt = torch.from_numpy(np.concatenate([state, goal_state], axis=0)).float().to(device)
    nn_predictor.to(device)

    indices = np.arange(train_input.shape[0])
    num_epoch = 30
    batch_size = 32
    from collections import deque
    losses = deque(maxlen=100)
    for i in range(num_epoch):
        np.random.shuffle(indices)
        for j in range(train_input.shape[0] // batch_size):
            mb_indices = indices[j * batch_size: (j + 1) * batch_size]
            mb_input = train_input[mb_indices]
            mb_gt = train_gt[mb_indices]
            loss = nn_predictor.get_loss(mb_input, mb_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i * (train_input.shape[0] // batch_size) + j) % 100 == 0:
                print("Epoch %d, mb %d, loss %f" % (i, j, np.mean(losses)))
        with torch.no_grad():
            pred = nn_predictor.forward(mb_input)
            print("pred", pred - mb_gt)
            

if __name__ == "__main__":
    main()
    # predict_privilege()

# TODO: let's check whether image embeddings can predict object and drawer parameters.
# TODO: if not, let's see how to get it working. Ego view?