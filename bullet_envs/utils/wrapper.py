import gym
import gym.spaces
import numpy as np
import torch
from bullet_envs.vec_env.base_vec_env import VecEnvWrapper
from collections import deque


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=0.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class SwitchGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SwitchGoalWrapper, self).__init__(env)
        if not hasattr(self.env.unwrapped, "goals"):
            assert hasattr(self.env.unwrapped, "goal")
            self.env.unwrapped.goals = []

    def set_goals(self, goals):
        if isinstance(goals, tuple):
            goals = list(goals)
        assert isinstance(goals, list)
        self.env.unwrapped.goals = goals
        self.env.unwrapped.goal = self.env.unwrapped.goals.pop(0).copy()
        self.env.unwrapped.visualize_goal(self.env.unwrapped.goal)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            if len(self.env.goals) > 0:
                if info.get("TimeLimit.truncated", False):
                    info["is_success"] = False
                else:
                    self.env.unwrapped.goal = self.env.unwrapped.goals.pop(0).copy()
                    self.env.unwrapped.visualize_goal(self.env.unwrapped.goal)
                    done = False
                    info["is_success"] = False
        return obs, reward, done, info


class ResetWrapper(gym.Wrapper):
    # applied before SwitchGoalWrapper and FlattenDictWrapper
    def __init__(self, env, ratio=0.5):
        super(ResetWrapper, self).__init__(env)
        self.states_to_restart = deque(maxlen=50)  # dicts of state and goal
        self.ratio = ratio
        self.is_restart = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.is_restart = False
        if len(self.states_to_restart) > 0 and self.env.unwrapped.np_random.uniform() < self.ratio:
            state_to_restart = self.states_to_restart.popleft()
            self.env.unwrapped.set_state(state_to_restart["state"])
            self.env.unwrapped.goal = state_to_restart["goal"]
            self.env.unwrapped.sync_attr()
            result = self.env.get_obs()
            self.is_restart = True
        return result

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["is_restart"] = self.is_restart
        return obs, reward, done, info

    def add_state_to_reset(self, state_and_goal: dict):
        self.states_to_restart.append(state_and_goal)


class ScaleRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=1.0):
        super(ScaleRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward /= self.reward_scale
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward / self.reward_scale


class FlexibleTimeLimitWrapper(gym.Wrapper):
    '''
    ONLY applicable to Stacking environment!
    We can set max_episode_steps = None for gym, (so gym.TimeLimitWrapper is not applied),
    then use this class to avoid potential conflict.
    '''
    def __init__(self, env):
        super(FlexibleTimeLimitWrapper, self).__init__(env)
        # self.time_limit = time_limit
        assert 'BulletStack' in env.spec.id
        assert env.spec.max_episode_steps is None
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        time_limit = self.env.unwrapped.n_to_stack * 50 if self.env.unwrapped.n_to_stack > 2 else 100
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= time_limit:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class MVPVecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, observation_space=None, action_space=None, ):
        super().__init__(venv, observation_space, action_space)
        self.device = device
        import mvp
        import torchvision.transforms
        self.mvp_model = mvp.load("vitb-mae-egosoup")
        self.mvp_model.to(self.device)
        self.mvp_model.freeze()
        self.image_transform = torchvision.transforms.Resize(224)
        self.im_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).to(self.device)
        self.im_std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).to(self.device)
        self.mvp_feat_dim = None
        self.robot_state_dim = None
        self.privilege_info_dim = None
        self.reset()
    
    def _normalize_obs(self, obs):
        img = self.image_transform(torch.from_numpy(obs["img"]).float().to(self.device))
        goal = self.image_transform(torch.from_numpy(obs["goal"]).float().to(self.device))
        normed_img = (img / 255.0 - self.im_mean) / self.im_std
        normed_goal = (goal / 255.0 - self.im_mean) / self.im_std
        return normed_img, normed_goal

    def mvp_process_obs(self, obs):
        assert "img" in obs.keys() and "goal" in obs.keys()
        assert "robot_state" in obs.keys() and "privilege_info" in obs.keys()
        # sometimes, we don't have goal images. The correct feature should be set in "mvp_goal_feature"
        precomputed_goal_feat_mask = obs["goal_source"].astype(np.bool).squeeze(axis=-1)
        normed_img, normed_goal = self._normalize_obs(obs)
        with torch.no_grad():
            scene_feat = self.mvp_model.extract_feat(normed_img.float())
            scene_feat = self.mvp_model.forward_norm(scene_feat)
        if np.all(precomputed_goal_feat_mask):
            # No need to forward mvp model, so I write it separately
            goal_feat = torch.from_numpy(obs["goal_feature"]).float().to(self.device)
        else:
            with torch.no_grad():
                goal_feat = self.mvp_model.extract_feat(normed_goal.float())
                goal_feat = self.mvp_model.forward_norm(goal_feat)
                goal_feat[precomputed_goal_feat_mask] = torch.from_numpy(
                    obs["goal_feature"][precomputed_goal_feat_mask]).float().to(self.device)
        robot_state = torch.from_numpy(obs["robot_state"]).float().to(self.device)
        privilege_info = torch.from_numpy(obs["privilege_info"]).float().to(self.device)
        obs = torch.cat([scene_feat, robot_state, goal_feat, privilege_info], dim=-1)
        if self.mvp_feat_dim is None:
            self.mvp_feat_dim = scene_feat.shape[-1]
        if self.robot_state_dim is None:
            self.robot_state_dim = robot_state.shape[-1]
        if self.privilege_info_dim is None:
            self.privilege_info_dim = privilege_info.shape[-1]
        return obs
    
    def reset(self):
        obs = self.venv.reset()
        obs = self.mvp_process_obs(obs)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs.shape[-1],))
        return obs
    
    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)
    
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self.mvp_process_obs(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        for _info in info:
            if "terminal_observation" in _info:
                normed_img, normed_goal = self._normalize_obs(_info["terminal_observation"])
                with torch.no_grad():
                    scene_feat = self.mvp_model.forward_norm(self.mvp_model.extract_feat(normed_img.float())).squeeze(dim=0)
                    goal_feat = self.mvp_model.forward_norm(self.mvp_model.extract_feat(normed_goal.float())).squeeze(dim=0)
                robot_state = torch.from_numpy(_info["terminal_observation"]["robot_state"]).float().to(self.device)
                privilege_info = torch.from_numpy(_info["terminal_observation"]["privilege_info"]).float().to(self.device)
                _info["terminal_observation"] = torch.cat([scene_feat, robot_state, goal_feat, privilege_info], dim=-1)
        return obs, reward, done, info

    def get_state_from_obs(self, obs):
        robot_state = obs[..., self.mvp_feat_dim: self.mvp_feat_dim + self.robot_state_dim]
        goal_feat = obs[..., self.mvp_feat_dim + self.robot_state_dim: 2 * self.mvp_feat_dim + self.robot_state_dim]
        privilege_info = obs[..., 2 * self.mvp_feat_dim + self.robot_state_dim: 2 * self.mvp_feat_dim + self.robot_state_dim + self.privilege_info_dim]
        return torch.cat([robot_state, privilege_info, goal_feat], dim=-1)
    
    def get_obs(self, indices=None):
        obs = self.venv.get_obs(indices)
        obs = self.mvp_process_obs(obs)
        return obs

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        # reward = np.expand_dims(reward, axis=1).astype(np.float32)
        for _info in info:
            if "terminal_observation" in _info:
                _info["terminal_observation"] = torch.from_numpy(_info["terminal_observation"]).float().to(self.device)
        return obs, reward, done, info

