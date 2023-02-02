# Create pixel based environment
# Assume goal images always come from agent generated dataset
# For base task learning, let's use dummy goal images, and relabel in hindsight. 
# Also, let's get fine-tuning tasks from the dataset as well.

# how to specify privileged info? all object 6dof poses and all goal sudo-6dof poses?
# then reward would be difficult to obtain, everything else seem fine


from bullet_envs.env.primitive_stacking import ArmStack
from bullet_envs.env.primitive_env import render, BasePrimitiveEnv
import os
import numpy as np
from functools import partial
from collections import deque
import pybullet as p
from pybullet_utils import bullet_client as bc
import pkgutil
egl = pkgutil.get_loader('eglRenderer')


class PixelStack(ArmStack):
    def __init__(self, view_mode="third", use_gpu_render=True, feature_dim=768, 
                 shift_params=(0, 0), *args, **kwargs):
        self.view_mode = view_mode
        self.use_gpu_render = use_gpu_render
        self.privilege_dim = None
        self.feature_dim = feature_dim
        self.shift_params = shift_params
        self.task_queue = deque(maxlen=3000)
        self.dist_threshold = 0.05
        # self.record_cfg = dict(
        #     save_video_path=os.path.join(os.path.dirname(__file__), "..", "tmp"),
        #     fps=10
        # )
        super().__init__(*args, **kwargs)
    
    def _create_simulation(self):
        self.p = bc.BulletClient(connection_mode=p.DIRECT)
        if self.use_gpu_render:
            plugin = self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            print("plugin=", plugin)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
    
    def _setup_callback(self):
        super()._setup_callback()
        for shape in self.p.getVisualShapeData(self.robot.id):
            self.p.changeVisualShape(self.robot.id, shape[1], rgbaColor=(0, 0, 0, 0))

    def reset(self):
        if len(self.task_queue) == 0:
            self._reset_sim()
            # sample goal
            self.goal_dict = self._sample_goal()
            self.goal = self.goal_dict["under_specify_state"]
        else:
            idx = np.random.randint(len(self.task_queue))
            task_array = self.task_queue[idx]
            self.set_task(task_array)
        self.n_step = 0
        # self.robot.render_fn = partial(render, robot=self.robot, view_mode=self.view_mode, width=128, height=128, 
        #             shift_params=self.shift_params)
        # self.robot.goal_img = self.goal_dict["img"].transpose((1, 2, 0)) if self.goal_dict["img"] is not None else None
        obs = self._get_obs()
        if len(self.task_queue) > 0:
            assert np.linalg.norm(obs["privilege_info"] - task_array[7: -self.feature_dim]) < 1e-3, (obs["privilege_info"], task_array[7: -self.feature_dim])
        return obs
    
    def _get_obs(self):
        # TODO: tweak view angle, or simply make the robot invisible
        scene = render(self.p, width=128, height=128, robot=self.robot, view_mode=self.view_mode,
                       shift_params=self.shift_params, pitch=-45, distance=0.6,
                       camera_target_position=(0.5, 0.0, 0.1)).transpose((2, 0, 1))[:3]
        robot_obs = self.robot.get_obs()
        privilege_info = self._get_privilege_info()
        if self.privilege_dim is None:
            self.privilege_dim = privilege_info.shape[0]
        return {"img": scene, "robot_state": robot_obs, 
                "goal": self.goal_dict["img"], "privilege_info": privilege_info,
                "goal_feature": self.goal_dict["feature"], "goal_source": self.goal_dict["source"]}
    
    def visualize_goal(self, goal, token_type=True):
        if self.body_goal is not None:
            for goal_id in self.body_goal:
                self.p.removeBody(goal_id)
    
    def _get_achieved_goal(self):
        cur_pos = []
        for i in range(self.n_object):
            pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            cur_pos.append(np.array(pos))
        return np.array(cur_pos), np.array(self.blocks_id)
    
    def _get_privilege_info(self):
        cur_object_poses = []
        for i in range(self.n_object):
            pos, quat = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            pos, quat = map(np.array, [pos, quat])
            if quat[-1] < 0:
                quat = -quat
            cur_object_poses.append(np.concatenate([pos, quat]))
        cur_object_poses = np.concatenate(cur_object_poses)
        goal_poses = self.goal_dict["full_state"]
        return np.concatenate([cur_object_poses, goal_poses])

    def set_task(self, task_array):
        # TODO: set initial state in env, set self.goal_dict for goal
        privilege_info = np.reshape(task_array[7: -self.feature_dim], (2, -1))
        goal_feature = task_array[-self.feature_dim:]
        init_state, goal_state = privilege_info[0], privilege_info[1]
        init_state = np.reshape(init_state, (self.n_object, 7))
        for i in range(self.n_object):
            self.p.resetBasePositionAndOrientation(self.blocks_id[i], init_state[i][:3], init_state[i][3:])
        # Set from agent generated task
        self.goal_dict["full_state"] = goal_state
        # dummy, since image feature will be set separately
        self.goal_dict["img"] = np.zeros((3, 128, 128), dtype=np.uint8)
        self.goal_dict["feature"] = goal_feature
        self.goal_dict["source"] = np.array([1])
        # TODO: seems still wrong
        # Take care of self.goal since it will be used to compute reward and success
        reshape_goal_state = goal_state.reshape(self.n_object, 7)
        goal_quat = reshape_goal_state[:, 3:]
        goal_euler = np.array([self.p.getEulerFromQuaternion(goal_quat[i]) for i in range(self.n_object)])
        self.goal = np.concatenate(
            [goal_euler, reshape_goal_state[:, :3], np.eye(self.n_object)], axis=-1)

    def add_tasks(self, task_arrays):
        self.task_queue.extend(task_arrays)
    
    def clear_tasks(self):
        self.task_queue.clear()
    
    def _sample_goal(self):
        # We need old sample_goal just to make sure demo collection can run
        # In other cases, we don't need it
        goal_state = ArmStack._sample_goal(self)
        goal_dict = {
            "under_specify_state": goal_state, "img": np.zeros((3, 128, 128)), 
            "feature": np.zeros(self.feature_dim), "source": np.array([0]),
            "full_state": np.zeros(7 * self.n_object)
        }
        if not hasattr(self, "goal_dict"):
            self.goal_dict = goal_dict
        return goal_dict
    
    def compute_reward_and_info(self):
        if self.goal_dict["source"][0] == 0:
            return super(PixelStack, self).compute_reward_and_info()
        # rewrite
        state_and_goal = self._get_privilege_info().reshape(2, -1)
        cur_state = state_and_goal[0].reshape(self.n_object, -1)
        goal = state_and_goal[1].reshape(self.n_object, -1)
        com_cond = np.all(np.linalg.norm(
            cur_state[:, :3] - goal[:, :3], axis=-1) < self.dist_threshold
        )
        cur_x_vec = [quat_apply(cur_state[i, 3:], np.array([1., 0., 0.])) for i in range(self.n_object)]
        goal_x_vec = [quat_apply(goal[i, 3:], np.array([1., 0., 0.])) for i in range(self.n_object)]
        rot_cond = np.all([np.abs(np.dot(cur_x_vec[i], goal_x_vec[i])) > 0.75 for i in range(self.n_object)])
        is_success = com_cond and rot_cond
        # if self.reward_type == "sparse":
        #     reward = float(is_success)
        # info = {"is_success": is_success}

        # TODO: revert to old one
        # diff_vec = goal[:, :3] - cur_state[:, :3]
        # local_diff_vec = np.array([(np.linalg.inv(
        #     np.array(self.p.getMatrixFromQuaternion(cur_state[i, 3:])).reshape((3, 3))
        # ) @ diff_vec[i].reshape((3, 1))).squeeze() for i in range(self.n_object)])
        # halfextent = np.array(
        #     [np.array(self.p.getCollisionShapeData(self.blocks_id[i], -1)[0][3]) / 2 for i in range(self.n_object)]
        # )
        # is_success = np.all(np.abs(local_diff_vec) < halfextent)
        if is_success:
            tmp_state = self.p.saveState()
            for _ in range(10):
                self.p.stepSimulation()
            future_pose = self._get_privilege_info().reshape(2, self.n_object, -1)[0]
            is_stable = np.all(np.linalg.norm(future_pose[:, :3] - cur_state[:, :3], axis=-1) < 1e-3)
            is_success = is_stable
            self.p.restoreState(stateId=tmp_state)
            self.p.removeState(tmp_state)
        if self.reward_type == "sparse":
            reward = float(is_success)
        info = {"is_success": is_success}
        return reward, info
    
    def oracle_feasible(self, obs: np.ndarray):
        return get_n_to_move(obs, self.n_object, self.dist_threshold, 0.75)
        is_feasible = (match_count == self.n_object - 1)
        assert is_feasible.shape == (obs.shape[0],)
        return is_feasible
    
    def render(self, mode="rgb_array"):
        return render(
            self.p, width=128, height=128, robot=self.robot, view_mode=self.view_mode,
            shift_params=self.shift_params, pitch=-45, distance=0.6,
            camera_target_position=(0.5, 0.0, 0.1)
        )
    
    def get_goal_image(self):
        # Only used for debugging purpose
        cur_bullet_state = self.p.saveState()
        goal_state = self.goal_dict["full_state"].reshape((self.n_object, 7))
        for i in range(self.n_object):
            self.p.resetBasePositionAndOrientation(self.blocks_id[i], goal_state[i][:3], goal_state[i][3:])
        goal_img = self.render()
        self.p.restoreState(cur_bullet_state)
        self.p.removeState(cur_bullet_state)
        return goal_img
    
    def set_dist_threshold(self, dist_threshold):
        self.dist_threshold = dist_threshold
    
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape((4,))
    b = b.reshape((3,))
    xyz = a[:3]
    t = np.cross(xyz, b) * 2
    return b + a[3:] * t + np.cross(xyz, t)

def quat_apply_batch(a, b):
    xyz = a[..., :3]
    t = np.cross(xyz, b) * 2
    return b + a[..., 3:] * t + np.cross(xyz, t)

def get_n_to_move(obs, n_object, dist_threshold, rot_threshold):
    privilege_info = obs[..., -2 * n_object * 7:].reshape((-1, 2, n_object, 7))
    achieved_state = privilege_info[:, 0]
    goal_state = privilege_info[:, 1]
    pos_cond = np.linalg.norm(achieved_state[:, :, :3] - goal_state[:, :, :3], axis=-1) < dist_threshold
    achieved_x_vec = quat_apply_batch(achieved_state[:, :, 3:], np.array([1., 0., 0.]).reshape((1, 1, 3)))
    goal_x_vec = quat_apply_batch(goal_state[:, :, 3:], np.array([1., 0., 0.]).reshape((1, 1, 3)))
    rot_cond = np.abs(np.sum(achieved_x_vec * goal_x_vec, axis=-1)) > rot_threshold
    match_count = np.sum(np.logical_and(pos_cond, rot_cond), axis=-1)
    n_to_move = n_object - match_count
    return n_to_move
# def test():
#     alpha = np.random.uniform(-np.pi, np.pi, size=(10,))
#     beta = np.random.uniform(-np.pi, np.pi, size=(10,))
#     theta = np.random.uniform(-np.pi, np.pi, size=(10,))
#     q = np.concatenate([np.expand_dims(np.sin(theta / 2), axis=-1) * np.stack(
#         [np.sin(alpha) * np.sin(beta), np.sin(alpha) * np.cos(beta), np.cos(alpha)], axis=-1), 
#         np.expand_dims(np.cos(theta / 2), axis=-1)], axis=-1)
#     v = np.random.uniform(-2, 2, size=(10, 3))
#     res1 = quat_apply_batch(q, v)
#     res2 = []
#     for i in range(q.shape[0]):
#         res2.append(quat_apply(q[i], v[i]))
#     res2 = np.array(res2)
#     assert np.linalg.norm(res1 - res2) < 1e-3


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = PixelStack(
        n_object=6, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
        n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", use_gpu_render=True,
    )
    print("created env")
    obs = env.reset()
    print("reset obs", obs["privilege_info"], obs["robot_state"])
    os.makedirs("tmp", exist_ok=True)
    for i in range(10):
        img = obs["img"].transpose((1, 2, 0))
        #ax.cla()
        #ax.imshow(img)
        plt.imsave("tmp/tmp%d.png" % i, img)

        action = env.act()
        action = action.squeeze(dim=0)
        # action = env.action_space.sample()
        # action[4:7] = [0, 0, 0.5]
        print(action)
        #action = [0, 0, 0, 0, 0, 0, 0.001]
        obs, reward, done, info = env.step(action)
    print(obs)
