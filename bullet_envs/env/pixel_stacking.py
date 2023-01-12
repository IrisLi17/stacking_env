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
        self.task_queue = []
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
            self.n_step = 0
        else:
            task_array = self.task_queue[np.random.randint(len(self.task_queue))]
            self.set_task(task_array)
        # self.robot.render_fn = partial(render, robot=self.robot, view_mode=self.view_mode, width=128, height=128, 
        #             shift_params=self.shift_params)
        # self.robot.goal_img = self.goal_dict["img"].transpose((1, 2, 0)) if self.goal_dict["img"] is not None else None
        obs = self._get_obs()
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
    
    def _get_privilege_info(self):
        cur_object_poses = []
        for i in range(self.n_object):
            pos, quat = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            cur_object_poses.append(np.concatenate([pos, quat]))
        cur_object_poses = np.concatenate(cur_object_poses)
        goal_poses = self.goal_dict["full_state"]
        return np.concatenate([cur_object_poses, goal_poses])

    def set_task(self, task_array):
        # TODO: set initial state in env, set self.goal_dict for goal
        pass

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
