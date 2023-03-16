import pybullet as p
import numpy as np
import random
from itertools import product, combinations
from bullet_envs.env.motion_planners.rrt_connect import birrt
from bullet_envs.env.robots import ArmRobot, Kinova2f85Robot, UR2f85Robot, XArm7Robot
from bullet_envs.env.bullet_rotations import quat_mul, quat_rot_vec, euler2quat, quat_diff, quat2euler
import time, os, shutil, copy
from bullet_envs.env.env_utils import PhysClientWrapper
import matplotlib.pyplot as plt

MAX_DISTANCE = 0.01
BASE_LINK = -1
CIRCULAR_LIMITS = None

# Phase code
PHASE_FETCH = 0
PHASE_CLOSE = 1
PHASE_GRIPPER_MOVE = 2
PHASE_BLOCK_MOVE = 3
PHASE_RELEASE = 4
PHASE_LIFT_UP = 5
PHASE_BACK = 6
PHASE_RESET = 7

# Error code
SUCCESS = 0
IK_FAIL = 1
END_IN_COLLISION = 2
PATH_IN_COLLISION = 3
EXECUTION_FAIL = 4


class CollisionInfo(object):
    def __init__(self, contactFlag, bodyA, bodyB, linkA, linkB, posonA, posonB, contactNormalOnB, contactDistance,
                 *args):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.linkA = linkA
        self.linkB = linkB
        self.positionOnA = posonA
        self.positionOnB = posonB
        self.contactDistance = contactDistance


def all_between(low, q, high):
    low = np.array(low)
    q = np.array(q)
    high = np.array(high)
    assert low.shape == q.shape == high.shape
    return np.all(low <= q) and np.all(q <= high)


def all_close(a, b, atol=1e-6, rtol=0.):
    assert len(a) == len(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


def set_joint_positions(wrapped_p, body, joints, q):
    assert len(joints) == len(q), (joints, q)
    for i in range(len(joints)):
        wrapped_p.resetJointState(body, joints[i], q[i])


def get_movable_joints(wrapped_p, body):
    res = []
    for i in range(wrapped_p.getNumJoints(body)):
        if wrapped_p.getJointInfo(body, i)[2] != wrapped_p.JOINT_FIXED:
            res.append(i)
    return res


def get_links(wrapped_p, body):
    return list(range(wrapped_p.getNumJoints(body)))


def get_link_parent(wrapped_p, body, link):
    # Assume joint index == link index
    info = wrapped_p.getJointInfo(body, link)
    return info[-1]


def get_custom_limits(wrapped_p, body, joints, custom_limits: dict, circular_limits=None):
    # TODO: check custom_limits
    low_limits, high_limits = [], []
    for i in range(len(joints)):
        info = wrapped_p.getJointInfo(body, joints[i])
        low_limit, high_limit = info[8], info[9]
        low_limits.append(low_limit)
        high_limits.append(high_limit)
    return low_limits, high_limits


def get_joint_positions(wrapped_p, body, joints):
    joint_states = wrapped_p.getJointStates(body, joints)
    joint_positions, *_ = zip(*joint_states)
    return joint_positions


# TODO: circular support
def get_difference_fn(wrapped_p, body, joints):
    def fn(q1, q2):
        assert len(q1) == len(q2) == len(joints)
        return np.array(q1) - np.array(q2)

    return fn


def get_refine_fn(wrapped_p, body, joints, num_steps):
    def fn(q1, q2):
        q1 = np.array(q1)
        q2 = np.array(q2)
        assert len(q1) == len(q2) == len(joints)
        q_seq = [q1 + i / num_steps * (q2 - q1) for i in range(1, num_steps + 1)]
        return q_seq

    return fn


def get_image(wrapped_p: PhysClientWrapper, width, height):
    view_matrix = wrapped_p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0., 0.1],
                                                              distance=1.2,
                                                              yaw=90,
                                                              pitch=-45,
                                                              roll=0,
                                                              upAxisIndex=2)
    proj_matrix = wrapped_p.computeProjectionMatrixFOV(fov=60,
                                                       aspect=1.0,
                                                       nearVal=0.1,
                                                       farVal=100.0)
    (_, _, px, _, _) = wrapped_p.getCameraImage(width=width,
                                                height=height,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                # renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                )
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (width, height, 4))

    rgb_array = rgb_array[:450, :, :3]
    return rgb_array


class ArmPose(object):
    def __init__(self, body, conf, joints, wrapped_p):
        self.wrapped_p = wrapped_p
        self.body = body
        self.conf = conf
        self.joints = joints
        assert len(self.joints) == len(self.conf), (self.joints, self.conf)

    def assign(self, wrapped_p):
        set_joint_positions(wrapped_p, self.body, self.joints, self.conf)


class ArmPath(object):
    def __init__(self, body, path, wrapped_p, grasp_config=None):
        self.wrapped_p = wrapped_p
        self.body = body
        assert isinstance(path, list)
        self.path = path
        self.grasp_config = grasp_config


class Attachment(object):
    def __init__(self, body, parent_body, parent_link, rel_xyz, rel_quat, topdown_quat, wrapped_p: PhysClientWrapper, obj_offset=None):
        self.wrapped_p = wrapped_p
        self.child = body
        self.parent_body = parent_body
        self.parent_link = parent_link
        self.rel_xyz = rel_xyz  # [0, 0, 0.01]
        self.rel_quat = rel_quat  # [0, 0, 0.707, 0.707]. child quat - parent quat
        self.topdown_quat = topdown_quat
        self.obj_offset = obj_offset

    def assign(self):
        parent_state = self.wrapped_p.getLinkState(self.parent_body, self.parent_link)
        parent_xyz, parent_quat, *_ = parent_state
        # print("parent quat", parent_quat, "topdown quat", self.topdown_quat)
        child_xyz = np.array(quat_rot_vec(quat_diff(parent_quat, self.topdown_quat), self.rel_xyz)) + np.array(
            parent_xyz)
        child_quat = quat_mul(parent_quat, self.rel_quat)
        if self.obj_offset:
            # child is a block and the grasp position has a offset
            child_longaxis = quat_rot_vec(child_quat, np.array([1., 0., 0.]))
            child_xyz -= child_longaxis * self.obj_offset
        self.wrapped_p.resetBasePositionAndOrientation(self.child, child_xyz, child_quat)


class Grasp(object):
    def __init__(self, robot_id, eef_link, wrapped_p: PhysClientWrapper):
        self.robot_id = robot_id
        self.eef_link = eef_link
        self.wrapped_p = wrapped_p
        self.attachments = []

    def set_attach(self, body_id, rel_quat=[0., 0., 0., 1.], rel_xyz=[0., 0., 0.01], topdown_quat=[0., 0., 0., 1.], obj_offset=None):
        self.attachments.append(Attachment(body_id, self.robot_id, self.eef_link, rel_xyz, rel_quat,
                                           topdown_quat, self.wrapped_p, obj_offset=obj_offset))

    def clear_attach(self):
        self.attachments = []

    def attachment(self):
        return self.attachments


class Planner(object):
    def __init__(self, robot: ArmRobot, wrapped_p, exec_p, verbose=0, smooth_path=True):
        self.robot_id = robot._robot
        self.robot = robot
        self.p = wrapped_p
        self.exec_p = exec_p
        self.fixed = []
        self.verbose = verbose
        self._collision_id = 0

        self.moving_links = self.moving_joints = self.robot.motorIndices
        self.ll, self.ul = get_custom_limits(self.p, self.robot_id, self.moving_joints, custom_limits={})
        self.robot_links = get_links(self.p, self.robot_id)
        self.fixed_links = list(filter(lambda x: x not in self.moving_links, self.robot_links))
        self.smooth_path = smooth_path

    def set_fixed(self, fixed):
        self.fixed = fixed

    def ik(self, pos, orn):
        # TODO: IKfast output None
        movable_pos, info = self.robot.run_ik(pos, orn)
        if len(np.array(movable_pos).shape) == 1:
            movable_pos = [movable_pos[:self.robot.ndof]]
        return movable_pos, info

    def check_qpos_collision(self, qs, q_finger, grasp=None, disabled_collisions=set(), start_disabled_collisions=set(),
                             disable_start_collision=False):
        get_fn_time = 0
        io_time = 0
        collision_time = 0
        t1 = time.time()
        collision_fn = self.get_collision_fn(self.robot_id, self.moving_joints, self.fixed,
                                             grasp.attachment() if grasp is not None else [],
                                             disabled_collisions=disabled_collisions)
        q_start = get_joint_positions(self.p, self.robot_id, self.moving_joints)
        if qs is None:
            qs = [q_start[:self.robot.ndof]]
        if not disable_start_collision:
            if len(start_disabled_collisions):
                start_collision_fn = self.get_collision_fn(self.robot_id, self.moving_joints, self.fixed,
                                                           grasp.attachment() if grasp is not None else [],
                                                           disabled_collisions=set.union(disabled_collisions,
                                                                                         start_disabled_collisions))
            else:
                start_collision_fn = collision_fn
            get_fn_time += time.time() - t1
            t1 = time.time()
            if start_collision_fn(q_start):
                if self.verbose > 1:
                    print("start collision")
                return np.array([[True] * len(qs)])
        collision_time += time.time() - t1

        t1 = time.time()
        io_time += time.time() - t1
        t1 = time.time()
        is_collision = []
        for q in qs:
            q_end = list(q) + list(q_finger)
            is_collision.append(collision_fn(q_end))
        is_collision = np.array(is_collision)
        collision_time += time.time() - t1
        t1 = time.time()
        io_time += time.time() - t1
        if self.verbose > 1:
            print("\tGet fn time", get_fn_time)
            print("\tCollision time", collision_time, "num q", len(qs))
            print("\tIO time", io_time)
        return is_collision

    def do_plan_motion(self, q_dest=None, q_finger=[0., 0., 0., 0.], grasp=None,
                       disabled_collisions=set(), plan_state_id=None, **kwargs):
        save_load_time = 0
        plan_path_time = 0
        t1 = time.time()
        self.p.restoreState(stateId=plan_state_id)
        save_load_time += time.time() - t1
        q_start = get_joint_positions(self.p, self.robot_id, self.moving_joints)
        if q_dest is None:
            q_dest = q_start[:self.robot.ndof]
        t1 = time.time()
        path = self.path_planner(
            ArmPose(self.robot_id, conf=list(q_start), wrapped_p=self.p, joints=self.moving_joints),
            ArmPose(self.robot_id, conf=list(q_dest) + list(q_finger), wrapped_p=self.p, joints=self.moving_joints),
            grasp=grasp,
            disabled_collisions=disabled_collisions,
            **kwargs
        )
        plan_path_time += time.time() - t1
        if path is None:
            return None
        if grasp is not None and len(path) > 1:
            path = [path[0]] + \
                   [tuple(list(p[:self.robot.ndof]) + list(q_finger)) for p in path[1:]]
        t1 = time.time()
        save_load_time += time.time() - t1
        if self.verbose > 1:
            print("\tSave load time", save_load_time, "plan path time", plan_path_time)
        return q_start, q_dest, ArmPath(self.robot_id, path, wrapped_p=self.p)

    def path_planner(self, armpose1, armpose2, grasp=None, teleport=False, self_collisions=True,
                     disabled_collisions=set(), **kwargs):
        t1 = time.time()
        assert ((armpose1.body == armpose2.body) and (
                armpose1.joints == armpose2.joints))
        if teleport:
            path = [armpose1.conf, armpose2.conf]
        else:
            obstacles = self.fixed
            if self.verbose:
                print(
                    f"[Planner/getter] Planning joint motion. Grasp = {grasp}. Obstacles = {obstacles}.\n\tArmpose1 = {armpose1.conf}.\n\tArmpose2 = {armpose2.conf}")
            path = self.plan_joint_motion(self.robot_id, armpose2.joints, armpose1.conf, armpose2.conf,
                                          obstacles=obstacles,
                                          attachments=grasp.attachment() if grasp is not None else [],
                                          self_collisions=self_collisions,
                                          disabled_collisions=disabled_collisions,
                                          smooth=20 if self.smooth_path else None,
                                          **kwargs
                                          )
            if path is None:
                return None
        return path

    def plan_joint_motion(self, body, joints, start_conf, end_conf, obstacles=[], attachments=[],
                          self_collisions=True, disabled_collisions=set(),
                          weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={},
                          **kwargs):

        preparation_time = 0
        t1 = time.time()
        assert len(joints) == len(end_conf)
        end_conf = tuple(end_conf)
        sample_fn = self.get_sample_fn(
            body, joints, custom_limits=custom_limits)
        distance_fn = self.get_distance_fn(body, joints, weights=weights)
        extend_fn = self.get_extend_fn(
            body, joints, resolutions=resolutions)
        collision_fn = self.get_collision_fn(body, joints, obstacles, attachments, disabled_collisions)
        preparation_time += time.time() - t1

        t1 = time.time()
        ret = birrt(start_conf, end_conf, distance_fn,
                    sample_fn, extend_fn, collision_fn, **kwargs)
        birrt_time = time.time() - t1
        if self.verbose > 1:
            print("\t\t\tpreparation time", preparation_time, "birrt time", birrt_time)
        return ret

    # Four functions for plan_joint_motion
    def get_sample_fn(self, body, joints, custom_limits={}, **kwargs):
        def uniform_generator(d):
            while True:
                yield np.random.uniform(size=d)

        def halton_generator(d):
            try:
                import ghalton
            except ImportError:
                ghalton = None
            seed = random.randint(0, 1000)
            sequencer = ghalton.GeneralizedHalton(d, seed)
            while True:
                [weights] = sequencer.get(1)
                yield np.array(weights)

        def unit_generator(d, use_halton=False):
            if use_halton:
                try:
                    import ghalton
                except ImportError:
                    print(
                        'ghalton is not installed (https://pypi.org/project/ghalton/)')
                    use_halton = False
            return halton_generator(d) if use_halton else uniform_generator(d)

        def interval_generator(lower, upper, **kwargs):
            assert len(lower) == len(upper)
            assert np.less_equal(lower, upper).all()
            if np.equal(lower, upper).all():
                return iter([lower])
            return (weights * lower + (1 - weights) * upper for weights in unit_generator(d=len(lower), **kwargs))

        lower_limits, upper_limits = get_custom_limits(
            self.p, body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
        generator = interval_generator(lower_limits, upper_limits, **kwargs)

        def fn():
            return tuple(next(generator))

        return fn

    def get_distance_fn(self, body, joints, weights=None):  # , norm=2):
        # TODO: use the energy resulting from the mass matrix here?
        if weights is None:
            weights = 1 * np.ones(len(joints))  # TODO: use velocities here
        difference_fn = get_difference_fn(self.p, body, joints)

        def fn(q1, q2):
            diff = np.array(difference_fn(q2, q1))
            return np.sqrt(np.dot(weights, diff * diff))

        return fn

    DEFAULT_RESOLUTION = 0.05

    def get_extend_fn(self, body, joints, resolutions=None, norm=2):
        # norm = 1, 2, INF
        if resolutions is None:
            resolutions = Planner.DEFAULT_RESOLUTION * np.ones(len(joints))
        difference_fn = get_difference_fn(self.p, body, joints)

        def fn(q1, q2):
            steps = int(np.linalg.norm(
                np.divide(difference_fn(q2, q1), resolutions), ord=norm))
            refine_fn = get_refine_fn(
                self.p, body, joints, num_steps=steps)
            return refine_fn(q1, q2)

        return fn

    def get_collision_fn(self, body, joints, obstacles, attachments: list, disabled_collisions):
        def get_check_pairs():
            '''
            self-link pairs from the robot arm, and between attachments and the arm
            Intra-collision pairs between obstacles and the robot with attachments
            :param body:
            :param joints:
            :param attachments:
            :param disabled_collisions:
            :return: a list of (body1, link1, body2, link2)
            '''
            moving_body_and_links = [(body, link) for link in self.moving_links]
            robot_body_and_links = [(body, link) for link in self.robot_links]
            fixed_body_and_links = [(body, link) for link in self.fixed_links]
            # Hard coded arm links
            arm_body_and_links = [(body, link) for link in range(2, self.robot.end_effector_index)]
            check_link_pairs = list(product(moving_body_and_links, fixed_body_and_links))
            check_link_pairs.extend(combinations(moving_body_and_links, 2))
            # Attachments
            for attachment in attachments:
                check_link_pairs.extend(product(arm_body_and_links, [(attachment.child, -1)]))

            def are_links_adjacent(body1, link1, body2, link2):
                return body1 == body2 and ((get_link_parent(self.p, body1, link1) == link2) or
                                           (get_link_parent(self.p, body1, link2) == link1))

            check_link_pairs = list(
                filter(lambda pair: not are_links_adjacent(pair[0][0], pair[0][1], pair[1][0], pair[1][1]),
                       check_link_pairs))

            # Between obstacle and robot + attachments
            robot_and_attach_body_and_links = robot_body_and_links + [(attachment.child, -1) for attachment in
                                                                      attachments]
            check_link_pairs.extend(
                product(robot_and_attach_body_and_links, [(obstacle, -1) for obstacle in obstacles]))

            # Filter out disabled pairs
            check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                        ((pair[1], pair[0]) not in disabled_collisions),
                                           check_link_pairs))
            return check_link_pairs

        t1 = time.time()
        check_collision_pairs = get_check_pairs()
        if self.verbose > 1:
            print("get check_collision_pairs time", time.time() - t1)
        
        def collision_fn(q):
            start_time = time.time()
            t1 = time.time()
            if not all_between(np.array(self.ll)[:self.robot.ndof] - 1e-3, q[:self.robot.ndof], np.array(self.ul)[:self.robot.ndof] + 1e-3):
                if self.verbose:
                    print(
                        f'[Planner/collision] Joint limits violated.\n\tll = {self.ll}\n\tq = {q}\n\tul = {self.ul}')
                return True
            # sync_plan_with_exec(self.p, self.exec_p)
            set_joint_positions(self.p, body, joints, q)
            # print("attachments", attachments)
            for attachment in attachments:
                attachment.assign()
            self.p.performCollisionDetection()
            if self.verbose > 2:
                print("in collision fn, env time", time.time() - t1, "n attachment", len(attachments))
            all_contact_points = self.p.getContactPoints()
            if len(all_contact_points):
                _, bodyA, bodyB, linkA, linkB, *_ = zip(*all_contact_points)
                set1 = set(zip(zip(bodyA, linkA), zip(bodyB, linkB)))
                set2 = set(zip(zip(bodyB, linkB), zip(bodyA, linkA)))
                all_contact_set = set.union(set1, set2)
            else:
                all_contact_set = set()
            check_collision_set = set(check_collision_pairs)
            found_collision_pairs = all_contact_set.intersection(check_collision_set)
            is_collision = not all_contact_set.isdisjoint(check_collision_set)
            if self.verbose > 1:
                if is_collision:
                    self._collision_id += 1
                    print("[DEBUG] collision idx", self._collision_id, get_body_pos_and_orn(5, self.p))
                    pose = ArmPose(self.robot_id, conf=list(q), wrapped_p=self.p, joints=self.moving_joints)
                    pose.assign(self.p)
                    img = get_image(self.p, 500, 500)
                    plt.imsave(f"video_tmp/debug_collision_{self._collision_id}.png", img)
            if self.verbose > 1:
                if is_collision:
                    print(all_contact_set.intersection(check_collision_set))
            return is_collision

        return collision_fn


class Executor(object):
    def __init__(self, robot: ArmRobot, wrapped_p, ctrl_mode="position", record=False, verbose=0):
        self.robot = robot
        self.robot_id = robot._robot
        self.p = wrapped_p
        self.ctrl_mode = ctrl_mode
        self.record = record
        if record:
            if os.path.exists("video_tmp"):
                shutil.rmtree("video_tmp")
            os.makedirs("video_tmp", exist_ok=True)
            import imageio
            self.video_writer = imageio.get_writer("robot_demo.mp4",
                                               fps=20,
                                               format='FFMPEG',
                                               codec='h264',)
        self.img_idx = 0
        self.verbose = verbose
        self.movable_joints = self.robot.motorIndices

    def set_ctrl_mode(self, mode):
        assert mode in ["teleport", "position"]
        self.ctrl_mode = mode

    def visualize(self, q, label):
        pose = ArmPose(self.robot_id, conf=list(q), wrapped_p=self.p, joints=self.movable_joints)
        pose.assign(self.p)
        self.robot.change_visual(True)
        img = get_image(self.p, 128, 128)
        plt.imsave(f"video_tmp/debugimg_{label}.png", img)


    def run(self, arm_path: ArmPath, attachments=[], atol=1e-3, early_stop=False, current_state=""):
        self.robot.change_visual(visible=True)
        if self.verbose > 1:
            print("[Executor] Arm path length", len(arm_path.path))
        set_position_time = 0
        set_attachment_time = 0
        step_simulation_time = 0
        if self.ctrl_mode == "teleport":
            for p_idx, path in enumerate(arm_path.path):
                t1 = time.time()
                set_joint_positions(self.p, self.robot_id, self.movable_joints, path)
                set_position_time += time.time() - t1
                t1 = time.time()
                for attachment in attachments:
                    attachment.assign()
                set_attachment_time += time.time() - t1
                # TODO: potential issue: insufficient simulation
                t1 = time.time()
                # self.p.stepSimulation()
                step_simulation_time += time.time() - t1
                # time.sleep(0.01)
                if self.record:
                    img = get_image(self.p, 500, 500)
                    self.video_writer.append_data(img)
                    # plt.imsave(f"video_tmp/tmpimg_{self.img_idx}{current_state}.png", img)
                    self.img_idx += 1
                if early_stop and (self.robot.get_end_effector_pos()[2] > 0.25 or quat_rot_vec(self.robot.get_end_effector_orn(as_type="quat"), np.array([0., 0., 1.]))[2] < -0.9):
                    arm_path.path = arm_path.path[:p_idx + 1]
                    break
            if self.verbose > 1:
                print("In execution, set position time", set_position_time, "set attachment time", set_attachment_time,
                      "step simulation time", step_simulation_time, "length", len(arm_path.path))

        elif self.ctrl_mode == "position":
            for path in arm_path.path:
                if self.verbose:
                    print("initial",
                          np.linalg.norm(np.array(self.robot.get_joint_pos(self.robot.motorIndices)) - np.array(path)))
                position_gain = np.ones((len(path),))
                position_gain[self.robot.ndof:] = 0.1
                self.robot.position_control(path, position_gain=position_gain)
                n_step = 0
                threshold = 500
                while not all_close(np.array(self.robot.get_joint_pos(self.robot.motorIndices)),
                                    np.array(path), atol=atol) and (n_step < threshold):
                    self.p.stepSimulation()
                    for attachment in attachments:
                        attachment.assign()
                    time.sleep(1 / 240.)
                    n_step += 1
                if self.verbose:
                    print(n_step,
                          np.linalg.norm(np.array(self.robot.get_joint_pos(self.robot.motorIndices)) - np.array(path)))
                time.sleep(1 / 240.)
                if not all_close(np.array(self.robot.get_joint_pos(self.robot.motorIndices)), np.array(path),
                                 atol=atol):
                    if self.verbose:
                        print("after exec", np.array(self.robot.get_joint_pos(self.robot.motorIndices)), "\ndesired",
                              path)
                    return False
        else:
            raise NotImplementedError
        return True
    
    def __del__(self):
        if self.record:
            self.video_writer.close()

def get_eef_orn(object_orn, candidate_gripper_orn):
    '''
        Compute a candidate eef quaternion from the quaternion of a block.
        :param object_orn: quat ([x, y, z, w]) of a block
               candidate_gripper_orn: quat ([x, y, z, w]) of the gripper relative to a block with orn ([1, 0, 0, 0])
        :return: eef orientation
    '''
    assert len(object_orn) == 4
    gripper_orn = quat_mul(object_orn, candidate_gripper_orn)
    return gripper_orn

def get_all_eef_orns(object_orn, candidate_gripper_orns):
    '''
        Compute eef quaternion of all candiates from the quaternion of a block.
        :param object_orn: quat ([x, y, z, w]) of a block
               candidate_gripper_orns: a list of candiate quat ([x, y, z, w]) of the gripper relative to a block with orn ([1, 0, 0, 0])
        :return: a list eef orientation
    '''
    res = []
    for candidate_gripper_orn in candidate_gripper_orns:
        res.append(get_eef_orn(object_orn, candidate_gripper_orn))
    return res
    
def get_obj_longaxis(object_orn):
    '''
        Compute the long axis of a block
        :param object_orn: quat ([x, y, z, w]) of a block
        :return: long axis of the object
    '''
    obj_longaxis = quat_rot_vec(object_orn, np.array([1., 0., 0.]))
    return obj_longaxis


def get_body_pos_and_orn(body_id, p: PhysClientWrapper):
    pos, orn = p.getBasePositionAndOrientation(body_id)
    return np.asarray(pos), np.asarray(orn)


class Primitive(object):
    def __init__(self, planner: Planner, executor: Executor, exec_body_blocks: list, body_cliff0, body_cliff1,
                 plan_body_blocks: list, body_tables: list, verbose=0, teleport_arm=False, force_scale=0):
        self.planner = planner
        self.executor = executor
        self.plan_robot = planner.robot
        self.plan_p = self.planner.p
        self.exec_p = self.executor.p
        self.plan_body_blocks = plan_body_blocks
        self.exec_body_blocks = exec_body_blocks
        self.body_cliff0 = None
        self.body_cliff1 = None
        self.body_tables = body_tables
        self.CLOSED_FINGER = self.plan_robot.gen_gripper_joint_command(1)
        self.OPEN_FINGER = self.plan_robot.gen_gripper_joint_command(0)
        self.topdown_euler = self.plan_robot.topdown_euler
        self.init_gripper_axis = self.plan_robot.init_gripper_axis
        self.init_gripper_quat = self.plan_robot.init_gripper_quat
        if isinstance(self.plan_robot, XArm7Robot):
            self.grasp_offset = -0.16
        else:
            self.grasp_offset = -0.17  # -0.17  # from point to grasp to eef link
        self._init_gripper_orn_candidates()
        self._init_grasp_config_candidates()
        self.verbose = verbose
        self.teleport_arm = teleport_arm
        self.force_scale = force_scale
        print("primitive force scale", force_scale)

    def _init_gripper_orn_candidates(self):
        self.gripper_orn_candidates = []
        topdown = euler2quat(self.topdown_euler)
        gripper_orn = topdown
        block_longaxis = np.array([1., 0., 0.])
        for i in range(4):
            _rot = np.concatenate([np.sin(i * np.pi / 2 / 2) * block_longaxis, [np.cos(i * np.pi / 2 / 2)]])
            candidate = quat_mul(_rot, gripper_orn)
            self.gripper_orn_candidates.append(candidate)
    
    def _init_grasp_config_candidates(self):
        '''
            grasp_configs stores a list of (gripper_orn, object_offset) tuples. The configs serve as candidates to pick up and place a block.
        '''
        self.object_offsets = [-0.04, 0., 0.04] # from which point to grasp the block. 0.04/-0.04 represents two ends of the block.
        self.grasp_configs = []
        gripper_orn = euler2quat(self.topdown_euler)

        # grasp along the long axis
        block_longaxis = np.array([1., 0., 0.])
        for i in range(4):
            _rot = np.concatenate([np.sin(i * np.pi / 2 / 2) * block_longaxis, [np.cos(i * np.pi / 2 / 2)]])
            candidate_gripper_orn = quat_mul(_rot, gripper_orn)
            for obj_offset in self.object_offsets:
                self.grasp_configs.append((candidate_gripper_orn, obj_offset))
                pass
        
        # grasp the two ends of the block
        _rot = np.concatenate([np.sin(3 * np.pi / 2 / 2) * np.array([0., 1., 0.]), [np.cos(3 * np.pi / 2 / 2)]])
        _gripper_orn = quat_mul(_rot, gripper_orn)
        for i in range(4):
            _rot = np.concatenate([np.sin(i * np.pi / 2 / 2) * block_longaxis, [np.cos(i * np.pi / 2 / 2)]])
            candidate_gripper_orn = quat_mul(_rot, _gripper_orn)
            self.grasp_configs.append((candidate_gripper_orn, -0.04))
        
        _rot = np.concatenate([np.sin(-3 * np.pi / 2 / 2) * np.array([0., 1., 0.]), [np.cos(-3 * np.pi / 2 / 2)]])
        _gripper_orn = quat_mul(_rot, gripper_orn)
        for i in range(4):
            _rot = np.concatenate([np.sin(i * np.pi / 2 / 2) * block_longaxis, [np.cos(i * np.pi / 2 / 2)]])
            candidate_gripper_orn = quat_mul(_rot, _gripper_orn)
            self.grasp_configs.append((candidate_gripper_orn, 0.04))


    def align_at_reset(self):
        sync_plan_with_exec(self.plan_p, self.exec_p)
        id_and_pos = [(i, get_body_pos_and_orn(i, self.plan_p)[0]) for i in range(3, self.plan_p.getNumBodies())]
        ids, plan_pos = zip(*id_and_pos)
        exec_id_and_pos = [(i, get_body_pos_and_orn(i, self.exec_p)[0]) for i in self.exec_body_blocks]
        assert len(id_and_pos) == len(exec_id_and_pos)
        self.plan_body_blocks = []
        for i in range(len(exec_id_and_pos)):
            self.plan_body_blocks.append(
                ids[np.argmin([np.linalg.norm(item - exec_id_and_pos[i][1]) for item in plan_pos])])
            
    def prob_vec_from_quat(self, quat):
        return quat_rot_vec(quat_diff(quat, euler2quat(self.topdown_euler)), np.array([0., 0., -1]))

    def _fetch_object(self, tgt_block):
        # TODO: offset in grasping direction
        start_time = time.time()
        if self.verbose > 0:
            print("#############")
            print("Start fetch")
        t1 = time.time()
        obstacles = self.plan_body_blocks + self.body_tables
        self.planner.set_fixed(obstacles)
        set_obstacle_time = time.time() - t1
        t1 = time.time()
        sync_plan_with_exec(self.plan_p, self.exec_p)
        # assert np.linalg.norm(get_body_pos_and_orn(self.plan_body_blocks[tgt_block], self.plan_p)[0] -
        #                       get_body_pos_and_orn(self.exec_body_blocks[tgt_block], self.exec_p)[0]) < 1e-3
        # assert np.linalg.norm(get_body_pos_and_orn(self.body_cliff0, self.plan_p)[0] -
        #                       get_body_pos_and_orn(self.body_cliff0, self.exec_p)[0]) < 1e-3
        # assert np.linalg.norm(np.array(self.plan_p.getCollisionShapeData(self.body_cliff0, -1)[0][3]) -
        #                       np.array(self.exec_p.getCollisionShapeData(self.body_cliff0, -1)[0][3])) < 1e-3
        # for i in range(len(self.plan_body_blocks)):
        #     assert np.linalg.norm(np.array(self.plan_p.getCollisionShapeData(self.plan_body_blocks[i], -1)[0][3]) -
        #                           np.array(
        #                               self.exec_p.getCollisionShapeData(self.exec_body_blocks[i], -1)[0][3])) < 1e-3

        plan_state_id = self.plan_p.saveState()
        sync_env_time = time.time() - t1
        # assert np.linalg.norm(get_body_pos_and_orn(self.plan_body_blocks[tgt_block], self.plan_p)[0] -
        #                       get_body_pos_and_orn(self.exec_body_blocks[tgt_block], self.exec_p)[0]) < 1e-3
        t1 = time.time()
        cur_q = get_joint_positions(self.plan_p, self.plan_robot._robot,
                                    self.plan_robot.motorIndices)[:self.plan_robot.ndof]
        # Get approach q
        block_pos, block_orn = get_body_pos_and_orn(self.plan_body_blocks[tgt_block], self.plan_p)
        ori_gripper_quats = [candidate_gripper_orn for candidate_gripper_orn, obj_offset in self.grasp_configs]
        gripper_quats = [get_eef_orn(block_orn, candidate_gripper_orn) for candidate_gripper_orn, obj_offset in self.grasp_configs]
        obj_offsets = [obj_offset for _, obj_offset in self.grasp_configs]
        tgt_obj_longaxis = get_obj_longaxis(block_orn)
        get_eef_orn_time = time.time() - t1

        t1 = time.time()
        approach_q = []
        final_q = []
        tgt_obj_offset = []
        tgt_gripper_quat = []
        base_gripper_quat = []
        for (obj_offset, gripper_quat, ori_gripper_quat) in zip(obj_offsets, gripper_quats, ori_gripper_quats):
                # if self.verbose > 1:
                #     print(block_pos, gripper_quat, self.prob_vec_from_quat(gripper_quat))
                moveable_q, info = self.planner.ik(block_pos + 1.2 * self.grasp_offset * self.prob_vec_from_quat(gripper_quat) + tgt_obj_longaxis * obj_offset,
                                                gripper_quat)
                _final_q, final_info = self.planner.ik(block_pos + self.grasp_offset * self.prob_vec_from_quat(gripper_quat) + tgt_obj_longaxis * obj_offset, gripper_quat)
                if info["is_success"] and final_info["is_success"]:
                    L = min(len(moveable_q), len(_final_q))
                    approach_q.extend(moveable_q[:L])
                    final_q.extend(_final_q[:L])
                    tgt_obj_offset.extend([obj_offset] * L)
                    tgt_gripper_quat.extend([gripper_quat] * L)
                    base_gripper_quat.extend([ori_gripper_quat] * L)
                    # approach_q.append(moveable_q[:6])
        ik_time = time.time() - t1
        if self.verbose > 1:
            print("ik time", ik_time)
        if not len(approach_q):
            self.plan_p.removeState(plan_state_id)
            return False, IK_FAIL, None, None
        # No memory leakage above
        print("Get", len(approach_q), "approach q")
        for i, q in enumerate(approach_q):
            self.executor.visualize(list(q) + list(self.OPEN_FINGER), f'approach_q_{i}')
        # Call planner
        t1 = time.time()
        valid_paths = []
        q_finger = self.OPEN_FINGER
        if isinstance(self.plan_robot, Kinova2f85Robot):
            disabled_pairs = set(combinations([(self.plan_robot._robot, i) for i in range(
                self.plan_robot.end_effector_index + 1, self.plan_robot.num_joints)], 2))
            # print("disabled pairs", disabled_pairs)
        elif isinstance(self.plan_robot, UR2f85Robot):
            disabled_pairs = set.union(self.plan_robot.collision_pairs,
                                       {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in self.body_tables})
        else:
            disabled_pairs = set.union(self.plan_robot.collision_pairs,
                                       {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in self.body_tables})
        disabled_pairs_time = time.time() - t1
        # approach_q = sorted(approach_q, key=lambda x: np.linalg.norm(x - cur_q))
        # No memory leakage above
        t1 = time.time()
        is_collision = self.planner.check_qpos_collision(approach_q, q_finger, disabled_collisions=disabled_pairs)
        check_collision_time = time.time() - t1
        if self.verbose > 1:
            print("check collision time", time.time() - t1)
        if np.all(is_collision):
            self.plan_p.removeState(plan_state_id)
            return False, END_IN_COLLISION, None, None

        t1 = time.time()
        approach_q = (np.array(approach_q)[~is_collision]).tolist()
        final_q = (np.array(final_q)[~is_collision]).tolist()
        tgt_obj_offset = (np.array(tgt_obj_offset)[~is_collision]).tolist()
        tgt_gripper_quat = (np.array(tgt_gripper_quat)[~is_collision]).tolist()
        base_gripper_quat = (np.array(base_gripper_quat)[~is_collision]).tolist()
        data = zip(approach_q, final_q, tgt_obj_offset, tgt_gripper_quat, base_gripper_quat)
        data = sorted(data, key=lambda x: np.linalg.norm(np.array(x[0]) - cur_q))
        approach_q, final_q, tgt_obj_offset, tgt_gripper_quat, base_gripper_quat = zip(*data)

        for q_idx, q in enumerate(approach_q):
            res = self.planner.do_plan_motion(q, q_finger, disabled_collisions=disabled_pairs,
                                              plan_state_id=plan_state_id)
            if self.verbose > 1:
                print(q_idx, time.time() - t1)
            if res is not None:
                # Append final q
                res[2].path.append(list(final_q[q_idx]) + list(q_finger))
                if self.verbose > 1:
                    print("path length", len(res[2].path))
                if self.teleport_arm and len(res[2].path) > 2:
                    res[2].path = [res[2].path[0], res[2].path[-1]]
                res[2].grasp_config = (base_gripper_quat[q_idx], tgt_gripper_quat[q_idx], tgt_obj_offset[q_idx])
                valid_paths.append(res[2])
                break
        self.plan_p.removeState(plan_state_id)
        planning_time = time.time() - t1
        if self.verbose > 1:
            print("Planning time", planning_time)
        if not len(valid_paths):
            return False, PATH_IN_COLLISION, None, None
        # No leakage above
        # Call executor
        t1 = time.time()
        assert len(valid_paths) == 1
        # valid_paths = sorted(valid_paths, key=lambda x: len(x.path))
        # old_state = self.exec_p.saveState()
        for armpath in valid_paths:
            res = self.executor.run(armpath, current_state='_fetch_object_')
            if res:
                execution_time = time.time() - t1
                # self.exec_p.removeState(old_state)
                if self.verbose > 1:
                    print("set_obstacle_time", set_obstacle_time)
                    print("sync_env_time", sync_env_time)
                    print("get_eef_orn_time", get_eef_orn_time)
                    print("ik time", ik_time)
                    print("disabled_pairs_time", disabled_pairs_time)
                    print("check_collision_time", check_collision_time)
                    print("planning_time", planning_time)
                    print("execution time", execution_time)
                    print("Summary: fetch time", time.time() - start_time)
                return True, SUCCESS, armpath.path, dict(grasp_config=armpath.grasp_config)
            # else:
            #     self.exec_p.restoreState(stateId=old_state)
        # self.exec_p.removeState(old_state)
        return False, EXECUTION_FAIL, None, None

    def _close_finger(self, tgt_block):
        if self.teleport_arm:
            cur_q = get_joint_positions(self.exec_p, self.plan_robot._robot,
                                        get_movable_joints(self.exec_p, self.plan_robot._robot))
            tgt_q = list(cur_q[:self.plan_robot.ndof]) + list(self.CLOSED_FINGER)
            valid_path = ArmPath(self.plan_robot._robot, [cur_q, tgt_q], self.exec_p)
        else:
            if self.verbose > 0:
                print("##########")
                print("Start closing finger")
            sync_plan_with_exec(self.plan_p, self.exec_p)
            plan_state_id = self.plan_p.saveState()
            # assert np.linalg.norm(get_body_pos_and_orn(self.plan_body_blocks[tgt_block], self.plan_p)[0] -
            #                       get_body_pos_and_orn(self.exec_body_blocks[tgt_block], self.exec_p)[0]) < 1e-3

            q_finger = self.CLOSED_FINGER
            if not isinstance(self.plan_robot, UR2f85Robot):
                disabled_intra_pairs = list(combinations([(self.plan_robot._robot, link) for link in range(
                    self.plan_robot.end_effector_index + 1, self.plan_robot.num_joints)], 2)) + \
                                       list(product([(self.plan_robot._robot, link) for link in
                                                     range(self.plan_robot.end_effector_index + 1,
                                                           self.plan_robot.num_joints)],
                                                    [(i, -1) for i in self.plan_body_blocks])) + \
                                       [((self.plan_robot._robot, 0), (table_id, -1)) for table_id in self.body_tables]
            else:
                disabled_intra_pairs = set.union(self.plan_robot.collision_pairs,
                                                 set(product([(self.plan_robot._robot, link) for link in [14, 19]],
                                                             [(self.plan_body_blocks[tgt_block], -1)])),
                                                 {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in
                                                  self.body_tables})
            is_collision = self.planner.check_qpos_collision(None, q_finger,
                                                             disabled_collisions=set(disabled_intra_pairs))
            if np.all(is_collision):
                self.plan_p.removeState(plan_state_id)
                return False, END_IN_COLLISION, None

            res = self.planner.do_plan_motion(
                None, q_finger, disabled_collisions=set(disabled_intra_pairs), plan_state_id=plan_state_id)
            self.plan_p.removeState(plan_state_id)
            if res is not None:
                if self.teleport_arm and len(res[2].path) > 2:
                    res[2].path = [res[2].path[0], res[2].path[-1]]
                valid_path = res[2]
            else:
                return False, PATH_IN_COLLISION, None
        res = self.executor.run(valid_path, current_state='_close_finger_')
        if res:
            return True, SUCCESS, valid_path.path
        return False, EXECUTION_FAIL, None

    def _contact_bodies(self, body_id, max_distance=MAX_DISTANCE):
        res = []
        for body in self.plan_body_blocks + self.body_tables:
            if body != body_id and len(self.plan_p.getClosestPoints(body_id, body, max_distance)) != 0:
                res.append(body)
        return res

    def _change_pose_with_grasp(self, approach_q, tgt_block, tgt_obj_offset, base_gripper_quat, plan_state_id, abstract_grasp, target_type):
        sync_plan_with_exec(self.plan_p, self.exec_p)
        t1 = time.time()
        q_finger = self.CLOSED_FINGER
        grasp = Grasp(self.plan_robot._robot, self.plan_robot.end_effector_index, self.plan_p)
        
        rel_quat = quat_diff(np.array([0., 0., 0., 1.]), base_gripper_quat)
        rel_xyz = [0., 0., self.grasp_offset + 0.003] # [-(self.grasp_offset + 0.003), 0., 0.]
        grasp.set_attach(self.plan_body_blocks[tgt_block], rel_quat=rel_quat,
                         topdown_quat=euler2quat(self.topdown_euler), rel_xyz=rel_xyz, obj_offset=tgt_obj_offset)
        
        grasp_time = time.time() - t1
        obstacles = self.planner.fixed.copy()
        if self.plan_body_blocks[tgt_block] in obstacles:
            obstacles.remove(self.plan_body_blocks[tgt_block])
        self.planner.set_fixed(obstacles)
        disabled_pairs_time = 0
        t1 = time.time()
        if not isinstance(self.plan_robot, UR2f85Robot):
            disabled_intra_pairs = set.union(
                self.plan_robot.collision_pairs,
                set(product([(self.plan_robot._robot, link) for link in range(
                    self.plan_robot.end_effector_index + 1, self.plan_robot.num_joints)],
                            [(self.plan_body_blocks[tgt_block], -1)])),
                {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in self.body_tables}
            )
        else:
            disabled_intra_pairs = set.union(self.plan_robot.collision_pairs,
                                             set(product([(self.plan_robot._robot, link) for link in [14, 19]],
                                                         [(self.plan_body_blocks[tgt_block], -1)])),
                                             {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in
                                              self.body_tables})
        disabled_table_tgt_pair = set({((self.plan_body_blocks[tgt_block], -1), (table_id, -1)) for table_id in self.body_tables})
        disabled_pairs_time += time.time() - t1
        t1 = time.time()
        contact_bodies = self._contact_bodies(self.plan_body_blocks[tgt_block])
        start_disabled = set([((self.plan_body_blocks[tgt_block], -1), (body, -1)) for body in contact_bodies])
        if self.verbose > 1:
            print("change pose, get contact body", time.time() - t1)
        t1 = time.time()
        disabled_inter_pairs = set()
        disabled_pairs_time += time.time() - t1
        valid_paths = []

        t1 = time.time()
        is_collision = self.planner.check_qpos_collision(approach_q, q_finger, grasp,
                                                         disabled_collisions=set.union(disabled_inter_pairs,
                                                                                       disabled_intra_pairs,
                                                                                       disabled_table_tgt_pair),
                                                         start_disabled_collisions=start_disabled)
        check_collision_time = time.time() - t1
        if self.verbose > 1:
            print("is collision", is_collision)
            print("check collision time", time.time() - t1)
        if np.all(is_collision):
            print("change pose end in collision")
            return False, END_IN_COLLISION, None

        t1 = time.time()
        approach_q = (np.array(approach_q)[~is_collision]).tolist()

        for q_idx, q in enumerate(approach_q):
            res = self.planner.do_plan_motion(q, q_finger, grasp, set.union(disabled_intra_pairs, disabled_inter_pairs),
                                              plan_state_id=plan_state_id)
            if self.verbose > 1:
                print(q_idx, time.time() - t1)
            if res is not None:
                if self.verbose > 1:
                    print("path length", len(res[2].path))
                if self.teleport_arm and len(res[2].path) > 2:
                    res[2].path = [res[2].path[0], res[2].path[-1]]
                valid_paths.append(res[2])
                break
        planning_time = time.time() - t1
        if self.verbose > 1:
            print("change pose, planning time", time.time() - t1)
        if not valid_paths:
            print("change pose path in collision")
            return False, PATH_IN_COLLISION, None
        assert len(valid_paths) == 1
        t1 = time.time()
        if abstract_grasp:
            exec_attachments = [
                Attachment(self.exec_body_blocks[tgt_block], self.plan_robot._robot, self.plan_robot.end_effector_index,
                           rel_xyz, rel_quat, euler2quat(self.topdown_euler), self.exec_p, obj_offset=tgt_obj_offset)]
        else:
            exec_attachments = []
        assert len(valid_paths) == 1
        for armpath in valid_paths:
            res = self.executor.run(armpath, attachments=exec_attachments,  current_state=f'_change_{target_type}_pose_')
            if res:
                execution_time = time.time() - t1
                if self.verbose > 1:
                    print("sync_env_time", sync_env_time)
                    print("get_eef_orn_time", get_eef_orn_time)
                    print("ik_time", ik_time)
                    print("grasp_time", grasp_time)
                    print("disabled_pairs_time", disabled_pairs_time)
                    print("check_collision_time", check_collision_time)
                    print("planning_time", planning_time)
                    print("execution_time", execution_time)
                    print("Summary: change pose time", time.time() - start_time)
                return True, SUCCESS, armpath.path
        return False, EXECUTION_FAIL, None

    def _change_pose(self, tgt_block, tgt_pos, tgt_orn, grasp_config=None, abstract_grasp=True, target_type="block"):
        '''
            :param: target_type: if 'block', tgt_pos and tgt_orn are with a block; if 'gripper', tgt_pos and tgt_orn are with the gripper
        '''
        start_time = time.time()
        assert target_type in ["gripper", "block"]
        tgt_pos[2] = np.maximum(tgt_pos[2], 0.045)
        # for vertical blocks, force gripper height
        # if abs(np.dot(quat_rot_vec(tgt_orn, np.array([0., 1., 0.])), np.array([0., 0., 1.]))) > 0.9:
        #     tgt_pos[2] = 0.105
        if self.verbose > 0:
            print("###########")
            print("Start change pos")
        t1 = time.time()
        sync_plan_with_exec(self.plan_p, self.exec_p)
        plan_state_id = self.plan_p.saveState()
        sync_env_time = time.time() - t1
        t1 = time.time()
        if grasp_config is None:
            grasp_configs = copy.deepcopy(self.grasp_configs)
        else:
            base_gripper_quat, _, tgt_obj_offset = grasp_config
            grasp_configs = [(base_gripper_quat, tgt_obj_offset)]
        tgt_obj_longaxis = get_obj_longaxis(tgt_orn)
        get_eef_orn_time = time.time() - t1
        # print("in change pos, gripper quats", gripper_quats)
        for grasp_config in grasp_configs:
            print("grasp config", grasp_config)
            t1 = time.time()

            base_gripper_quat, tgt_obj_offset = grasp_config
            if target_type == "block":
                gripper_quat = get_eef_orn(tgt_orn, base_gripper_quat)
                moveable_q, info = self.planner.ik(tgt_pos + self.grasp_offset * self.prob_vec_from_quat(gripper_quat) + tgt_obj_longaxis * tgt_obj_offset,
                                                gripper_quat)
            elif target_type == "gripper":
                moveable_q, info = self.planner.ik(tgt_pos, tgt_orn)
            else:
                raise NotImplementedError(f"target_type '{target_type}' is not supported.")

            ik_time = time.time() - t1
            if info["is_success"]:
                approach_q = moveable_q
            else:
                if self.verbose > 1:
                    print("IK Failed for grasp_config:", grasp_config)
                continue
            if self.verbose > 1:
                print("IK time", time.time() - t1)
            res = self._change_pose_with_grasp(approach_q, tgt_block, tgt_obj_offset, base_gripper_quat, plan_state_id, abstract_grasp, target_type=target_type)
            if res[0]:
                self.plan_p.removeState(plan_state_id)
                return res
        self.plan_p.removeState(plan_state_id)
        return False, EXECUTION_FAIL, None

    def _release_finger(self, tgt_block):
        if self.verbose > 0:
            print("############\nStart release finger")
        if self.teleport_arm:
            cur_q = get_joint_positions(self.exec_p, self.plan_robot._robot,
                                        get_movable_joints(self.exec_p, self.plan_robot._robot))
            tgt_q = list(cur_q[:self.plan_robot.ndof]) + list(self.OPEN_FINGER)
            valid_paths = [ArmPath(self.plan_robot._robot, [cur_q, tgt_q], self.exec_p)]
        else:
            sync_plan_with_exec(self.plan_p, self.exec_p)
            plan_state_id = self.plan_p.saveState()
            q_finger = self.OPEN_FINGER
            valid_paths = []
            disabled_collision_pairs = set.union(self.plan_robot.collision_pairs,
                                                 {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in
                                                  self.body_tables})
            is_collision = self.planner.check_qpos_collision(None, q_finger,
                                                             disabled_collisions=disabled_collision_pairs)
            if np.all(is_collision):
                self.plan_p.removeState(plan_state_id)
                return False, END_IN_COLLISION, None

            res = self.planner.do_plan_motion(None, q_finger, disabled_collisions=disabled_collision_pairs,
                                              plan_state_id=plan_state_id)
            self.plan_p.removeState(plan_state_id)
            if res is not None:
                if self.teleport_arm and len(res[2].path) > 2:
                    res[2].path = [res[2].path[0], res[2].path[-1]]
                valid_paths.append(res[2])
            if not valid_paths:
                return False, PATH_IN_COLLISION, None
        for armpath in valid_paths:
            res = self.executor.run(armpath, current_state='_release_finger_')
            if res:
                if self.verbose > 1:
                    print("before release finger", self.exec_p.getBasePositionAndOrientation(self.exec_body_blocks[tgt_block]))
                self.exec_p.setJointMotorControlArray(
                    self.plan_robot._robot, self.plan_robot.motorIndices, self.exec_p.POSITION_CONTROL,
                    armpath.path[-1])
                t1 = time.time()
                # inject noise
                self.exec_p.applyExternalForce(self.exec_body_blocks[tgt_block], -1,
                                               forceObj=np.random.normal(0., self.force_scale, size=(3,)),
                                               posObj=(0, 0, 0), flags=self.exec_p.LINK_FRAME)
                for _ in range(10):
                    self.exec_p.stepSimulation()
                    print("After {} simulation steps".format(_ + 1))
                self._sim_until_stable()
                if self.verbose > 1:
                    print("sim until stable time", time.time() - t1)
                    print("after release finger", self.exec_p.getBasePositionAndOrientation(self.exec_body_blocks[tgt_block]))
                return True, SUCCESS, armpath.path
        return False, EXECUTION_FAIL, None

    def _sim_until_stable(self):
        count = 0
        while count < 10 and np.linalg.norm(np.concatenate(
                [self.exec_p.getBaseVelocity(self.exec_body_blocks[i])[0]
                 for i in range(len(self.exec_body_blocks))])) > 1e-3:
            for _ in range(50):
                self.exec_p.stepSimulation()
            count += 1

    def _lift_up(self):
        if self.verbose > 0:
            print("###############\nStart lift up")
        sync_plan_with_exec(self.plan_p, self.exec_p)
        plan_state_id = self.plan_p.saveState()
        tgt_pos = self.plan_robot.get_end_effector_pos()
        tgt_poses = []
        if tgt_pos[2] < 0.2:
            tgt_poses.append(tgt_pos + np.array([-0.05, 0., 0.]))
            tgt_poses.append(tgt_pos + np.array([0., 0., 0.2]))
            tgt_quat = self.plan_robot.get_end_effector_orn(as_type="quat")
        else:
            self.plan_p.removeState(plan_state_id)
            return True, SUCCESS, None
        cur_q = get_joint_positions(self.plan_p, self.plan_robot._robot,
                                    self.plan_robot.motorIndices[:self.plan_robot.ndof])
        approach_q = []
        for i in range(len(tgt_poses)):
            q, info = self.planner.ik(tgt_poses[i], tgt_quat)
            if info["is_success"]:
                approach_q.extend(q)
        if not approach_q:
            self.plan_p.removeState(plan_state_id)
            return False, IK_FAIL, None
        q_finger = self.OPEN_FINGER
        disabled_collision_pairs = set.union(
            self.plan_robot.collision_pairs,
            {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in self.body_tables},
        )
        obstacles = self.plan_body_blocks + self.body_tables
        self.planner.set_fixed(obstacles)
        # TODO: if initial state is in collision, the planner *may* fail
        is_collision = self.planner.check_qpos_collision(approach_q, q_finger,
                                                         disabled_collisions=disabled_collision_pairs,
                                                         disable_start_collision=True)
        if np.all(is_collision):
            self.plan_p.removeState(plan_state_id)
            return False, END_IN_COLLISION, None
        approach_q = (np.array(approach_q)[~is_collision]).tolist()
        approach_q = sorted(approach_q, key=lambda x: np.linalg.norm(np.array(x) - np.array(cur_q)))
        valid_paths = []
        for q_idx, q in enumerate(approach_q):
            res = self.planner.do_plan_motion(q, q_finger, disabled_collisions=disabled_collision_pairs,
                                              plan_state_id=plan_state_id, disable_start_collision=True)
            if res is not None:
                if self.verbose > 1:
                    print("path length", len(res[2].path))
                if self.teleport_arm and len(res[2].path) > 2:
                    res[2].path = [res[2].path[0], res[2].path[-1]]
                valid_paths.append(res[2])
                break
        self.plan_p.removeState(plan_state_id)
        if not valid_paths:
            return False, PATH_IN_COLLISION, None
        for armpath in valid_paths:
            res = self.executor.run(armpath, early_stop=True, current_state='_lift_up_')
            if res:
                return True, SUCCESS, armpath.path
        return False, EXECUTION_FAIL, None

    def _move_back(self):
        if self.verbose > 0:
            print("###############\nStart move back")
        sync_plan_with_exec(self.plan_p, self.exec_p)
        plan_state_id = self.plan_p.saveState()
        tgt_pos = []
        tgt_pos.append(self.plan_robot.endEffectorPos)
        tgt_quat = euler2quat(self.plan_robot.topdown_euler)
        approach_q = []
        for i in range(len(tgt_pos)):
            q, info = self.planner.ik(tgt_pos[i], tgt_quat)
            if info['is_success']:
                approach_q.extend(q)
        if not approach_q:
            self.plan_p.removeState(plan_state_id)
            return False, IK_FAIL, None
        q_finger = self.OPEN_FINGER
        disabled_collision_pairs = set.union(self.plan_robot.collision_pairs,
                                             {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in
                                              self.body_tables})
        obstacles = self.plan_body_blocks + self.body_tables
        self.planner.set_fixed(obstacles)
        is_collision = self.planner.check_qpos_collision(approach_q, q_finger,
                                                         disabled_collisions=disabled_collision_pairs)
        if np.all(is_collision):
            self.plan_p.removeState(plan_state_id)
            return False, END_IN_COLLISION, None
        approach_q = (np.array(approach_q)[~is_collision]).tolist()
        valid_paths = []
        for q_idx, q in enumerate(approach_q):
            res = self.planner.do_plan_motion(q, q_finger, disabled_collisions=disabled_collision_pairs,
                                              plan_state_id=plan_state_id)
            if res is not None:
                if self.verbose > 1:
                    print("path length", len(res[2].path))
                if self.teleport_arm and len(res[2].path) > 2:
                    res[2].path = [res[2].path[0], res[2].path[-1]]
                valid_paths.append(res[2])
                break
        self.plan_p.removeState(plan_state_id)
        if not valid_paths:
            return False, PATH_IN_COLLISION, None
        for armpath in valid_paths:
            res = self.executor.run(armpath, current_state='_move_back_')
            if res:
                self.exec_p.setJointMotorControlArray(
                    self.plan_robot._robot, self.plan_robot.motorIndices, self.exec_p.POSITION_CONTROL,
                    armpath.path[-1])
                return True, SUCCESS, armpath.path
        return False, EXECUTION_FAIL, None

    def _reset_pose(self):
        start_time = time.time()
        if self.verbose > 0:
            print("##############")
            print("Start reset pos")

        t1 = time.time()
        sync_plan_with_exec(self.plan_p, self.exec_p)
        plan_state_id = self.plan_p.saveState()
        sync_env_time = time.time() - t1
        planning_time = 0
        t1 = time.time()
        q = np.array(self.plan_robot.init_qpos)[self.plan_robot.motorIndices[:self.plan_robot.ndof]]
        q_finger = self.OPEN_FINGER
        obstacles = self.plan_body_blocks + self.body_tables
        self.planner.set_fixed(obstacles)
        planning_time += time.time() - t1
        t1 = time.time()
        valid_paths = []
        disabled_collision_pairs = set.union(self.plan_robot.collision_pairs,
                                             {((self.plan_robot._robot, 0), (table_id, -1)) for table_id in
                                              self.body_tables})
        disabled_pairs_time = time.time() - t1
        t1 = time.time()
        res = self.planner.do_plan_motion(q, q_finger, disabled_collisions=disabled_collision_pairs,
                                          plan_state_id=plan_state_id)
        self.plan_p.removeState(plan_state_id)
        planning_time += time.time() - t1
        if res is not None:
            if self.teleport_arm and len(res[2].path) > 2:
                res[2].path = [res[2].path[0], res[2].path[-1]]
            valid_paths.append(res[2])
        if not valid_paths:
            return False, PATH_IN_COLLISION, None
        t1 = time.time()
        for armpath in valid_paths:
            res = self.executor.run(armpath, current_state='_reset_pose_')
            execution_time = time.time() - t1
            if res:
                if self.verbose > 1:
                    print("sync_env_time", sync_env_time)
                    print("disabled_pairs_time", disabled_pairs_time)
                    print("planning_time", planning_time)
                    print("execution_time", execution_time)
                    print("Summary: reset pose time", time.time() - start_time)
                return True, SUCCESS, armpath.path
        return False, EXECUTION_FAIL, None

    def move_one_object(self, tgt_block: int, tgt_pos, tgt_orn):
        print("[DEBUG] move one object")
        for block_id in self.plan_body_blocks:
            print(f"[DEBUG] planner block{block_id}", self.exec_p.getCollisionShapeData(block_id, -1), self.plan_p.getCollisionShapeData(block_id, -1), flush=True)
        # TODO: too many rounds, looks unnecessary
        path = dict()
        move_object_start = time.time()

        print("[DEBUG] try to fetch object")
        t1 = time.time()
        res = self._fetch_object(tgt_block)
        fetch_object_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("fetch object fail", res[1])
            return 10 * PHASE_FETCH + res[1], None
        else:
            path["fetch_object"] = res[2]
        grasp_config = res[3]['grasp_config']
        print("[DEBUG] grasp_config is", grasp_config, flush=True)

        print("[DEBUG] try to close finger")
        t1 = time.time()
        res = self._close_finger(tgt_block)
        close_finger_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("close finger fail", res[1])
            return 10 * PHASE_CLOSE + res[1], None
        else:
            path["close_finger"] = res[2]

        print("[DEBUG] try to change to default pose of the gripper")
        t1 = time.time()
        res = self._change_pose(tgt_block, np.asarray(self.plan_robot.endEffectorPos), euler2quat(self.plan_robot.topdown_euler), grasp_config=grasp_config, abstract_grasp=True, target_type='gripper')
        change_gripper_pose_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("change gripper pos fail", res[1])
            return 10 * PHASE_GRIPPER_MOVE + res[1], None
        else:
            path["change_pose_gripper"] = res[2]


        print("[DEBUG] try to change to target pose of the block")
        t1 = time.time()
        res = self._change_pose(tgt_block, tgt_pos, tgt_orn, grasp_config=None, abstract_grasp=True, target_type='block')
        change_block_pose_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("change block pos fail", res[1])
            exit()
            return 10 * PHASE_BLOCK_MOVE + res[1], None
        else:
            path["change_pose"] = res[2]

        print("[DEBUG] try to release finger")
        t1 = time.time()
        res = self._release_finger(tgt_block)
        release_finger_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("release finger fail", res[1])
            return 10 * PHASE_RELEASE + res[1], None
        else:
            path["release_finger"] = res[2]

        print("[DEBUG] try to lift up")
        t1 = time.time()
        res = self._lift_up()
        lift_up_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("lift up fail", res[1])
            return 10 * PHASE_LIFT_UP + res[1], None
        else:
            path["lift_up"] = res[2]

        print("[DEBUG] try to move back")
        t1 = time.time()
        res = self._move_back()
        move_back_time = time.time() - t1
        if not res[0]:
            if self.verbose > 0:
                print("move back fail", res[1])
            return 10 * PHASE_BACK + res[1], None
        else:
            path["move_back"] = res[2]

        print("[DEBUG] SUCCESS!!", self.verbose)
        if self.verbose > 0:
            print("primitive success")
        if self.verbose > 1:
            print("move object time", time.time() - move_object_start)
            print("\tfetch object time", fetch_object_time)
            print("\tclose finger time", close_finger_time)
            print("\tchange gripper pose time", change_gripper_pose_time)
            print("\tchange block pose time", change_block_pose_time)
            print("\trelease finger time", release_finger_time)
            print("\tlift up time", lift_up_time)
            print("\tmove back time", move_back_time)
        return SUCCESS, path


def sync_plan_with_exec(plan_p, exec_p):
    # Looks like pybullet caches the file. so we randomize the file name
    import uuid
    state_file = "tmp" + str(uuid.uuid4()) + ".bullet"
    t1 = time.time()
    exec_p.saveBullet(state_file)
    # print("save file", time.time() - t1)
    t1 = time.time()
    plan_p.restoreState(fileName=state_file)
    os.remove(state_file)


def main():
    raise NotImplementedError
    robot = "xarm"
    env = BulletBridgeConstructionHigh(7, random_size=True, min_num_blocks=7, discrete=False, random_mode="long",
                                       action_scale=0.6, block_thickness=0.025, cliff_thickness=0.1, cliff_height=0.1,
                                       noop=True, render=True, need_visual=True, robot=robot)
    env.set_hard_ratio(1.0, 7)
    inner_env = BulletBridgeConstructionLow(7, random_size=True, discrete=False, mode="long", block_thickness=0.025,
                                            cliff_thickness=0.1, cliff_height=0.1, render=False, need_visual=False,
                                            robot=robot)
    planning_p = inner_env.p
    planning_robot = inner_env.robot
    exec_p = env.env.p
    exec_robot = env.env.robot

    env.reset()

    planner = Planner(planning_robot, planning_p, verbose=1, smooth_path=True)
    obstacles = [inner_env.body_cliff0, inner_env.body_cliff1] + inner_env.body_blocks
    planner.set_fixed(obstacles)

    executor = Executor(exec_robot, exec_p, ctrl_mode="teleport", record=False)

    primitive = Primitive(planner, executor, exec_body_blocks=env.env.body_blocks.copy(),
                          plan_body_blocks=inner_env.body_blocks.copy(), body_cliff0=env.env.body_cliff0,
                          body_cliff1=env.env.body_cliff1, body_tables=env.env.body_tables, verbose=2,
                          teleport_arm=False)
    primitive.align_at_reset()

    # for i in range(env.get_cur_num_objects()):
    #     res, path = primitive.move_one_object(i, env.get_block_reset_pos(i) + np.array([0., 0., 0.02]), [0, 0., 0., 1.])
    #     if res != 0:
    #         exit()
    # idx, tgt_pos, tgt_orn, out_of_reach = env.convert_action(np.array([0., 0.3437, -0.5625, 0.96875]))
    # alpha, beta, theta = tgt_orn * np.pi
    # obj_quaternion = np.array([np.cos(alpha) * np.cos(beta) * np.sin(theta / 2),
    #                            np.cos(alpha) * np.sin(beta) * np.sin(theta / 2),
    #                            np.sin(alpha) * np.sin(theta / 2), np.cos(theta / 2)])
    # tgt_pos[2] += 0.01
    # print(tgt_pos, obj_quaternion)
    # res, path = primitive.move_one_object(idx, tgt_pos, obj_quaternion)
    # if res != 0:
    #     exit()
    # res, path = primitive.move_one_object(0, (env.get_cliff_pos(0) + env.get_cliff_pos(1)) / 2 + np.array([0., 0., 0.05]), [np.sin(np.pi / 4), 0., 0., np.cos(np.pi / 4)])
    res, path = primitive.move_one_object(0, np.array([1.3, 0.72, 0.15]), [np.sin(np.pi / 4), 0., 0., np.cos(np.pi / 4)])
    if res != 0:
        exit()
    res, path = primitive.move_one_object(env.get_cur_num_objects() - 1, np.array([1.3, 0.32, 0.25]), [0., 0., 0., 1.])
    if res != 0:
        exit()
    # res, path = primitive.move_one_object(2, np.array([1.3, 0.6, 0.28]), [0., 0., 0., 1.])
    # if res != 0:
    #     exit()


if __name__ == "__main__":
    main()
