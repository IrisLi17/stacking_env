import pybullet as p 
import pybullet_utils.bullet_client as bc
from env.bullet_rotations import quat_diff
import numpy as np
import os


class XArmRobot(object):
    def __init__(self, physics_client: bc.BulletClient, init_qpos=None, base_pos=(0., 0., 0.), base_orn=(0, 0, 0, 1)):
        self.p = physics_client
        self.num_substeps = 60
        self.urdf_path = os.path.join(os.path.dirname(__file__), 'xarm_description/urdf/xarm7_with_gripper.urdf')
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.x_workspace = (base_pos[0] + 0.21, base_pos[0] + 0.61)
        self.y_workspace = (base_pos[1] - 0.2, base_pos[1] + 0.2)
        self.z_workspace = (base_pos[2] + 0.172, base_pos[2] + 0.4 + 0.172)
        self.init_eef_height = 0.2
        self.id = self.p.loadURDF(self.urdf_path, base_pos, base_orn, useFixedBase=True)
        if init_qpos is None:
            init_qpos = [0.] * self.p.getNumJoints(self.id)
        
        self.motor_indices = []
        self.joint_ll = []
        self.joint_ul = []
        self.joint_damping = []
        for j in range(self.p.getNumJoints(self.id)):
            joint_info = self.p.getJointInfo(self.id, j)
            if joint_info[2] != self.p.JOINT_FIXED:
                self.motor_indices.append(joint_info[0])
                self.joint_damping.append(joint_info[6])
                self.joint_ll.append(joint_info[8])
                self.joint_ul.append(joint_info[9])
        self.joint_ranges = [5] * len(self.joint_ll)
        self.rest_poses = (np.array(self.joint_ll) + np.array(self.joint_ul)) / 2
        self.eef_index = 8
        self.finger_drive_index = 10
        self.finger_range = (0, 0.85)
        # Create constraints for fingers
        # self._create_constraints()
        # con_children = [11]
        # con_multipliers = [-1, -1, 1, -1]
        # c = self.p.createConstraint(
        #     self.id, 10, self.id, 12, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0]
        # )
        # self.p.changeConstraint(c, gearRatio=1, maxForce=100, erp=1)
        # for c_idx in range(len(con_children)):
        #     c = self.p.createConstraint(
        #         self.id, 10, self.id, con_children[c_idx], self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        #     self.p.changeConstraint(c, gearRatio=con_multipliers[c_idx], maxForce=100, erp=1)
        # self.p.setJointMotorControlArray(
        #     self.id, [self.finger_drive_index, 13], self.p.POSITION_CONTROL, [0.5, 0.5],
        #     forces=[100, 100],
        # )
        # for _ in range(60):
        #     self.p.stepSimulation()
        #     time.sleep(0.1)
        # for i in range(10, 16):
        #     print(i, self.p.getJointState(self.id, i)[0])
        from env.kinematics import XArmKinematics
        self.kin = XArmKinematics(self.joint_ll[:7], self.joint_ul[:7])
    
    def _create_constraints(self):
        # gear between left outer knuckle and right outer knuckle
        c = self.p.createConstraint(self.id, 10, self.id, 13, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=1000)
        # gear between outer knuckle and finger
        # Failed
        # # joint position in parent frame (link 10)
        # parent_pos = self.p.getJointInfo(self.id, 11)[14]
        # # joint position in link 11 frame
        # child_pos = -np.array(self.p.getLinkState(self.id, 11)[2])
        # c = self.p.createConstraint(self.id, 10, self.id, 11, self.p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
        # self.p.changeConstraint(c, gearRatio=1, maxForce=100, erp=0.1)

    def get_obs(self):
        eef_state = self.p.getLinkState(self.id, self.eef_index, computeLinkVelocity=1, computeForwardKinematics=1)
        eef_pos, eef_orn, _, _, _, _, eef_vl, eef_va = eef_state
        eef_orn = quat_diff(eef_orn, np.array([1, 0, 0, 0]))
        eef_euler = self.p.getEulerFromQuaternion(eef_orn)
        eef_pos, eef_euler, eef_vl, eef_va = map(np.asarray, [eef_pos, eef_euler, eef_vl, eef_va])
        finger_position, finger_vel, *_ = self.p.getJointState(self.id, self.finger_drive_index)
        eef_vl *= 1. / 240 * self.num_substeps
        finger_vel *= 1. / 240 * self.num_substeps
        return np.concatenate([eef_pos, eef_euler, eef_vl, [finger_position, finger_vel]])

    def get_eef_position(self):
        eef_pos, *_ = self.p.getLinkState(self.id, self.eef_index)
        return np.array(eef_pos)

    def get_eef_orn(self, as_type="quat"):
        _, eef_orn, *_ = self.p.getLinkState(self.id, self.eef_index)
        if as_type == "euler":
            eef_orn = self.p.getEulerFromQuaternion(eef_orn)
        return np.array(eef_orn)

    def control(self, eef_pos, eef_orn, finger, relative=True, teleport=False):
        cur_eef_pos, *_ = self.p.getLinkState(self.id, self.eef_index)
        if relative:
            desired_eef_pos = np.array(cur_eef_pos) + eef_pos
        else:
            desired_eef_pos = np.array(eef_pos)
        if desired_eef_pos[2] < self.base_pos[2] + 0.172:
            desired_eef_pos[2] = self.base_pos[2] + 0.172
        assert np.linalg.norm(cur_eef_pos[:2] - self.base_pos[:2]) > 0.206
        while np.linalg.norm(desired_eef_pos[:2] - self.base_pos[:2]) < 0.206:
            desired_eef_pos[:2] = cur_eef_pos[:2] + 0.5 * (desired_eef_pos[:2] - cur_eef_pos[:2])
        # print(self.joint_ll, self.joint_ul, self.joint_ranges, self.rest_poses, self.joint_damping)
        # use ikfastpy
        solutions = self.kin.inverse(desired_eef_pos, eef_orn)
        if len(solutions) > 0:
            cur_joints = self._get_joint_positions(self.id, self.motor_indices[:7])
            _distances = [np.linalg.norm(np.array(cur_joints) - solution) for solution in solutions]
            joint_positions = solutions[np.argmin(_distances)]
            assert not np.any(np.isnan(joint_positions)), "IKFAST bug"
        else:
            return
            joint_positions = self.p.calculateInverseKinematics(
                self.id, self.eef_index, desired_eef_pos, eef_orn, 
                self.joint_ll[:7], self.joint_ul[:7], self.joint_ranges[:7], self.rest_poses[:7], self.joint_damping,
                maxNumIterations=100, residualThreshold=1e-4
            )
            if np.any(np.isnan(joint_positions)):
                print("skip step due to pybullet nan")
                return
            else:
                print("pybullet ik normal")
        # ik_pos, ik_orn = self.kin.forward(joint_positions[:7])
        # print("ik error", np.linalg.norm(ik_pos - desired_eef_pos))
        if teleport:
            for idx, j in enumerate(self.motor_indices[:7]):
                self.p.resetJointState(self.id, j, joint_positions[idx])
            for j in self.motor_indices[7:]:
                self.p.resetJointState(self.id, j, finger)
        else:
            self.p.setJointMotorControlArray(
                self.id, self.motor_indices[:7], self.p.POSITION_CONTROL, joint_positions[:7],
                forces=[1000] * 7,
                # positionGains=[1] * 7, velocityGains=[0.1] * 7
            )
            self.p.setJointMotorControlArray(
                self.id, self.motor_indices[7:], self.p.POSITION_CONTROL, [finger] * len(self.motor_indices[7:]),
                forces=[1000] * len(self.motor_indices[7:]),
                # positionGains=[1] * len(self.motor_indices[7:]), velocityGains=[0.1] * len(self.motor_indices[7:])
            )
            for i in range(self.num_substeps):
                self.p.stepSimulation()
#                if i % 10 == 0:
#                    achieved_positions = self._get_joint_positions(self.id, self.motor_indices)
#                    # print("control error at step", i, np.linalg.norm(np.array(achieved_positions) - np.array(
#                    #     list(joint_positions[:7]) + [finger] * (len(self.motor_indices) - 7))))
#                    if np.linalg.norm(np.array(achieved_positions) - np.array(
#                        list(joint_positions[:7]) + [finger] * (len(self.motor_indices) - 7))) < 1e-3:
#                        # print("early exit", i)
#                        break
            # print("control error", np.linalg.norm(np.array(achieved_positions) - np.array(
            #             list(joint_positions[:7]) + [finger] * (len(self.motor_indices) - 7))),
            #       np.linalg.norm(np.array(self.p.getLinkState(self.id, self.eef_index)[0]) - desired_eef_pos))
    
    def _get_joint_positions(self, body_id, joint_indices):
        joint_states = self.p.getJointStates(body_id, joint_indices)
        positions, *_ = zip(*joint_states)
        return positions

    def get_state(self):
        joint_states = self.p.getJointStates(self.id, list(range(self.p.getNumJoints(self.id))))
        joint_pos, joint_vel, *_ = zip(*joint_states)
        return dict(qpos=np.array(joint_pos), qvel=np.array(joint_vel))

    def set_state(self, state_dict):
        for j in range(self.p.getNumJoints(self.id)):
            self.p.resetJointState(self.id, j, state_dict["qpos"][j], state_dict["qvel"][j])


class PandaRobot(object):
    def __init__(self, physics_client: bc.BulletClient, init_qpos=None, base_pos=(0., 0., 0.), base_orn=(0, 0, 0, 1)):
        self.p = physics_client
        self.num_substeps = 20
        import pybullet_data
        data_root = pybullet_data.getDataPath()
        self.urdf_path = os.path.join(data_root, 'franka_panda/panda.urdf')
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.x_workspace = (base_pos[0] + 0.25, base_pos[0] + 0.6)
        self.y_workspace = (base_pos[1] - 0.25, base_pos[1] + 0.25)
        self.z_workspace = (base_pos[2], base_pos[2] + 0.4)
        self.init_eef_height = 0.05
        self.id = self.p.loadURDF(self.urdf_path, base_pos, base_orn, useFixedBase=True)
        if init_qpos is None:
            # init_qpos = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0, 0, 0.02, 0.02, 0]
            init_qpos = [0.0, 0.0, 0.0, -2.0, 0., 2.0, 1.0, 0, 0, 0.0, 0.0, 0]

        self.motor_indices = []
        self.joint_ll = []
        self.joint_ul = []
        self.joint_damping = []
        for j in range(self.p.getNumJoints(self.id)):
            self.p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            joint_info = self.p.getJointInfo(self.id, j)
            self.p.resetJointState(self.id, j, init_qpos[j])
            if joint_info[2] != self.p.JOINT_FIXED:
                self.motor_indices.append(joint_info[0])
                self.joint_damping.append(joint_info[6])
                self.joint_ll.append(joint_info[8])
                self.joint_ul.append(joint_info[9])

        self.joint_ranges = [5] * len(self.joint_ll)
        self.rest_poses = np.array(init_qpos)[self.motor_indices]
        self.eef_index = 11
        self.finger_drive_index = 9
        self.finger_range = (0, 0.04)
        c = self.p.createConstraint(self.id,
                                    9,
                                    self.id,
                                    10,
                                    jointType=self.p.JOINT_GEAR,
                                    jointAxis=[1, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def get_obs(self):
        eef_state = self.p.getLinkState(self.id, self.eef_index, computeLinkVelocity=1, computeForwardKinematics=1)
        eef_pos, eef_orn, _, _, _, _, eef_vl, eef_va = eef_state
        eef_orn = quat_diff(eef_orn, np.array([1, 0, 0, 0]))
        eef_euler = self.p.getEulerFromQuaternion(eef_orn)
        eef_pos, eef_euler, eef_vl, eef_va = map(np.asarray, [eef_pos, eef_euler, eef_vl, eef_va])
        finger_position, finger_vel, *_ = self.p.getJointState(self.id, self.finger_drive_index)
        eef_vl *= 1. / 240 * self.num_substeps
        finger_vel *= 1. / 240 * self.num_substeps
        return np.concatenate([eef_pos, eef_euler, eef_vl, [finger_position, finger_vel]])

    def get_eef_position(self):
        eef_pos, *_ = self.p.getLinkState(self.id, self.eef_index)
        return np.array(eef_pos)

    def get_eef_orn(self, as_type="quat"):
        _, eef_orn, *_ = self.p.getLinkState(self.id, self.eef_index)
        if as_type == "euler":
            eef_orn = self.p.getEulerFromQuaternion(eef_orn)
        return np.array(eef_orn)

    def control(self, eef_pos, eef_orn, finger, relative=True, teleport=False):
        # self._change_dynamics()
        cur_eef_pos, *_ = self.p.getLinkState(self.id, self.eef_index)
        if relative:
            desired_eef_pos = np.array(cur_eef_pos) + eef_pos
        else:
            desired_eef_pos = np.array(eef_pos)
        if desired_eef_pos[2] < 0:
            desired_eef_pos[2] = 0.
        # print(self.joint_ll, self.joint_ul, self.joint_ranges, self.rest_poses, self.joint_damping)
        joint_positions = self.p.calculateInverseKinematics(
            self.id, self.eef_index, desired_eef_pos, eef_orn,
            self.joint_ll[:7], self.joint_ul[:7], self.joint_ranges[:7], self.rest_poses[:7],
            maxNumIterations=20,
        )
        if np.any(np.isnan(joint_positions)):
            print("skip step due to pybullet nan")
            return
        # ik_pos, ik_orn = self.kin.forward(joint_positions[:7])
        # print("ik error", np.linalg.norm(ik_pos - desired_eef_pos))
        if teleport:
            for idx, j in enumerate(self.motor_indices[:7]):
                self.p.resetJointState(self.id, j, joint_positions[idx])
            for j in self.motor_indices[7:]:
                self.p.resetJointState(self.id, j, finger)
        else:
            self.p.setJointMotorControlArray(
                self.id, self.motor_indices[:7], self.p.POSITION_CONTROL, joint_positions[:7],
                forces=[1000] * 7,
                # positionGains=[1] * 7, velocityGains=[0.1] * 7
            )
            self.p.setJointMotorControlArray(
                self.id, self.motor_indices[7:], self.p.POSITION_CONTROL, [finger] * len(self.motor_indices[7:]),
                forces=[10] * len(self.motor_indices[7:]),
                # positionGains=[1] * len(self.motor_indices[7:]), velocityGains=[0.1] * len(self.motor_indices[7:])
            )
            for i in range(self.num_substeps):
                self.p.stepSimulation()
            # print(self.p.getLinkState(self.id, 11)[0])
            # joint_states = self.p.getJointStates(self.id, self.motor_indices)
            # positions, *_ = zip(*joint_states)
            # print(positions)
        # self._change_dynamics()

    def _change_dynamics(self):
        return
        self.p.performCollisionDetection()
        contact1 = self.p.getContactPoints(bodyA=self.id, linkIndexA=9)
        contact2 = self.p.getContactPoints(bodyA=self.id, linkIndexA=10)
        if len(contact1) == 0 and len(contact2) == 0:
            # Nothing grasped
            self.p.changeDynamics(self.id, 9, lateralFriction=1)
            self.p.changeDynamics(self.id, 10, lateralFriction=1)
        elif len(contact1) and len(contact2):
            _, _, bodyB1, _, linkB1, *_ = zip(*contact1)
            _, _, bodyB2, _, linkB2, *_ = zip(*contact2)
            grasped_id = list(set.intersection(set(bodyB1), set(bodyB2)))
            for obj_id in grasped_id:
                if np.linalg.norm(
                        np.array(self.p.getBasePositionAndOrientation(obj_id)[0]) -
                        np.array(self.p.getLinkState(self.id, self.eef_index)[0])) < 0.02:
                    # Grasp something
                    self.p.changeDynamics(self.id, 9, lateralFriction=10)
                    self.p.changeDynamics(self.id, 10, lateralFriction=10)
                    break

    def get_state(self):
        joint_states = self.p.getJointStates(self.id, list(range(self.p.getNumJoints(self.id))))
        joint_pos, joint_vel, *_ = zip(*joint_states)
        return dict(qpos=np.array(joint_pos), qvel=np.array(joint_vel))

    def set_state(self, state_dict):
        for j in range(self.p.getNumJoints(self.id)):
            self.p.resetJointState(self.id, j, state_dict["qpos"][j], state_dict["qvel"][j])


if __name__ == "__main__":
    from pybullet_utils import bullet_client as bc
    client = bc.BulletClient(connection_mode=p.GUI)
    client.resetSimulation()
    client.setTimeStep(1 / 240)
    client.setGravity(0., 0., -9.8)
    client.resetDebugVisualizerCamera(1.5, 80, -0, [0, 0, 0.2, ])
    init_qpos = None
    base_position = [0, 0, 0]
    robot = PandaRobot(client, init_qpos, base_position)
    robot.control([-0.05, 0.05, -0.15], [1, 0, 0, 0], 0.025, relative=True, teleport=False)
    halfExtents = [0.025, 0.025, 0.025]
    col_id = client.createCollisionShape(client.GEOM_BOX, halfExtents=halfExtents)
    vis_id = client.createVisualShape(client.GEOM_BOX, halfExtents=halfExtents)
    body_id = client.createMultiBody(0.1, col_id, vis_id, client.getLinkState(robot.id, 11)[0], (0, 0, 0, 1))
    robot.control([0, 0, 0], [1, 0, 0, 0], 0.01, relative=True, teleport=False)
    print(client.getDynamicsInfo(robot.id, 9))
    # lateral friction: 1, restitution: 0, rolling friction 0, spinning friction: 0.1, contact damping: 1000, contact stiffness: 30000,
    print(client.getDynamicsInfo(robot.id, 10))
    client.changeDynamics(robot.id, 9, lateralFriction=10)
    client.changeDynamics(robot.id, 10, lateralFriction=10)
    for i in range(50):
        robot.control([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), np.random.uniform(-0.02, 0.05)], [1, 0, 0, 0], 0.01, relative=True, teleport=False)
