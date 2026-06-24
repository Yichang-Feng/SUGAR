import sys
import os

with open("deploy/scripts/sim2sim_ros2.py", "r") as f:
    content = f.read()

# 1. Imports
imports = """
import cv2
from pupil_apriltags import Detector
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class Sim2SimRosNode(Node):
    def __init__(self):
        super().__init__('sim2sim_ros2_node')
        self.image_pub = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.obj_pose_pub = self.create_publisher(PoseStamped, 'object_pose_torso', 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, 'target_pose_torso', 10)
        self.bridge = CvBridge()

    def publish_image(self, img_rgb):
        msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.image_pub.publish(msg)

    def publish_pose(self, pub, pos, quat_wxyz):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "torso_link"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.orientation.z = float(quat_wxyz[3])
        pub.publish(msg)

"""

content = content.replace("import torch\n", "import torch\n" + imports)

# 2. Add ROS init and variables
ros_init = """
    rclpy.init()
    ros_node = Sim2SimRosNode()

    # Setup Rendering
    cam_name = "depth_camera"
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    cam_params = [f, f, width / 2, height / 2]
    detector = Detector(families='tag36h11')

    latest_vision_obj_pos_b = np.array([0.3, 0.0, 0.1])
    latest_vision_obj_ori_b = np.array([1.0, 0.0, 0.0, 0.0])
    latest_vision_target_pos_b = np.array([0.3, 0.0, 0.0])
    latest_vision_target_ori_b = np.array([1.0, 0.0, 0.0, 0.0])
"""
content = content.replace("    generator = None", ros_init + "\n    generator = None")

# 3. Replace generator obs variables
old_gen_obs = """            p_obj = data.xpos[box_body_id]
            p_torso = data.xpos[torso_body_id]
            R_torso = data.xmat[torso_body_id].reshape(3, 3)
            
            q_torso = np.zeros(4)
            mujoco.mju_mat2Quat(q_torso, data.xmat[torso_body_id])
            q_obj = np.zeros(4)
            mujoco.mju_mat2Quat(q_obj, data.xmat[box_body_id])
            
            obj_pos_b_val = R_torso.T @ (p_obj - p_torso)
            obj_ori_b_val = quat_mul(quat_inv(q_torso), q_obj)
            
            joint_pos_val = np.array([data.qpos[adr] for adr in joint_qpos_indices]) - q_default
            
            R_pelvis_val = data.xmat[pelvis_body_id].reshape(3, 3)
            project_gravity_val = R_pelvis_val.T @ np.array([0, 0, -1.0])
            
            # Target is the last frame of reference motion
            obj_target_pos_w = ref_data["obj_pos"][-1]
            obj_target_quat_w = ref_data["obj_quat"][-1]
            
            target_obj_pos_b_val = R_torso.T @ (obj_target_pos_w - p_torso)
            target_obj_ori_b_val = quat_mul(quat_inv(q_torso), obj_target_quat_w)"""

new_gen_obs = """            p_torso = data.xpos[torso_body_id]
            R_torso = data.xmat[torso_body_id].reshape(3, 3)
            
            obj_pos_b_val = latest_vision_obj_pos_b
            obj_ori_b_val = latest_vision_obj_ori_b
            
            joint_pos_val = np.array([data.qpos[adr] for adr in joint_qpos_indices]) - q_default
            
            R_pelvis_val = data.xmat[pelvis_body_id].reshape(3, 3)
            project_gravity_val = R_pelvis_val.T @ np.array([0, 0, -1.0])
            
            target_obj_pos_b_val = latest_vision_target_pos_b
            target_obj_ori_b_val = latest_vision_target_ori_b"""

content = content.replace(old_gen_obs, new_gen_obs)


# 4. Modify run_step
old_run_step = """    # Modular step function for simulation control loop
    def run_step(frame_idx, gen_cmd, gen_preds):
        nonlocal gen_history
        
        # 1. Update State Variables
        R_pelvis = data.xmat[pelvis_body_id].reshape(3, 3)
        curr_base_ang_vel = data.qvel[3:6].copy()
        
        q_actual = np.array([data.qpos[adr] for adr in joint_qpos_indices])
        v_actual = np.array([data.qvel[adr] for adr in joint_qvel_indices])
        curr_joint_pos_rel = q_actual - q_default
        curr_joint_vel = v_actual
        curr_project_gravity = R_pelvis.T @ np.array([0, 0, -1.0])
        
        # Object pose relative to Torso Link
        p_obj = data.xpos[box_body_id]
        p_torso = data.xpos[torso_body_id]
        R_torso = data.xmat[torso_body_id].reshape(3, 3)
        
        obj_pos_b = R_torso.T @ (p_obj - p_torso)
        
        R_obj = data.xmat[box_body_id].reshape(3, 3)
        R_obj_torso = R_torso.T @ R_obj
        obj_ori_b = R_obj_torso[:, :2].flatten() # first 2 columns (6 elements)"""

new_run_step = """    # Modular step function for simulation control loop
    def run_step(frame_idx, gen_cmd, gen_preds):
        nonlocal gen_history
        nonlocal latest_vision_obj_pos_b, latest_vision_obj_ori_b, latest_vision_target_pos_b, latest_vision_target_ori_b
        
        # --- Randomize box position slightly occasionally ---
        if frame_idx > 0 and frame_idx % 200 == 0:
            box_qpos_adr = model.joint("floating_box_base_joint").qposadr[0]
            box_dof_adr = model.joint("floating_box_base_joint").dofadr[0]
            data.qpos[box_qpos_adr] = 10.0 + np.random.uniform(-0.05, 0.05)
            data.qpos[box_qpos_adr+1] = 10.0 + np.random.uniform(-0.05, 0.05)
            data.qvel[box_dof_adr:box_dof_adr+6] = 0.0
            print(f"[Sim2Sim] Box position randomized")

        # --- Render and Vision ---
        renderer.update_scene(data, camera=cam_name)
        img_rgb = renderer.render()
        ros_node.publish_image(img_rgb)
        
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        detections = detector.detect(img_gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=0.03)
        
        T_cv2mj = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float32)
        T_tag2obj = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        
        cam_pos_w = data.cam_xpos[cam_id]
        cam_mat_w = data.cam_xmat[cam_id].reshape(3, 3)
        p_torso = data.xpos[torso_body_id]
        R_torso = data.xmat[torso_body_id].reshape(3, 3)

        for det in detections:
            pos_mj = T_cv2mj @ det.pose_t.flatten()
            rot_mj = T_cv2mj @ det.pose_R @ np.linalg.inv(T_tag2obj)
            
            tag_pos_w = cam_pos_w + cam_mat_w @ pos_mj
            tag_rot_w = cam_mat_w @ rot_mj
            
            pos_b = R_torso.T @ (tag_pos_w - p_torso)
            rot_b = R_torso.T @ tag_rot_w
            quat_b = matrix_to_quat(rot_b)
            
            if det.tag_id == 0:
                # Apply Z-offset to get center of object
                pos_b[2] -= 0.175  # The tag is at Z=0.1755, subtract half-height to get object center
                latest_vision_obj_pos_b = pos_b
                latest_vision_obj_ori_b = quat_b
                ros_node.publish_pose(ros_node.obj_pose_pub, latest_vision_obj_pos_b, latest_vision_obj_ori_b)
            elif det.tag_id == 1:
                latest_vision_target_pos_b = pos_b
                latest_vision_target_ori_b = quat_b
                ros_node.publish_pose(ros_node.target_pose_pub, latest_vision_target_pos_b, latest_vision_target_ori_b)
        
        rclpy.spin_once(ros_node, timeout_sec=0)

        # 1. Update State Variables
        R_pelvis = data.xmat[pelvis_body_id].reshape(3, 3)
        curr_base_ang_vel = data.qvel[3:6].copy()
        
        q_actual = np.array([data.qpos[adr] for adr in joint_qpos_indices])
        v_actual = np.array([data.qvel[adr] for adr in joint_qvel_indices])
        curr_joint_pos_rel = q_actual - q_default
        curr_joint_vel = v_actual
        curr_project_gravity = R_pelvis.T @ np.array([0, 0, -1.0])
        
        obj_pos_b = latest_vision_obj_pos_b
        
        # Extract 6D orientation from quaternion for Tracker
        rot_matrix = Rotation.from_quat([latest_vision_obj_ori_b[1], latest_vision_obj_ori_b[2], latest_vision_obj_ori_b[3], latest_vision_obj_ori_b[0]]).as_matrix()
        obj_ori_b = rot_matrix[:, :2].flatten()
        p_obj = tag_pos_w if 'tag_pos_w' in locals() else data.xpos[box_body_id]
"""

content = content.replace(old_run_step, new_run_step)

with open("deploy/scripts/sim2sim_ros2.py", "w") as f:
    f.write(content)

