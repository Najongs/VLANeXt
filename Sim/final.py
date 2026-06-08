import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time

# 1. Load Model
model_path = "meca_add_3trocar.xml"
print(f"Loading Model: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# Select target trocar at startup: A=red (0°)  /  B=green (+80°)
import sys
if len(sys.argv) > 1 and sys.argv[1].upper() in ("A", "B"):
    TARGET_PORT = sys.argv[1].upper()
else:
    _sel = input("Target port (A=red / B=green) [A]: ").strip().upper()
    TARGET_PORT = _sel if _sel in ("A", "B") else "A"
print(f">>> Target port: {TARGET_PORT}")

# 2. Get IDs
try:
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"trocar_entry_{TARGET_PORT}")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"trocar_back_{TARGET_PORT}")
    viz_tip_tgt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "viz_target_tip")

    phantom_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly") 
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")   

    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link") 

    dof = model.nv
except Exception as e:
    print(f"[Error] XML ID Load Failed: {e}")
    raise e

# === Control Parameters ===
damping = 1e-3
base_speed = 1.0

TARGET_DISTANCE_FROM_ENTRY = 0.0001
TARGET_INSERTION_DEPTH = 0.022
SENSOR_THRESHOLD = 0.010

# [NEW] 속도 고정 파라미터 (m/s) -> 초당 1cm 이동
ALIGN_SPEED = 0.05

# A1: Standoff — entry에서 3cm 떨어진 trocar 축 위 점을 중간 목표로 두고,
#     standoff → entry 구간은 *순수 axis 방향 직선*이 되도록 → 진입각이 부드러워짐.
STANDOFF_DIST = 0.03

# B3: Soft DLS — vanilla pinv를 거의 그대로 두되 σ→0 근처에서만 댐핑.
#     λ=0.005는 정밀도 손실 거의 없음, 싱귤러 폭주만 막음.
DLS_LAMBDA = 0.005
def dls_inv(J, lam=DLS_LAMBDA):
    m, n = J.shape
    if m <= n:
        return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))
    else:
        return np.linalg.inv(J.T @ J + (lam ** 2) * np.eye(n)) @ J.T

# B1: Adaptive home pose — target port별 작업영역에 맞춘 시작자세 (싱귤러 회피용).
#     port B는 phantom local X 80° 회전이라 J1/J5 다르게 잡아야 함. viewer로 튜닝.
HOME_POSE_TABLE = {
    "A": np.array([0.8, -0.5, 0.5, 0, 0.65, 1]),
    "B": np.array([0.5, -0.5, 0.5, 0, 0.65, 1]),
}

# State Variables
task_state = 1
phase1_substate = "standoff"     # "standoff" → "approach"
align_timer = 0
insertion_started = False
accumulated_depth = 0.0
phase3_base_tip = np.zeros(3)
hold_start_time = None

# === Trajectory Variables ===
traj_initialized = False
traj_start_time = 0.0
DYNAMIC_DURATION = 0.0 # [NEW] 거리에 따라 유동적으로 계산될 시간 변수
start_tip_pos = np.zeros(3)
start_back_pos = np.zeros(3)

def randomize_phantom_pos(model, data, phantom_id, rot_id):
    offset_x = 0.0 # np.random.uniform(-0.05, 0.05) 
    offset_y = 0.0 # np.random.uniform(-0.4, 0.0)   
    offset_z = 0.0 
    
    model.body_pos[phantom_id] = np.array([offset_x, offset_y, offset_z])
    
    if offset_y >= -0.25:
        random_angle_deg = np.random.uniform(-15, 15)
    else:
        random_angle_deg = np.random.uniform(-15 - 90, 15 - 90)
        
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    
    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)
    
    return np.array([offset_x, offset_y, offset_z], dtype=np.float32), new_quat.astype(np.float32), np.float32(random_angle_deg)


# === Helper: SmoothStep Function ===
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

randomize_phantom_pos(model, data, phantom_pos_id, rotating_id)

print("Started: Smooth Trajectory Mode with Ray-Casting Sensor")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Home pose 
    # home_pose = np.array([np.random.uniform(-0.45, 0.55), np.random.uniform(-0.4, -0.3), np.random.uniform(0.3, 0.4), 0.0000, np.random.uniform(0.45, 0.55), np.random.uniform(0.95, 1.05)])
    home_pose = HOME_POSE_TABLE[TARGET_PORT].copy()    # B1: target port별 자세
    # home_pose = np.array([0.0, 0.0, 0.0, 0, 0.75, 1])
    # home_pose = np.array([
    #     np.random.uniform(-0.5, 0.5),    # J1 (base rotation)
    #     np.random.uniform(-0.6, -0.4),    # J2 (shoulder pitch)
    #     np.random.uniform(0.75, 0.25),    # J3 (elbow pitch)
    #     np.random.uniform(-0.3, 0.3),    # J4 (roll)
    #     np.random.uniform(0.4, 0.6),     # J5 (wrist pitch)
    #     np.random.uniform(0.9, 1.1),    # J6
    #     ])
    data.qpos[:6] = home_pose  
    mujoco.mj_forward(model, data)
    needle_len = np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[back_id])

    step_count = 0
    target_tip_pos = data.site_xpos[tip_id].copy()
    target_back_pos = data.site_xpos[back_id].copy()
    
    while viewer.is_running():
        step_start = time.time()
        curr_time = data.time
        
        # Current States
        p_entry = data.site_xpos[target_entry_id].copy()
        p_depth = data.site_xpos[target_depth_id].copy()
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        p_sensor = data.site_xpos[sensor_id].copy()

        # Target Vectors
        axis_vec = p_depth - p_entry
        axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)

        # ==========================================
        # Ray-Casting Sensor Logic
        # ==========================================
        needle_dir = (curr_tip - curr_back)
        needle_dir /= (np.linalg.norm(needle_dir) + 1e-10)
        geom_id_out = np.zeros(1, dtype=np.int32) 
        dist_to_surface = mujoco.mj_ray(model, data, p_sensor, needle_dir, 
                                        None, 1, link6_id, geom_id_out)

        obstacle_detected = False
        sensor_msg = "Sensor: Clear"
        
        if 0 <= dist_to_surface <= SENSOR_THRESHOLD:
            obstacle_detected = True
            sensor_msg = f"OBSTACLE! ({dist_to_surface*1000:.1f}mm)"

        if task_state == 1:
            status_color = (255, 0, 255)
            msg = "Aligning..."
        elif task_state == 2:
            status_color = (0, 255, 0)
            msg = "Inserting..."
        else:
            status_color = (0, 0, 255)
            msg = "Finished"

        if obstacle_detected:
            status_color = (0, 165, 255) 
            msg = sensor_msg

        current_speed = 0.1 if obstacle_detected else base_speed

        # === State Machine ===
        if task_state == 1:
            # [Phase 1a: Approach to standoff (axis 위 3cm out) — 자유롭게 회전+이동]
            # [Phase 1b: Standoff → entry (순수 axis 방향 직선) — 진입각 부드러움 보장]
            current_offset = STANDOFF_DIST if phase1_substate == "standoff" else TARGET_DISTANCE_FROM_ENTRY

            if not traj_initialized:
                traj_start_time = curr_time
                start_tip_pos = curr_tip.copy()
                start_back_pos = curr_back.copy()

                goal_tip_temp = p_entry - (axis_dir * current_offset)
                distance = np.linalg.norm(goal_tip_temp - start_tip_pos)
                DYNAMIC_DURATION = max(distance / ALIGN_SPEED, 0.3)

                traj_initialized = True
                print(f">>> [{phase1_substate}] Trajectory Generated. Distance: {distance*1000:.1f}mm | Duration: {DYNAMIC_DURATION:.2f}sec")

            elapsed_t = curr_time - traj_start_time
            raw_progress = elapsed_t / DYNAMIC_DURATION if DYNAMIC_DURATION > 0 else 1.0
            alpha = smooth_step(raw_progress)

            goal_tip = p_entry - (axis_dir * current_offset)
            goal_back = p_entry - (axis_dir * (current_offset + needle_len))

            target_tip_pos = (1 - alpha) * start_tip_pos + alpha * goal_tip
            target_back_pos = (1 - alpha) * start_back_pos + alpha * goal_back

            if not obstacle_detected:
                msg = f"Aligning [{phase1_substate}]... {alpha*100:.1f}%"

            if raw_progress >= 1.0:
                dist_error = np.linalg.norm(curr_tip - goal_tip)

                if dist_error < 0.002:
                    align_timer += 1
                    if not obstacle_detected:
                        msg = f"Holding [{phase1_substate}]..."
                else:
                    align_timer = 0

                if align_timer > 20:
                    if phase1_substate == "standoff":
                        phase1_substate = "approach"
                        traj_initialized = False     # 새 trajectory 다시 만들기
                        align_timer = 0
                        print(">>> Standoff reached. Approaching entry along axis.")
                    else:
                        task_state = 2
                        insertion_started = False
                        print(">>> Alignment Complete. Starting Insertion.")

        elif task_state == 2:
            # [Phase 2: Insertion & Hold]
            if not insertion_started:
                phase3_base_tip = curr_tip.copy() 
                insertion_started = True
                accumulated_depth = 0.0
                hold_start_time = None

            step_z = 0.000025 
            
            if accumulated_depth < TARGET_INSERTION_DEPTH:
                if not obstacle_detected: msg = f"Inserting ({accumulated_depth*1000:.1f}mm)"
                accumulated_depth += step_z
                
                target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
                
                if accumulated_depth >= TARGET_INSERTION_DEPTH:
                    hold_start_time = curr_time
                    print(">>> Reached Target Depth. Holding for 1 sec...")
            
            else:
                if hold_start_time is None:
                    hold_start_time = curr_time
                    
                target_tip_pos = phase3_base_tip + (axis_dir * TARGET_INSERTION_DEPTH)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
                
                if not obstacle_detected: msg = "Holding Position..."
                
                if curr_time - hold_start_time >= 1.0:
                    task_state = 3
                    print(">>> Task Finished Successfully.")

        elif task_state == 3:
            msg = "FINISHED"

        # === Stacked IK Solver ===
        data.site_xpos[viz_tip_tgt_id] = target_tip_pos
        
        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back
        
        tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
        offset_angle = np.deg2rad(180+30)
        offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
        current_side_vec = tip_rot_mat @ offset_local_vec
        
        needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
        if np.linalg.norm(target_side_vec) < 1e-3: target_side_vec = np.array([1, 0, 0])
        else: target_side_vec /= np.linalg.norm(target_side_vec)
        err_roll = np.cross(current_side_vec, target_side_vec)
        
        jac_tip_full = np.zeros((6, dof))
        jac_back = np.zeros((3, dof))
        mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
        mujoco.mj_jacSite(model, data, jac_back, None, back_id)
        
        J_p1 = jac_tip_full[:3]
        e_p1 = err_tip * 50.0
        if np.linalg.norm(e_p1) > 1.0: e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0

        J_p1_pinv = dls_inv(J_p1)                         # B3: soft DLS (λ=0.005)
        dq_p1 = J_p1_pinv @ e_p1
        P_null_1 = np.eye(dof) - (J_p1_pinv @ J_p1)

        J_p2 = jac_back
        e_p2 = err_back * 50.0
        J_p2_proj = J_p2 @ P_null_1
        J_p2_pinv = dls_inv(J_p2_proj)
        dq_p2 = J_p2_pinv @ (e_p2 - J_p2 @ dq_p1)
        P_null_2 = P_null_1 - (J_p2_pinv @ J_p2_proj)

        J_p3 = jac_tip_full[3:]
        e_p3 = err_roll * 10.0
        J_p3_proj = J_p3 @ P_null_2
        J_p3_pinv = dls_inv(J_p3_proj)
        dq_p3 = J_p3_pinv @ (e_p3 - J_p3 @ (dq_p1 + dq_p2))

        dq = dq_p1 + dq_p2 + dq_p3

        data.ctrl[:] = data.qpos[:dof] + (dq * current_speed)
        mujoco.mj_step(model, data)
        step_count += 1
        
        if step_count % 67 == 0:
            frames = []
            for cam in ["side_camera", "tool_camera", "top_camera"]:
                renderer.update_scene(data, camera=cam)
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            combined = np.hstack(frames)
            
            cv2.putText(combined, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("Smooth Trajectory & Ray Sensor", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                mujoco.mj_resetData(model, data)

                # 리셋 시 target port에 맞는 home pose 다시 적용
                home_pose = HOME_POSE_TABLE[TARGET_PORT].copy()
                data.qpos[:6] = home_pose

                randomize_phantom_pos(model, data, phantom_pos_id, rotating_id)
                mujoco.mj_forward(model, data)
                task_state = 1
                phase1_substate = "standoff"      # A1: substate도 리셋
                traj_initialized = False
                align_timer = 0
                insertion_started = False
                accumulated_depth = 0.0
                hold_start_time = None

        viewer.sync()
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
 
cv2.destroyAllWindows()