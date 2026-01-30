#======================================================================================
# LIPM 기반 Cubic Spline Curve 이족보행 궤적 생성
#======================================================================================

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time
import crocoddyl
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#======================================================================================
# URDF 로드
#======================================================================================
urdf_path = "./bipedal_v3_stable.URDF"
model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
data = model.createData()

state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)
L_FOOT_ID = model.getFrameId("l_foot_link")
R_FOOT_ID = model.getFrameId("r_foot_link")

def deg2rad(deg):
    return deg * np.pi / 180.0

hip_roll_yaw_indices = [("l_hip_yaw_joint", 6), ("l_hip_roll_joint", 7), 
                        ("r_hip_yaw_joint", 12), ("r_hip_roll_joint", 13)]

#======================================================================================
# Physical Constraints
#======================================================================================
MAX_COM_ACC_XY = 3.0    # m/s²
MAX_COM_ACC_Z = 2.0     # m/s²
MAX_COM_VEL_XY = 0.3    # m/s
MAX_COM_JERK = 50.0     # m/s³

#======================================================================================
# Initial Configuration
#======================================================================================
print("\n=== Setting up initial configuration ===")

initialHipPitch = 30.0
initialKneePitch = -60.0
initialAnklePitch = 30.0

q_init = pin.neutral(model)
q_init[2] = 0.70

q_init[model.joints[model.getJointId(f"l_hip_pitch_joint")].idx_q] = deg2rad(initialHipPitch)
q_init[model.joints[model.getJointId(f"r_hip_pitch_joint")].idx_q] = deg2rad(initialHipPitch)
q_init[model.joints[model.getJointId(f"l_knee_joint")].idx_q] = deg2rad(initialKneePitch)
q_init[model.joints[model.getJointId(f"r_knee_joint")].idx_q] = deg2rad(initialKneePitch)
q_init[model.joints[model.getJointId(f"l_ankle_pitch_joint")].idx_q] = deg2rad(initialAnklePitch)
q_init[model.joints[model.getJointId(f"r_ankle_pitch_joint")].idx_q] = deg2rad(initialAnklePitch)

pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)

foot_z_offset = max(data.oMf[L_FOOT_ID].translation[2], data.oMf[R_FOOT_ID].translation[2])
desired_base_height = q_init[2] - foot_z_offset + 0.01
q_init[2] = desired_base_height
q_init[3:7] = [0, 0, 0, 1]

pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
pin.centerOfMass(model, data, q_init)

foot_width = abs(data.oMf[L_FOOT_ID].translation[1] - data.oMf[R_FOOT_ID].translation[1])
com_height = data.com[0][2]
#======================================================================================
# Trajectory Smoothing Utilities
#======================================================================================
def generate_smooth_com_trajectory(waypoints, durations, dt):
    # Qubic Spline 보간법 : 주어진 데이터 점들을 통과하는 부드러운 곡선 생성

    # Build time array : 시간 누적 합
    times = [0.0]
    for d in durations:
        times.append(times[-1] + d)
    times = np.array(times)

    waypoints = np.array(waypoints)

    # Create cubic spline (natural boundary conditions)
    cs_x = CubicSpline(times, waypoints[:, 0], bc_type='natural')
    cs_y = CubicSpline(times, waypoints[:, 1], bc_type='natural')
    
    # Sample at dt intervals
    t_samples = np.arange(0, times[-1], dt) #dt 간격으로 이산화
    n_samples = len(t_samples) 
    
    com_traj = np.zeros((n_samples, 2))
    com_vel = np.zeros((n_samples, 2))
    com_acc = np.zeros((n_samples, 2))
    
    for i, t in enumerate(t_samples):
        com_traj[i, 0] = cs_x(t)
        com_traj[i, 1] = cs_y(t)
        com_vel[i, 0] = cs_x(t, 1)
        com_vel[i, 1] = cs_y(t, 1)
        com_acc[i, 0] = cs_x(t, 2)
        com_acc[i, 1] = cs_y(t, 2)
    
    max_acc_x = np.max(np.abs(com_acc[:, 0]))
    max_acc_y = np.max(np.abs(com_acc[:, 1]))
    
    if max_acc_x > MAX_COM_ACC_XY or max_acc_y > MAX_COM_ACC_XY:
        print(f"  WARNING: Acceleration exceeds limit")
        # Time scaling factor
        #a_new = a_old / scale²  -> scale = sqrt(a_old / a_new)
        scale = max(max_acc_x, max_acc_y) / MAX_COM_ACC_XY
        scale = np.sqrt(scale)
        
        # 시간 스케일 만큼 늘리기
        new_duration = times[-1] * scale
        t_samples_new = np.arange(0, new_duration, dt)
        n_samples_new = len(t_samples_new)
        
        # Rebuild with scaled durations
        durations_scaled = [d * scale for d in durations]
        times_scaled = [0.0]
        for d in durations_scaled:
            times_scaled.append(times_scaled[-1] + d)
        times_scaled = np.array(times_scaled)
        
        cs_x = CubicSpline(times_scaled, waypoints[:, 0], bc_type='natural')
        cs_y = CubicSpline(times_scaled, waypoints[:, 1], bc_type='natural')
        
        com_traj = np.zeros((n_samples_new, 2))
        com_vel = np.zeros((n_samples_new, 2))
        com_acc = np.zeros((n_samples_new, 2))
        
        for i, t in enumerate(t_samples_new):
            com_traj[i, 0] = cs_x(t)
            com_traj[i, 1] = cs_y(t)
            com_vel[i, 0] = cs_x(t, 1)
            com_vel[i, 1] = cs_y(t, 1)
            com_acc[i, 0] = cs_x(t, 2)
            com_acc[i, 1] = cs_y(t, 2)
        
        print(f"  Time scaled by {scale:.2f}x, new duration: {new_duration:.2f}s")
        print(f"  Scaled max acceleration: ax={np.max(np.abs(com_acc[:,0])):.2f}, ay={np.max(np.abs(com_acc[:,1])):.2f} m/s²")
        
        return com_traj, com_vel, com_acc, durations_scaled
    
    return com_traj, com_vel, com_acc, durations


def generate_swing_trajectory(start, end, height, n_points):
    trajectory = []
    T = n_points - 1
    
    for i in range(n_points):
        t = i / max(T, 1)
        
        # XY: quintic polynomial for smooth start/end
        poly = 10*t**3 - 15*t**4 + 6*t**5
        x = start[0] + (end[0] - start[0]) * poly
        y = start[1] + (end[1] - start[1]) * poly        
        # Z: sine curve for natural arc, z는 원점으로 돌아와야함
        z = height * np.sin(np.pi * t) ** 2        
        trajectory.append(np.array([x, y, z])) 
    return trajectory

#======================================================================================
# Footstep Planning
#======================================================================================
# Parameters
step_length = 0.15   
step_height = 0.05  
dt = 0.01
# Timing
T_init = 0.8            # 초기 안정화
T_shift = 0.6           # Weight shift
T_swing = 0.8           # Swing phase
T_ds = 0.4              # Double support

# Steps
num_steps = 5

pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
init_left = data.oMf[L_FOOT_ID].translation.copy()
init_right = data.oMf[R_FOOT_ID].translation.copy()

# CoM waypoints와 durations 정의
com_waypoints =[]
waypoint_durations = []
footstep_sequence = []

curr_left = init_left.copy()
curr_right = init_right.copy()

center = (curr_left[:2] + curr_right[:2]) / 2
#Initial Double Support
com_waypoints.append(center.copy())
footstep_sequence.append({'type': 'DS_INIT', 'left': curr_left.copy(), 'right': curr_right.copy() })
waypoint_durations.append(T_init)

support_foot = 'left'

#Step Sequence Generation
for step_i in range(num_steps):
    if support_foot == 'left':
        # Shift to left foot
        com_waypoints.append(curr_left[:2].copy() * 0.9 + center * 0.1)
        waypoint_durations.append(T_shift)
        footstep_sequence.append({'type': 'SHIFT_L', 'left': curr_left.copy(), 'right': curr_right.copy()})
        
        # Swing right foot
        next_right = curr_right.copy()
        next_right[0] = curr_left[0] + step_length
        com_waypoints.append(curr_left[:2].copy() * 0.9 + center * 0.1)
        waypoint_durations.append(T_swing)
        footstep_sequence.append({'type': 'SWING_R', 'left': curr_left.copy(), 'right_start': curr_right.copy(), 'right_end': next_right.copy()})
        
        # Double support - move to center
        curr_right = next_right.copy()
        center = (curr_left[:2] + curr_right[:2]) / 2
        com_waypoints.append(center.copy())
        waypoint_durations.append(T_ds)
        footstep_sequence.append({'type': 'DS', 'left': curr_left.copy(), 'right': curr_right.copy()})
    
        support_foot = 'right'
        
    else:
        # Shift to right foot
        com_waypoints.append(curr_right[:2].copy() * 0.9 + center * 0.1)
        waypoint_durations.append(T_shift)
        footstep_sequence.append({'type': 'SHIFT_R', 'left': curr_left.copy(), 'right': curr_right.copy()})
        
        # Swing left foot
        next_left = curr_left.copy()
        next_left[0] = curr_right[0] + step_length
        com_waypoints.append(curr_right[:2].copy() * 0.9 + center * 0.1)
        waypoint_durations.append(T_swing)
        footstep_sequence.append({'type': 'SWING_L', 'left_start': curr_left.copy(), 'left_end': next_left.copy(), 'right': curr_right.copy()})
        
        # Double support
        curr_left = next_left.copy()
        center = (curr_left[:2] + curr_right[:2]) / 2
        com_waypoints.append(center.copy())
        waypoint_durations.append(T_ds)
        footstep_sequence.append({'type': 'DS', 'left': curr_left.copy(), 'right': curr_right.copy()})
        
        support_foot = 'left'

#======================================================================================
#왼 발 자기 위치
#======================================================================================
# Shift to right foot
'''
com_waypoints.append(curr_right[:2].copy() * 0.9 + center * 0.1)
waypoint_durations.append(T_shift)
footstep_sequence.append({'type': 'SHIFT_R', 'left': curr_left.copy(), 'right': curr_right.copy()})
        
# Swing left foot
next_left = curr_left.copy()
next_left[0] = curr_right[0]
com_waypoints.append(curr_right[:2].copy() * 0.9 + center * 0.1)
waypoint_durations.append(T_swing)
footstep_sequence.append({'type': 'SWING_L', 'left_start': curr_left.copy(), 'left_end': next_left.copy(), 'right': curr_right.copy()})
        
# Double support
curr_left = next_left.copy()
center = (curr_left[:2] + curr_right[:2]) / 2
com_waypoints.append(center.copy())
waypoint_durations.append(T_ds)
footstep_sequence.append({'type': 'DS', 'left': curr_left.copy(), 'right': curr_right.copy()})       
support_foot = 'left'
'''
#======================================================================================
#오른 발 자기 위치
#======================================================================================

# Shift to right foot
com_waypoints.append(curr_left[:2].copy() * 0.9 + center * 0.1)
waypoint_durations.append(T_shift)
footstep_sequence.append({'type': 'SHIFT_L', 'left': curr_left.copy(), 'right': curr_right.copy()})
        
# Swing right foot
next_right = curr_right.copy()
next_right[0] = curr_left[0]
com_waypoints.append(curr_left[:2].copy() * 0.9 + center * 0.1)
waypoint_durations.append(T_swing)
footstep_sequence.append({'type': 'SWING_R', 'left': curr_left.copy(), 'right_start': curr_right.copy(), 'right_end': next_right.copy()})
        
# Double support - move to center
curr_right = next_right.copy()
center = (curr_left[:2] + curr_right[:2]) / 2
com_waypoints.append(center.copy())
waypoint_durations.append(T_ds)
footstep_sequence.append({'type': 'DS', 'left': curr_left.copy(), 'right': curr_right.copy()})


# Final waypoint (no duration after - this is the end point)
com_waypoints.append(center.copy())
#footstep_sequence.append({'type': 'DS_FINAL', 'left': curr_left.copy(), 'right': curr_right.copy()})

com_waypoints = np.array(com_waypoints)
print(f"  CoM waypoints: {len(com_waypoints)}")
print(f"  Footstep sequence: {len(footstep_sequence)}")
#======================================================================================
# Generate Smooth CoM Trajectory
#======================================================================================

com_traj, com_vel, com_acc, durations_final = generate_smooth_com_trajectory(com_waypoints, waypoint_durations, dt)

total_time = sum(durations_final)
n_phases = len(com_traj)

#======================================================================================
# Phase Sequence
#======================================================================================
phases = []
time_boundaries = [0.0]
for d in durations_final: #각 phase 시간
    time_boundaries.append(time_boundaries[-1] + d)

for i in range(n_phases):
    t = i * dt
    
    # Find current footstep
    step_idx = 0
    for j, tb in enumerate(time_boundaries[1:]):
        if t < tb:
            step_idx = j
            break
        step_idx = j
    
    step = footstep_sequence[min(step_idx, len(footstep_sequence)-1)]
    step_type = step['type']
    
    # Time within current step
    t_step_start = time_boundaries[step_idx]
    t_step_end = time_boundaries[min(step_idx+1, len(time_boundaries)-1)]
    T_step = t_step_end - t_step_start
    t_ratio = (t - t_step_start) / max(T_step, dt)
    
    phase = {
        'time': t,
        'step_type': step_type,
        'com_x_target': com_traj[i, 0],
        'com_y_target': com_traj[i, 1],
        'com_vx': com_vel[i, 0],
        'com_vy': com_vel[i, 1]
    }
    
    if step_type in ['DS_INIT', 'DS', 'DS_FINAL', 'SHIFT_L', 'SHIFT_R']:
        phase['left_contact'] = True
        phase['right_contact'] = True
        phase['left_target'] = step['left'].copy()
        phase['right_target'] = step['right'].copy()
        
    elif step_type == 'SWING_R':
        phase['left_contact'] = True
        phase['right_contact'] = False
        phase['left_target'] = step['left'].copy()
        
        n_swing = int(T_step / dt)
        swing_traj = generate_swing_trajectory(
            step['right_start'], step['right_end'], step_height, max(n_swing, 2)
        )
        swing_idx = min(int(t_ratio * (len(swing_traj)-1)), len(swing_traj)-1)
        phase['right_target'] = swing_traj[swing_idx]
        
    elif step_type == 'SWING_L':
        phase['left_contact'] = False
        phase['right_contact'] = True
        phase['right_target'] = step['right'].copy()
        
        n_swing = int(T_step / dt)
        swing_traj = generate_swing_trajectory(
            step['left_start'], step['left_end'], step_height, max(n_swing, 2)
        )
        swing_idx = min(int(t_ratio * (len(swing_traj)-1)), len(swing_traj)-1)
        phase['left_target'] = swing_traj[swing_idx]
    
    phases.append(phase)

#======================================================================================
# Standing Model
#======================================================================================
left_foot_init = np.array([0.0, foot_width/2, 0.0])
right_foot_init = np.array([0.0, -foot_width/2, 0.0])

def createStandingModel():
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    left_pose = pin.SE3(np.eye(3), left_foot_init)
    right_pose = pin.SE3(np.eye(3), right_foot_init)

    contact_l = crocoddyl.ContactModel6D(state, L_FOOT_ID, left_pose, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0, 0]))
    contact_r = crocoddyl.ContactModel6D(state, R_FOOT_ID, right_pose, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0, 0]))
    contacts.addContact("left_foot", contact_l)
    contacts.addContact("right_foot", contact_r)

    foot_res_l = crocoddyl.ResidualModelFramePlacement(state, L_FOOT_ID, left_pose, actuation.nu)
    foot_res_r = crocoddyl.ResidualModelFramePlacement(state, R_FOOT_ID, right_pose, actuation.nu)
    costs.addCost("foot_l", crocoddyl.CostModelResidual(state, foot_res_l), 1e6)
    costs.addCost("foot_r", crocoddyl.CostModelResidual(state, foot_res_r), 1e6)

    x_target = np.concatenate([q_init, np.zeros(model.nv)])
    state_res = crocoddyl.ResidualModelState(state, x_target, actuation.nu)
    
    weights = np.ones(state.ndx)
    weights[0:3] = 1e3
    weights[3:6] = 1e5
    weights[6:model.nv] = 1.0
    weights[model.nv:] = 2.0
    
    activation = crocoddyl.ActivationModelWeightedQuad(weights)
    costs.addCost("state", crocoddyl.CostModelResidual(state, activation, state_res), 1.0)

    pin.centerOfMass(model, data, q_init)
    com_target = data.com[0].copy()
    com_target[0] = 0.0
    com_target[1] = 0.0
    com_res = crocoddyl.ResidualModelCoMPosition(state, com_target, actuation.nu)
    costs.addCost("com", crocoddyl.CostModelResidual(state, com_res), 1e4)

    ctrl_res = crocoddyl.ResidualModelControl(state, actuation.nu)
    costs.addCost("ctrl", crocoddyl.CostModelResidual(state, ctrl_res), 1e-2)

    d_model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contacts, costs, 0.0, True)
    return crocoddyl.IntegratedActionModelEuler(d_model, 0.01)

#======================================================================================
# Standing Problem Setup
#======================================================================================
#서있는 시간
T_stand = 50
x_init = np.concatenate([q_init, np.zeros(model.nv)])

problem_stand = crocoddyl.ShootingProblem(x_init, [createStandingModel()] * T_stand, createStandingModel())
solver_stand = crocoddyl.SolverFDDP(problem_stand)
solver_stand.setCallbacks([crocoddyl.CallbackVerbose()])
solver_stand.solve([x_init] * (T_stand + 1), [np.zeros(actuation.nu)] * T_stand, 200)

q0 = solver_stand.xs[-1][:model.nq]
x0 = solver_stand.xs[-1]

#======================================================================================
# Walking Model
#======================================================================================
def createWalkingModel(phase, phase_idx, total_phases, all_phases):
    left_contact = phase['left_contact']
    right_contact = phase['right_contact']
    left_target = phase['left_target']
    right_target = phase['right_target']
    com_x = phase['com_x_target']
    com_y = phase['com_y_target']
    
    # Desired CoM velocity from trajectory
    com_vx = phase.get('com_vx', 0.0)
    com_vy = phase.get('com_vy', 0.0)

    costs = crocoddyl.CostModelSum(state, actuation.nu)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    left_pose = pin.SE3(np.eye(3), left_target)
    right_pose = pin.SE3(np.eye(3), right_target)

    # ===== Detect transition phases for smoother contact handling =====
    is_transition = False
    transition_window = 10  # frames around contact change
    
    for offset in range(-transition_window, transition_window + 1):
        neighbor_idx = phase_idx + offset
        if 0 <= neighbor_idx < total_phases:
            neighbor = all_phases[neighbor_idx]
            if (neighbor['left_contact'] != left_contact or  neighbor['right_contact'] != right_contact):
                is_transition = True
                break

    if left_contact:
        contact_l = crocoddyl.ContactModel6D(state, L_FOOT_ID, left_pose,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0, 0]))
        contacts.addContact("left_foot", contact_l)
        
        foot_res_l = crocoddyl.ResidualModelFramePlacement(state, L_FOOT_ID, left_pose, actuation.nu)
        weight = 5e5 if is_transition else (1e6 if not right_contact else 5e5)
        costs.addCost("stance_l", crocoddyl.CostModelResidual(state, foot_res_l), weight)

    if right_contact:
        contact_r = crocoddyl.ContactModel6D(state, R_FOOT_ID, right_pose,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0, 0]))
        contacts.addContact("right_foot", contact_r)
        
        foot_res_r = crocoddyl.ResidualModelFramePlacement(state, R_FOOT_ID, right_pose, actuation.nu)
        weight = 5e5 if is_transition else (1e6 if not left_contact else 5e5)
        costs.addCost("stance_r", crocoddyl.CostModelResidual(state, foot_res_r), weight)

    
    if not left_contact:
        swing_res_l = crocoddyl.ResidualModelFrameTranslation(state, L_FOOT_ID, left_target, actuation.nu)
        costs.addCost("swing_pos_l", crocoddyl.CostModelResidual(state, swing_res_l), 1e5)
        swing_rot_l = crocoddyl.ResidualModelFrameRotation(state, L_FOOT_ID, np.eye(3), actuation.nu)
        costs.addCost("swing_rot_l", crocoddyl.CostModelResidual(state, swing_rot_l), 1e5)

    if not right_contact:
        swing_res_r = crocoddyl.ResidualModelFrameTranslation(state, R_FOOT_ID, right_target, actuation.nu)
        costs.addCost("swing_pos_r", crocoddyl.CostModelResidual(state, swing_res_r), 1e5)
        swing_rot_r = crocoddyl.ResidualModelFrameRotation(state, R_FOOT_ID, np.eye(3), actuation.nu)
        costs.addCost("swing_rot_r", crocoddyl.CostModelResidual(state, swing_rot_r), 1e5)

    q_target = q0.copy()
    q_target[0] = com_x
    q_target[1] = com_y
    q_target[2] = desired_base_height
    q_target[3:7] = [0, 0, 0, 1]

    v_target = np.zeros(model.nv)
    v_target[0] = com_vx  # Base X velocity
    v_target[1] = com_vy  # Base Y velocity

    x_target = np.concatenate([q_target, v_target])
    state_res = crocoddyl.ResidualModelState(state, x_target, actuation.nu)

    weights = np.ones(state.ndx)
    
    # Position
    weights[0] = 1e2      # X
    weights[1] = 5e2      # Y - lateral stability
    weights[2] = 5e4      # Z - height stability
    
    # Orientation
    weights[3:6] = 1e4
    
    # Joints
    weights[6:model.nv] = 0.5

    for joint_name, idx_v in hip_roll_yaw_indices:
        weights[idx_v] = 1e3
    
    # Velocities - track desired velocity
    weights[model.nv:model.nv+2] = 1e2    # Base XY velocity tracking
    weights[model.nv+2] = 5e2             # Base Z velocity - penalize vertical motion
    weights[model.nv+3:model.nv+6] = 1e2  # Angular velocities
    weights[model.nv+6:] = 0.5            # Joint velocities

    activation = crocoddyl.ActivationModelWeightedQuad(weights)
    costs.addCost("state_reg", crocoddyl.CostModelResidual(state, activation, state_res), 1.0)

    # CoM tracking
    pin.forwardKinematics(model, data, q_target)
    pin.centerOfMass(model, data, q_target)
    com_ref = data.com[0].copy()
    com_ref[0] = com_x
    com_ref[1] = com_y

    com_res = crocoddyl.ResidualModelCoMPosition(state, com_ref, actuation.nu)
    is_single = not (left_contact and right_contact)
    com_weight = 1e5 if is_single else 5e4
    costs.addCost("com", crocoddyl.CostModelResidual(state, com_res), com_weight)

    # Control regularization - increased for smoother torques
    ctrl_res = crocoddyl.ResidualModelControl(state, actuation.nu)
    ctrl_weight = 5e-3 if is_transition else 1e-3  # Higher during transitions
    costs.addCost("ctrl", crocoddyl.CostModelResidual(state, ctrl_res), ctrl_weight)

    d_model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contacts, costs, 0.0, True)
    
    return crocoddyl.IntegratedActionModelEuler(d_model, dt)

#======================================================================================
# Create Action Models
#======================================================================================
action_models = []
for i, phase in enumerate(phases):
    action_models.append(createWalkingModel(phase, i, len(phases), phases))

terminal_model = createWalkingModel(phases[-1], len(phases)-1, len(phases), phases)

#======================================================================================
# Optimization
#======================================================================================
problem = crocoddyl.ShootingProblem(x0, action_models, terminal_model)
solver = crocoddyl.SolverFDDP(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs_init = [x0] * (len(phases) + 1)
us_init = [np.zeros(actuation.nu)] * len(phases)

converged = solver.solve(xs_init, us_init, 300)

print(f"\nFinal cost: {solver.cost:.6f}")
print(f"Converged: {converged}")
#======================================================================================
# Analysis
#======================================================================================
print("\n=== Trajectory Analysis ===")

base_x = [x[0] for x in solver.xs]
base_y = [x[1] for x in solver.xs]
base_z = [x[2] for x in solver.xs]

print(f"Base X: {base_x[0]:.4f} to {base_x[-1]:.4f} (travel: {base_x[-1]-base_x[0]:.4f} m)")
print(f"Base Y: [{min(base_y):.4f}, {max(base_y):.4f}] (range: {max(base_y)-min(base_y):.4f} m)")
print(f"Base Z: [{min(base_z):.4f}, {max(base_z):.4f}]")

# CoM analysis
com_actual = []
for x in solver.xs:
    q = x[:model.nq]
    pin.centerOfMass(model, data, q)
    com_actual.append(data.com[0].copy())
com_actual = np.array(com_actual)

com_vel_actual = np.gradient(com_actual, dt, axis=0)
com_acc_actual = np.gradient(com_vel_actual, dt, axis=0)

trim = 10
com_acc_trimmed = com_acc_actual[trim:-trim] if len(com_acc_actual) > 2*trim else com_acc_actual

print(f"\n=== CoM Acceleration (OPTIMIZED) ===")
print(f"  Max |ax| = {np.max(np.abs(com_acc_trimmed[:,0])):.3f} m/s²")
print(f"  Max |ay| = {np.max(np.abs(com_acc_trimmed[:,1])):.3f} m/s²")
print(f"  Max |az| = {np.max(np.abs(com_acc_trimmed[:,2])):.3f} m/s²")

# Check if within limits (using trimmed data)
ax_max = np.max(np.abs(com_acc_trimmed[:,0]))
ay_max = np.max(np.abs(com_acc_trimmed[:,1]))
az_max = np.max(np.abs(com_acc_trimmed[:,2]))

ax_ok = ax_max <= MAX_COM_ACC_XY * 1.2 
ay_ok = ay_max <= MAX_COM_ACC_XY * 1.2
az_ok = az_max <= MAX_COM_ACC_Z * 1.5

print(f"\n  Acceleration within limits (with margin):")
print(f"    ax ({ax_max:.2f} vs {MAX_COM_ACC_XY*1.2:.1f}): {'✓' if ax_ok else '✗'}")
print(f"    ay ({ay_max:.2f} vs {MAX_COM_ACC_XY*1.2:.1f}): {'✓' if ay_ok else '✗'}")
print(f"    az ({az_max:.2f} vs {MAX_COM_ACC_Z*1.5:.1f}): {'✓' if az_ok else '✗'}")

if ax_ok and ay_ok and az_ok:
    print("\n Trajectory appears FEASIBLE for real robot")
else:
    print("\n Trajectory may need further tuning for real robot")

# Save
trajectory = {
    'q': [x[:model.nq] for x in solver.xs],
    'v': [x[model.nq:] for x in solver.xs],
    'tau': [u for u in solver.us],
    'dt': dt,
    'phases': phases,
    'com_acc': com_acc_actual
}

np.save('walking_trajectory_safe.npy', trajectory)
print("\nSaved: walking_trajectory_safe.npy")

#======================================================================================
# Plotting
#======================================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Top view
ax = axes[0, 0]
ax.set_title("Top View: CoM Trajectory")
ax.plot(com_actual[:, 1], com_actual[:, 0], 'b-', linewidth=2, label='Actual CoM')
ax.plot(com_traj[:, 1], com_traj[:, 0], 'g--', alpha=0.5, label='Planned CoM')
ax.scatter(com_actual[0, 1], com_actual[0, 0], c='green', s=100, marker='o', label='Start')
ax.scatter(com_actual[-1, 1], com_actual[-1, 0], c='red', s=100, marker='x', label='End')
ax.set_xlabel("Y (lateral) [m]")
ax.set_ylabel("X (forward) [m]")
ax.axis('equal')
ax.grid(True)
ax.legend()

# 2. CoM Y over time
ax = axes[0, 1]
ax.set_title("Lateral CoM Motion")
time_vec = np.arange(len(com_actual)) * dt
ax.plot(time_vec, com_actual[:, 1], 'b-', linewidth=2, label='Actual')
ax.plot(time_vec[:len(com_traj)], com_traj[:, 1], 'g--', alpha=0.7, label='Planned')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Y [m]")
ax.grid(True)
ax.legend()

# 3. Base height
ax = axes[0, 2]
ax.set_title("Base Height")
ax.plot(time_vec, base_z, 'b-', linewidth=2)
ax.axhline(y=desired_base_height, color='r', linestyle='--', label='Target')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Z [m]")
ax.grid(True)
ax.legend()

# 4. CoM Acceleration X
ax = axes[1, 0]
ax.set_title("CoM Acceleration X")
ax.plot(time_vec, com_acc_actual[:, 0], 'b-', linewidth=1, alpha=0.5, label='Raw')
# Highlight valid region
ax.plot(time_vec[trim:-trim], com_acc_actual[trim:-trim, 0], 'b-', linewidth=2, label='Valid region')
ax.axhline(y=MAX_COM_ACC_XY, color='r', linestyle='--', label=f'Limit (±{MAX_COM_ACC_XY})')
ax.axhline(y=-MAX_COM_ACC_XY, color='r', linestyle='--')
ax.fill_between(time_vec, -MAX_COM_ACC_XY, MAX_COM_ACC_XY, alpha=0.1, color='green')
ax.axvspan(0, time_vec[trim], alpha=0.2, color='gray')  # trim region
ax.axvspan(time_vec[-trim], time_vec[-1], alpha=0.2, color='gray')
ax.set_xlabel("Time [s]")
ax.set_ylabel("ax [m/s²]")
ax.grid(True)
ax.legend(fontsize=8)

# 5. CoM Acceleration Y
ax = axes[1, 1]
ax.set_title("CoM Acceleration Y (CRITICAL)")
ax.plot(time_vec, com_acc_actual[:, 1], 'b-', linewidth=1, alpha=0.5, label='Raw')
ax.plot(time_vec[trim:-trim], com_acc_actual[trim:-trim, 1], 'b-', linewidth=2, label='Valid region')
ax.axhline(y=MAX_COM_ACC_XY, color='r', linestyle='--', label=f'Limit (±{MAX_COM_ACC_XY})')
ax.axhline(y=-MAX_COM_ACC_XY, color='r', linestyle='--')
ax.fill_between(time_vec, -MAX_COM_ACC_XY, MAX_COM_ACC_XY, alpha=0.1, color='green')
ax.axvspan(0, time_vec[trim], alpha=0.2, color='gray')
ax.axvspan(time_vec[-trim], time_vec[-1], alpha=0.2, color='gray')
ax.set_xlabel("Time [s]")
ax.set_ylabel("ay [m/s²]")
ax.grid(True)
ax.legend(fontsize=8)

# 6. CoM Velocity
ax = axes[1, 2]
ax.set_title("CoM Velocity")
ax.plot(time_vec, com_vel_actual[:, 0], 'r-', label='Vx')
ax.plot(time_vec, com_vel_actual[:, 1], 'g-', label='Vy')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Velocity [m/s]")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('trajectory_analysis_safe.png', dpi=150)
print("Saved: trajectory_analysis_safe.png")

#======================================================================================
# Visualization
#======================================================================================
try:
    visual_model = pin.buildGeomFromUrdf(
        model, urdf_path, pin.GeometryType.VISUAL,
        package_dirs=[os.path.dirname(urdf_path)]
    )
    v = MeshcatVisualizer(model, visual_model, visual_model)
    v.initViewer(open=True)
    v.loadViewerModel("pinocchio")

    try:
        while True:
            for i, x in enumerate(solver.xs):
                v.display(x[:model.nq])
                time.sleep(dt * 0.5)
                
                if i % 100 == 0:
                    phase = phases[min(i, len(phases)-1)]
                    acc = com_acc_actual[min(i, len(com_acc_actual)-1)]
                    print(f"t={i*dt:.2f}s [{phase['step_type']:10s}] "
                          f"acc=[{acc[0]:+.2f}, {acc[1]:+.2f}] m/s²")
            
            print("--- Replay ---")
    except KeyboardInterrupt:
        print("\nStopped")

except Exception as e:
    print(f"Visualization error: {e}")