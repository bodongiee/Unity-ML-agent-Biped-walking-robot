#======================================================================================
# Fixed CoM 이족보행 궤적 생성 (Cubic Spline 미사용)
#======================================================================================

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time
import crocoddyl
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

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
# Trajectory Utilities
#======================================================================================
def generate_swing_trajectory(start, end, height, n_points):
    trajectory = []
    T = n_points - 1

    for i in range(n_points):
        t = i / max(T, 1)

        # XY: quintic polynomial for smooth start/end
        poly = 10*t**3 - 15*t**4 + 6*t**5
        x = start[0] + (end[0] - start[0]) * poly
        y = start[1] + (end[1] - start[1]) * poly
        # Z: sine curve for natural arc
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
T_ds = 0.8              # Double support

# Steps
num_steps = 3

pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
init_left = data.oMf[L_FOOT_ID].translation.copy()
init_right = data.oMf[R_FOOT_ID].translation.copy()

# Footstep sequence - 각 단계의 CoM 목표 위치 포함
footstep_sequence = []

curr_left = init_left.copy()
curr_right = init_right.copy()

center = (curr_left[:2] + curr_right[:2]) / 2

# Initial Double Support
footstep_sequence.append({
    'type': 'DS_INIT',
    'duration': T_init,
    'left': curr_left.copy(),
    'right': curr_right.copy(),
    'com_start': center.copy(),
    'com_end': center.copy()
})

support_foot = 'left'

# Step Sequence Generation
for step_i in range(num_steps):
    if support_foot == 'left':
        # Shift to left foot
        shift_target = curr_left[:2].copy() * 0.98 + center * 0.02
        footstep_sequence.append({
            'type': 'SHIFT_L',
            'duration': T_shift,
            'left': curr_left.copy(),
            'right': curr_right.copy(),
            'com_start': center.copy(),
            'com_end': shift_target.copy()
        })

        # Swing right foot - CoM 고정
        next_right = curr_right.copy()
        next_right[0] = curr_left[0] + step_length
        footstep_sequence.append({
            'type': 'SWING_R',
            'duration': T_swing,
            'left': curr_left.copy(),
            'right_start': curr_right.copy(),
            'right_end': next_right.copy(),
            'com_start': shift_target.copy(),
            'com_end': shift_target.copy()  # 고정
        })

        # Double support
        curr_right = next_right.copy()
        center = (curr_left[:2] + curr_right[:2]) / 2
        footstep_sequence.append({
            'type': 'DS',
            'duration': T_ds,
            'left': curr_left.copy(),
            'right': curr_right.copy(),
            'com_start': shift_target.copy(),
            'com_end': center.copy()
        })

        support_foot = 'right'

    else:
        # Shift to right foot
        shift_target = curr_right[:2].copy() * 0.98 + center * 0.02
        footstep_sequence.append({
            'type': 'SHIFT_R',
            'duration': T_shift,
            'left': curr_left.copy(),
            'right': curr_right.copy(),
            'com_start': center.copy(),
            'com_end': shift_target.copy()
        })

        # Swing left foot - CoM 고정
        next_left = curr_left.copy()
        next_left[0] = curr_right[0] + step_length
        footstep_sequence.append({
            'type': 'SWING_L',
            'duration': T_swing,
            'left_start': curr_left.copy(),
            'left_end': next_left.copy(),
            'right': curr_right.copy(),
            'com_start': shift_target.copy(),
            'com_end': shift_target.copy()  # 고정
        })

        # Double support
        curr_left = next_left.copy()
        center = (curr_left[:2] + curr_right[:2]) / 2
        footstep_sequence.append({
            'type': 'DS',
            'duration': T_ds,
            'left': curr_left.copy(),
            'right': curr_right.copy(),
            'com_start': shift_target.copy(),
            'com_end': center.copy()
        })

        support_foot = 'left'

# 마지막 오른발 자기 위치
shift_target = curr_left[:2].copy() * 0.98 + center * 0.02
footstep_sequence.append({
    'type': 'SHIFT_L',
    'duration': T_shift,
    'left': curr_left.copy(),
    'right': curr_right.copy(),
    'com_start': center.copy(),
    'com_end': shift_target.copy()
})

next_right = curr_right.copy()
next_right[0] = curr_left[0]
footstep_sequence.append({
    'type': 'SWING_R',
    'duration': T_swing,
    'left': curr_left.copy(),
    'right_start': curr_right.copy(),
    'right_end': next_right.copy(),
    'com_start': shift_target.copy(),
    'com_end': shift_target.copy()
})

curr_right = next_right.copy()
center = (curr_left[:2] + curr_right[:2]) / 2
footstep_sequence.append({
    'type': 'DS',
    'duration': T_ds,
    'left': curr_left.copy(),
    'right': curr_right.copy(),
    'com_start': shift_target.copy(),
    'com_end': center.copy()
})

print(f"  Footstep sequence: {len(footstep_sequence)}")

#======================================================================================
# Phase Sequence - Cubic Spline 없이 직접 생성
#======================================================================================
phases = []
current_time = 0.0

for step in footstep_sequence:
    step_type = step['type']
    duration = step['duration']
    com_start = step['com_start']
    com_end = step['com_end']

    n_frames = int(duration / dt)

    for i in range(n_frames):
        t = i * dt
        t_ratio = i / max(n_frames - 1, 1)

        # 선형 보간으로 CoM 계산
        com_current = com_start * (1 - t_ratio) + com_end * t_ratio

        # CoM 속도 계산 (일정 속도)
        com_vel = (com_end - com_start) / max(duration, dt)

        phase = {
            'time': current_time + t,
            'step_type': step_type,
            'com_x_target': com_current[0],
            'com_y_target': com_current[1],
            'com_vx': com_vel[0],
            'com_vy': com_vel[1]
        }

        if step_type in ['DS_INIT', 'DS', 'SHIFT_L', 'SHIFT_R']:
            phase['left_contact'] = True
            phase['right_contact'] = True
            phase['left_target'] = step['left'].copy()
            phase['right_target'] = step['right'].copy()

        elif step_type == 'SWING_R':
            phase['left_contact'] = True
            phase['right_contact'] = False
            phase['left_target'] = step['left'].copy()

            swing_traj = generate_swing_trajectory(
                step['right_start'], step['right_end'], step_height, n_frames
            )
            swing_idx = min(i, len(swing_traj) - 1)
            phase['right_target'] = swing_traj[swing_idx]

        elif step_type == 'SWING_L':
            phase['left_contact'] = False
            phase['right_contact'] = True
            phase['right_target'] = step['right'].copy()

            swing_traj = generate_swing_trajectory(
                step['left_start'], step['left_end'], step_height, n_frames
            )
            swing_idx = min(i, len(swing_traj) - 1)
            phase['left_target'] = swing_traj[swing_idx]

        phases.append(phase)

    current_time += duration

n_phases = len(phases)
total_time = current_time

print(f"  Total phases: {n_phases}")
print(f"  Total time: {total_time:.2f}s")

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

    com_vx = phase.get('com_vx', 0.0)
    com_vy = phase.get('com_vy', 0.0)

    costs = crocoddyl.CostModelSum(state, actuation.nu)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    left_pose = pin.SE3(np.eye(3), left_target)
    right_pose = pin.SE3(np.eye(3), right_target)

    is_transition = False
    transition_window = 10

    for offset in range(-transition_window, transition_window + 1):
        neighbor_idx = phase_idx + offset
        if 0 <= neighbor_idx < total_phases:
            neighbor = all_phases[neighbor_idx]
            if (neighbor['left_contact'] != left_contact or neighbor['right_contact'] != right_contact):
                is_transition = True
                break

    if left_contact:
        contact_l = crocoddyl.ContactModel6D(state, L_FOOT_ID, left_pose, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0, 0]))
        contacts.addContact("left_foot", contact_l)

        foot_res_l = crocoddyl.ResidualModelFramePlacement(state, L_FOOT_ID, left_pose, actuation.nu)
        weight = 5e5 if is_transition else (1e6 if not right_contact else 5e5)
        costs.addCost("stance_l", crocoddyl.CostModelResidual(state, foot_res_l), weight)

    if right_contact:
        contact_r = crocoddyl.ContactModel6D(state, R_FOOT_ID, right_pose, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0, 0]))
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
    v_target[0] = com_vx
    v_target[1] = com_vy

    x_target = np.concatenate([q_target, v_target])
    state_res = crocoddyl.ResidualModelState(state, x_target, actuation.nu)

    weights = np.ones(state.ndx)

    weights[0] = 1e2
    weights[1] = 5e2
    weights[2] = 5e4

    weights[3:6] = 1e4

    weights[6:model.nv] = 0.5

    for joint_name, idx_v in hip_roll_yaw_indices:
        weights[idx_v] = 1e3

    weights[model.nv:model.nv+2] = 1e2
    weights[model.nv+2] = 5e2
    weights[model.nv+3:model.nv+6] = 1e2
    weights[model.nv+6:] = 0.5

    activation = crocoddyl.ActivationModelWeightedQuad(weights)
    costs.addCost("state_reg", crocoddyl.CostModelResidual(state, activation, state_res), 1.0)

    pin.forwardKinematics(model, data, q_target)
    pin.centerOfMass(model, data, q_target)
    com_ref = data.com[0].copy()
    com_ref[0] = com_x
    com_ref[1] = com_y

    com_res = crocoddyl.ResidualModelCoMPosition(state, com_ref, actuation.nu)
    is_single = not (left_contact and right_contact)
    com_weight = 1e5 if is_single else 5e4
    costs.addCost("com", crocoddyl.CostModelResidual(state, com_res), com_weight)

    ctrl_res = crocoddyl.ResidualModelControl(state, actuation.nu)
    ctrl_weight = 5e-3 if is_transition else 1e-3
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

# Save
trajectory = {
    'q': [x[:model.nq] for x in solver.xs],
    'v': [x[model.nq:] for x in solver.xs],
    'tau': [u for u in solver.us],
    'dt': dt,
    'phases': phases,
    'com_acc': com_acc_actual
}

np.save('walking_trajectory_fix_com.npy', trajectory)
print("\nSaved: walking_trajectory_fix_com.npy")

#======================================================================================
# Plotting
#======================================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Top view
ax = axes[0, 0]
ax.set_title("Top View: CoM Trajectory")
ax.plot(com_actual[:, 1], com_actual[:, 0], 'b-', linewidth=2, label='Actual CoM')
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
ax.plot(time_vec[trim:-trim], com_acc_actual[trim:-trim, 0], 'b-', linewidth=2, label='Valid region')
ax.set_xlabel("Time [s]")
ax.set_ylabel("ax [m/s²]")
ax.grid(True)
ax.legend(fontsize=8)

# 5. CoM Acceleration Y
ax = axes[1, 1]
ax.set_title("CoM Acceleration Y")
ax.plot(time_vec, com_acc_actual[:, 1], 'b-', linewidth=1, alpha=0.5, label='Raw')
ax.plot(time_vec[trim:-trim], com_acc_actual[trim:-trim, 1], 'b-', linewidth=2, label='Valid region')
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
plt.savefig('trajectory_analysis_fix_com.png', dpi=150)
print("Saved: trajectory_analysis_fix_com.png")

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
