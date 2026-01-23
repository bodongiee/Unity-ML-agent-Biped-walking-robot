using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

/// BipedalAgent_v2: Implements "Learning Point-to-Point Bipedal Walking Without Global Navigation"
/// Features: Dual Clocks (Movement + Gait), Trajectory Tracking, Paper-based Rewards

public class BipedalAgent_v2 : Agent {
    
    // ===== JOINT REFERENCES (6 DOF) =====
    [Header("Joint References")]
    public ArticulationBody leftHipJoint;
    public ArticulationBody leftKneeJoint;
    public ArticulationBody leftAnkleJoint;
    public ArticulationBody rightHipJoint;
    public ArticulationBody rightKneeJoint;
    public ArticulationBody rightAnkleJoint;
    private ArticulationBody[] joints;

    [Header("Robot Base")]
    public ArticulationBody baseLink;

    [Header("Foot Contacts")]
    public FootContact leftFoot;
    public FootContact rightFoot;
    public Transform leftFootTransform;
    public Transform rightFootTransform;
    public ArticulationBody leftFootBody;   // For velocity tracking
    public ArticulationBody rightFootBody;  // For velocity tracking

    [Header("Target")]
    public Transform target;

    // ===== REWARD SCALES (Walking Quality Focus) =====
    [Header("Reward Scales")]
    public float scale_forward_velocity = 15f;    // 저속 전진 (Z축 직진)
    public float scale_knee_bend = 5f;            // 무릎 굽힘 (새로 추가)
    public float scale_target_velocity = 5f;      // 타겟 방향 속도 추적
    public float scale_facing_target = 2f;        // 타겟 방향 바라보기
    public float scale_pos_stop = 5f;             // 정지 보상 (논문)
    public float scale_feet_contact_number = 3f;
    public float scale_feet_swingZ = 10f;
    public float scale_feet_slip = -0.1f;
    public float scale_orientation = 3f;
    public float scale_base_height = 4f;
    public float scale_action_smoothness = -0.2f;
    public float scale_dof_vel = -5e-4f;
    public float scale_survival = 0.01f;
    public float scale_feet_dswingY = 3f;
    public float scale_feet_dswingZ = 1.5f;
    public float scale_default_joint_pos = 0f;//4f;
    public float scale_base_acc = 0.2f;
    public float scale_virtual_leg_sym = 55f;
    public float scale_virtual_leg_sym_cont = 1.2f;
    public float scale_flat_orientation_l2 = -6f;
    public float scale_lateral_penalty = -2f;   // 측면 이동 페널티 (월드 X축)
    public float scale_yaw_penalty = -0.05f;    // Yaw 회전 페널티 (초기 방향 유지)

    public float scale_collision = -10f;

    // ===== TASK PARAMETERS =====
    private Vector3 startPos;
    private Quaternion startRot;
    private Vector3 targetPos;
    private float targetHeading; // Relative heading change
    private float movementDuration = 5f; // T
    private float movementTimer = 0f;

    // ===== CLOCKS =====
    private float m_MovementPhase = 0f; // phi: Task progress 0→1
    private float m_GaitPhase = 0f;     // phi_gait: Cyclic stepping
    private float m_GaitPeriod = 1.5f;  // Paper: 0.9s per cycle

    // ===== COMMANDS (Paper format) =====
    // [dx, dz, dtheta, T_duration]
    private float[] commands = new float[4];
    private float[] commandsScale = new float[] { 2f, 2f, 1f, 0.5f }; // Observation scaling

    // ===== STATE TRACKING =====
    private float[] lastActions;
    private float[] lastLastActions;
    private float[] defaultJointAngles;
    private Vector3 lastBaseVelocity;
    private float previousDistanceToTarget;
    private float  dfeY_swing_max = 1.0f;
    private float stoppingTimer = 0f;  // 타겟 근처 정지 시간 추적

    // ===== DESIRED STATE (for velocity tracking) =====
    private Vector3 basePosDes;
    private Vector3 baseLinVelDes;
    private float baseThetaYDes;
    private float baseAngVelYDes;

    // ===== CONFIGURATION =====
    private float initialHipAngle = -30f;  // Changed from -30 to allow more forward swing
    private float initialKneeAngle = 60f;  // Changed from 60 to match new hip angle
    private float initialAnkleAngle = -30f; // Changed from -30 to match
    private float targetBaseHeight = 0.65f;
    private float targetFeetSwingHeight = 0.2f; // Paper: 0.1m swing height
    private float targetWalkingSpeed = 0.3f;    // 목표 보행 속도 (m/s)
    //UnityEditor.TransformWorldPlacementJSON:{"position":{"x":-0.15000009536743165,"y":0.06700000166893006,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"scale":{"x":1.0,"y":1.0,"z":1.0}}
    //UnityEditor.TransformWorldPlacementJSON:{"position":{"x":0.0,"y":0.7670000195503235,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"scale":{"x":1.0,"y":1.0,"z":1.0}}
    // ===== OBSERVATION SCALING =====
    private float obs_scale_dof_pos = 1f;
    private float obs_scale_dof_vel = 0.05f;
    private float obs_scale_ang_vel = 0.2f;
    private float obs_scale_quat = 1f;
    private float obs_scale_feet_pos = 1f;

    // ===== VIRTUAL LEG TRACKING =====
    private bool lastLeftFootGrounded;
    private bool lastRightFootGrounded;
    private Vector3 virtualLegL_LF;   // 왼발 Lift-off 시점의 virtual leg
    private Vector3 virtualLegL_TD;   // 왼발 Touch-down 시점의 virtual leg
    private Vector3 virtualLegL_Mid;  // 왼발 mid-swing 시점의 virtual leg
    private Vector3 virtualLegR_LF;   // 오른발 Lift-off
    private Vector3 virtualLegR_TD;   // 오른발 Touch-down  
    private Vector3 virtualLegR_Mid;  // 오른발 mid-swing
    private bool eventLF_L, eventLF_R;  // Lift-off 이벤트 플래그
    private bool eventTD_L, eventTD_R;  // Touch-down 이벤트 플래그

    public override void Initialize() {
        joints = new ArticulationBody[] { 
            leftHipJoint, leftKneeJoint, leftAnkleJoint, 
            rightHipJoint, rightKneeJoint, rightAnkleJoint 
        };

        defaultJointAngles = new float[6];
        lastActions = new float[6];
        lastLastActions = new float[6];

        for (int i = 0; i < 6; i++) {
            if (i == 0 || i == 3) defaultJointAngles[i] = initialHipAngle;
            else if (i == 1 || i == 4) defaultJointAngles[i] = initialKneeAngle;
            else if (i == 2 || i == 5) defaultJointAngles[i] = initialAnkleAngle;

            if (joints[i] == null) continue;
            var xDrive = joints[i].xDrive;
            xDrive.stiffness = 10000f;
            xDrive.damping = 100f;
            xDrive.forceLimit = 1000f;
            joints[i].xDrive = xDrive;
        }

        //if (baseLink != null) {
            //targetBaseHeight = baseLink.transform.position.y;
            
        //}
    }

    public override void OnEpisodeBegin() {
        // Reset Clocks
        m_MovementPhase = 0f;
        m_GaitPhase = 0f;
        movementTimer = 0f;
        stoppingTimer = 0f;

        // Reset Actions History (6)
        for (int i = 0; i < 6; i++) {
            lastActions[i] = 0f;
            lastLastActions[i] = 0f;
        }
        lastBaseVelocity = Vector3.zero;
        // Reset Pose
        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;
            joints[i].jointPosition = new ArticulationReducedSpace(defaultJointAngles[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);
            var dr = joints[i].xDrive;
            dr.target = defaultJointAngles[i];
            joints[i].xDrive = dr;
        }

        // Initialize startPos relative to the agent's current location (Training Area support)
        // Adjust for target height
        startPos = transform.position;
        startPos.y = targetBaseHeight;
        startRot = Quaternion.identity;
        baseLink.TeleportRoot(startPos, startRot);
        baseLink.linearVelocity = Vector3.zero;
        baseLink.angularVelocity = Vector3.zero;
        lastBaseVelocity = Vector3.zero;

        // Generate Random Task
        // angle is relative to START orientation (local frame)
        // angle = 0 means FORWARD (+Z in Unity)
        // 정면만: 각도 0 고정 (Hip Yaw 없이 회전 불가)
        float localAngle = 0f; // 정면 고정
        float dist = Random.Range(3f, 8f);

        // Convert local displacement to world displacement using startRot
        // Unity coordinate: +Z = forward, +X = right
        // So: X = Sin(angle), Z = Cos(angle) for angle=0 to be forward
        Vector3 localDisplacement = new Vector3(Mathf.Sin(localAngle) * dist, 0, Mathf.Cos(localAngle) * dist);
        Vector3 worldDisplacement = startRot * localDisplacement;
        targetPos = startPos + worldDisplacement;

        targetHeading = 0f; // 회전 없음 (Hip Yaw 없이 회전 불가)
        movementDuration = dist / targetWalkingSpeed;

        if (target != null) target.position = targetPos;

        // Setup Command Vector - stored in START-LOCAL frame (consistent across Training Areas)
        commands[0] = localDisplacement.x; // dx in start-local frame
        commands[1] = localDisplacement.z; // dz in start-local frame
        commands[2] = targetHeading;       // dtheta (relative to start)
        commands[3] = movementDuration;    // T

        previousDistanceToTarget = Vector3.Distance(startPos, targetPos);
        
        velErrorHistory.Clear();

        //Debug.Log($"Start: {startPos}, Target: {targetPos}, Commands: ({commands[0]:F2}, {commands[1]:F2})");
    }

    public override void CollectObservations(VectorSensor sensor) {

        // ===== 1. CLOCK SIGNALS (4 dims) =====
        float phase_mov = m_MovementPhase * 2 * Mathf.PI;
        float phase_gait = m_GaitPhase * 2 * Mathf.PI;
        sensor.AddObservation(Mathf.Sin(phase_mov));
        sensor.AddObservation(Mathf.Cos(phase_mov));
        sensor.AddObservation(Mathf.Sin(phase_gait));
        sensor.AddObservation(Mathf.Cos(phase_gait));

        // ===== 2. TARGET INFO (Body Frame) (3 dims) =====
        // 타겟까지의 방향과 거리를 로컬 프레임으로 관측
        Vector3 toTarget = targetPos - baseLink.transform.position;
        Vector3 localTarget = baseLink.transform.InverseTransformDirection(toTarget);
        Vector3 localDir = localTarget.normalized;
        float distToTarget = toTarget.magnitude;
        sensor.AddObservation(localDir.x);  // 로컬 X 방향 (좌우)
        sensor.AddObservation(localDir.z);  // 로컬 Z 방향 (전후)
        sensor.AddObservation(Mathf.Clamp(distToTarget, 0f, 8f) / 8f); // 정규화된 거리

        // ===== 3. PROJECTED GRAVITY (2 dims) =====
        // Gravity vector projected into body frame (indicates tilt)
        Vector3 gravityWorld = Vector3.down;
        Vector3 projectedGravity = baseLink.transform.InverseTransformDirection(gravityWorld);
        sensor.AddObservation(projectedGravity.x); // tilt in X
        sensor.AddObservation(projectedGravity.z); // tilt in Z

        // ===== 4. JOINT POSITIONS (6 dims) =====
        foreach (var j in joints) {
            float pos = (j != null) ? j.jointPosition[0] : 0f;
            sensor.AddObservation(pos * obs_scale_dof_pos);
        }

        // ===== 5. JOINT VELOCITIES (6 dims) =====
        foreach (var j in joints) {
            float vel = (j != null) ? j.jointVelocity[0] : 0f;
            sensor.AddObservation(vel * obs_scale_dof_vel);
        }

        // ===== 6. LAST ACTIONS (6 dims) =====
        foreach (float a in lastActions) {
            sensor.AddObservation(a);
        }

        // ===== 7. BASE ANGULAR VELOCITY in Body Frame (3 dims) =====
        Vector3 angVelWorld = baseLink.angularVelocity;
        Vector3 angVelBody = baseLink.transform.InverseTransformDirection(angVelWorld);
        sensor.AddObservation(angVelBody * obs_scale_ang_vel);

        // ===== 8. BASE LINEAR VELOCITY in Body Frame (3 dims) - Paper Style =====
        Vector3 linVelWorld = baseLink.linearVelocity;
        Vector3 linVelBody = baseLink.transform.InverseTransformDirection(linVelWorld);
        sensor.AddObservation(linVelBody * 0.5f); // scaled

        // ===== 9. ANKLE POSITION (Body Frame) (6 dims) =====
        if (leftAnkleJoint != null && rightAnkleJoint != null) {
            Vector3 lPosB = baseLink.transform.InverseTransformPoint(leftAnkleJoint.transform.position) * obs_scale_feet_pos;
            Vector3 rPosB = baseLink.transform.InverseTransformPoint(rightAnkleJoint.transform.position) * obs_scale_feet_pos;
            sensor.AddObservation(lPosB);
            sensor.AddObservation(rPosB);
        } else {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(Vector3.zero);
        }


        // TOTAL: 4 + 3 + 2 + 6 + 6 + 6 + 3 + 3 + 6 = 39 dims
    }

    private float NormalizeAngle(float angle) {
        if (angle > 180f) angle -= 360f;
        return angle;
    }

    [Header("Control Parameters")]
    public float actionScale = 40f; // Scale for joint target adjustment

    public override void OnActionReceived(ActionBuffers actions) {
        float dt = Time.fixedDeltaTime;

        // ===== UPDATE CLOCKS =====
        movementTimer += dt;
        m_MovementPhase = Mathf.Clamp01(movementTimer / movementDuration);
        // 논문 방식: 시간 기반 phase만 사용 (거리 기반 조기 정지 제거)

        m_GaitPhase += dt / m_GaitPeriod;
        m_GaitPhase %= 1f;

        // commands는 OnEpisodeBegin에서 설정된 초기값 유지 (논문 방식)

        // ===== UPDATE DESIRED STATE (for velocity tracking) =====
        UpdateDesiredState();
        // ===== UPDATE VIRTUAL LEG EVENTS =====
        UpdateVirtualLegEvents();
        // ===== APPLY ACTIONS =====
        var continuousActions = actions.ContinuousActions;

        // Store actions for smoothness reward
        float[] currentActions = new float[6];
        for (int i = 0; i < 6; i++) {
            currentActions[i] = continuousActions[i];
        }

        // Apply to joints (Position Control)
        // action = target_angle (relative to default)
        for (int i = 0; i < 6; i++) {
            if (joints[i] == null) continue;
            
            float targetAngle = defaultJointAngles[i] + currentActions[i] * actionScale;
            
            // Set Drive Target
            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }

        // ===== CALCULATE REWARDS =====
        CalculateRewards(currentActions);

        // ===== UPDATE HISTORY =====
        for (int i = 0; i < 6; i++) {
            lastLastActions[i] = lastActions[i];
            lastActions[i] = currentActions[i];
        }
        lastBaseVelocity = baseLink.linearVelocity;
    }

    private void UpdateDesiredState() {
        // Linear interpolation of desired position based on MovementPhase
        basePosDes = Vector3.Lerp(startPos, targetPos, m_MovementPhase);

        // baseThetaYDes is the WORLD yaw we want at current phase
        // startRot's yaw + relative target heading * phase
        float startYawWorld = startRot.eulerAngles.y * Mathf.Deg2Rad;
        baseThetaYDes = startYawWorld + Mathf.Lerp(0, targetHeading, m_MovementPhase);

        // Desired velocity = total displacement / total duration
        if (movementDuration > 0) {
            baseLinVelDes = (targetPos - startPos) / movementDuration;
            baseAngVelYDes = targetHeading / movementDuration;
        }

        // Phase가 1이면 속도를 0으로 (정지 상태)
        if (m_MovementPhase >= 1f) {
            baseLinVelDes = Vector3.zero;
            baseAngVelYDes = 0f;
        }
    }


    private void CalculateRewards(float[] currentActions) {
        float totalReward = 0f;

        // ===== 1. SURVIVAL CHECK =====
        float upright = Vector3.Dot(baseLink.transform.up, Vector3.up);
        if (upright < 0.5f) {
            AddReward(-5f);
            EndEpisode();
            return;
        }
        totalReward += scale_survival;

        // ===== 2. 걷기 품질 보상 (Walking Quality) =====
        // 2a. 월드 Z축 직진
        totalReward += RewardForwardVelocity() * scale_forward_velocity;
        totalReward += RewardLateralPenalty() * scale_lateral_penalty;
        totalReward += RewardYawPenalty() * scale_yaw_penalty;

        // 2b. 걷기 패턴
        totalReward += RewardFeetContactNumber() * scale_feet_contact_number;
        totalReward += RewardFeetSwingZ() * scale_feet_swingZ;
        totalReward += RewardFeetDSwingY() * scale_feet_dswingY;
        totalReward += RewardFeetDSwingZ() * scale_feet_dswingZ;
        totalReward += RewardKneeBend() * scale_knee_bend;  // 무릎 굽힘
        totalReward += RewardVirtualLegSym() * scale_virtual_leg_sym;
        totalReward += RewardVirtualLegSymCont() * scale_virtual_leg_sym_cont;

        // ===== 3. 안정성 보상 (Stability) =====
        totalReward += RewardOrientation() * scale_orientation;
        totalReward += RewardBaseHeight() * scale_base_height;
        totalReward += RewardActionSmoothness(currentActions) * scale_action_smoothness;
        totalReward += RewardDofVel() * scale_dof_vel;
        totalReward += RewardFeetSlip() * scale_feet_slip;
        totalReward += RewardBaseAcc() * scale_base_acc;
        totalReward += RewardFlatOrientationL2() * scale_flat_orientation_l2;

        AddReward(totalReward);

        // ===== TIMEOUT CHECK =====
        if (movementTimer > 10f) {  // 10초 동안 걷기
            EndEpisode();
        }

        /* ===== 타겟 추적 모드 (주석 처리) =====
        // ===== 2. PHASE-BASED REWARDS (논문 방식) =====
        if (m_MovementPhase < 1f) {
            // === 이동 중 (φ < 1): 타겟 방향 등속 이동 ===

            // 2a. 타겟 방향 속도 보상
            totalReward += RewardTargetVelocity() * scale_target_velocity;

            // 2b. 타겟 방향 바라보기 보상
            totalReward += RewardFacingTarget() * scale_facing_target;

            // === 걷기 관련 보상 ===
            totalReward += RewardFeetContactNumber() * scale_feet_contact_number;
            totalReward += RewardFeetSwingZ() * scale_feet_swingZ;
            totalReward += RewardFeetDSwingY() * scale_feet_dswingY;
            totalReward += RewardFeetDSwingZ() * scale_feet_dswingZ;
            totalReward += RewardVirtualLegSym() * scale_virtual_leg_sym;
            totalReward += RewardVirtualLegSymCont() * scale_virtual_leg_sym_cont;
        } else {
            // === 도착 시간 (φ >= 1): 정지 보상 ===
            totalReward += RewardPosStop() * scale_pos_stop;

            // 성공 종료: 속도가 낮으면 타이머 누적
            float speed = baseLink.linearVelocity.magnitude;
            if (speed < 0.1f) {
                stoppingTimer += Time.fixedDeltaTime;
                if (stoppingTimer >= 1.0f) {
                    AddReward(10f);  // 성공 보너스
                    EndEpisode();
                    return;
                }
            } else {
                stoppingTimer = 0f;
            }
        }

        // ===== 3. 공통 보상 (이동/정지 모두) =====
        totalReward += RewardOrientation() * scale_orientation;
        totalReward += RewardBaseHeight() * scale_base_height;
        totalReward += RewardActionSmoothness(currentActions) * scale_action_smoothness;
        totalReward += RewardDofVel() * scale_dof_vel;
        totalReward += RewardFeetSlip() * scale_feet_slip;
        totalReward += RewardDefaultJointPos() * scale_default_joint_pos;
        totalReward += RewardBaseAcc() * scale_base_acc;
        totalReward += RewardFlatOrientationL2() * scale_flat_orientation_l2;

        AddReward(totalReward);

        // ===== TIMEOUT CHECK =====
        if (movementTimer > movementDuration + 5f) {
            EndEpisode();
        }
        */
    }

    // 타겟 방향으로 일정 속도 유지 보상
    private float RewardTargetVelocity() {
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        Vector3 targetDir = toTarget.normalized;

        Vector3 currentVel = baseLink.linearVelocity;
        currentVel.y = 0;

        // 타겟 방향 속도 성분
        float velTowardTarget = Vector3.Dot(currentVel, targetDir);

        // 타겟 수직 방향 속도 성분
        Vector3 velLateralTarget = currentVel - velTowardTarget * targetDir;
        float lateralSpeed = velLateralTarget.magnitude;

        // 목표 속도 (0.3 m/s)
        float speedError = Mathf.Abs(velTowardTarget - targetWalkingSpeed);

        return Mathf.Exp((-speedError * 5f) - lateralSpeed * 2f );
    }

    // 타겟 방향 바라보기 보상
    private float RewardFacingTarget() {
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        Vector3 forward = baseLink.transform.forward;
        forward.y = 0;

        float dot = Vector3.Dot(forward.normalized, toTarget.normalized);
        return Mathf.Max(0, dot);
    }

    // 정지 보상 (타겟 도달 후)
    private float RewardStopping() {
        Vector3 vel = baseLink.linearVelocity;
        vel.y = 0;
        float speed = vel.magnitude;

        // 속도가 작을수록 높은 보상
        return Mathf.Exp(-speed * 10f);
    }

    // ==================== REWARD FUNCTIONS ====================
    private float RewardDefaultJointPos(){
        // Stance 다리만 기본 자세 유지 (Swing 다리는 자유롭게)
        bool leftIsStance = (m_GaitPhase >= 0.5f);
        bool rightIsStance = (m_GaitPhase < 0.5f);

        float totalDiff = 0f;
        if (leftIsStance) {
            totalDiff += Mathf.Abs(leftHipJoint.jointPosition[0] - defaultJointAngles[0] * Mathf.Deg2Rad);
        }
        if (rightIsStance) {
            totalDiff += Mathf.Abs(rightHipJoint.jointPosition[0] - defaultJointAngles[3] * Mathf.Deg2Rad);
        }

        totalDiff = Mathf.Clamp(totalDiff - 0.1f, 0f, 50f);
        return Mathf.Exp(-totalDiff * 10f);
    }

    private float RewardPosTracePXZ() {
        Vector3 currentPos = baseLink.transform.position;
        
        // Error to interpolated position (current desired)
        Vector2 posError = new Vector2(basePosDes.x - currentPos.x, basePosDes.z - currentPos.z);
        float errorNorm = posError.magnitude;
        
        // Error to FINAL target position (for log term)
        Vector2 posErrorEnd = new Vector2(targetPos.x - currentPos.x, targetPos.z - currentPos.z);
        float errorEndNorm = posErrorEnd.magnitude;

        // Paper formula: exp(-error * sigma) - 0.5 * error - 1.3 * log(clamp(end_error * 2, 0.001, 1.0))
        float errLogUse = Mathf.Clamp(errorEndNorm * 2f, 1e-3f, 1f);
        float rew = Mathf.Exp(errorNorm * -10f) - 0.5f * errorNorm - 1.3f * Mathf.Log(errLogUse + 1e-5f);
        
        return rew;
    }

    // History Buffer for Velocity Error (approx 0.5s window)
    private Queue<float> velErrorHistory = new Queue<float>();
    private int historyWindowSize = 50; // 0.01s dt * 50 = 0.5s

    private float RewardPosTracePXZ_Vel() {
        Vector3 currentVel = baseLink.linearVelocity;
        Vector2 velError = new Vector2(baseLinVelDes.x - currentVel.x, baseLinVelDes.z - currentVel.z);
        float currentErrorNorm = velError.magnitude;

        // Add to history
        velErrorHistory.Enqueue(currentErrorNorm);
        if (velErrorHistory.Count > historyWindowSize) {
            velErrorHistory.Dequeue();
        }

        // Calculate Average Error over horizon
        float avgError = 0f;
        foreach (float err in velErrorHistory) {
            avgError += err;
        }
        avgError /= velErrorHistory.Count;

        // Paper formula using Averaged Error
        float rew = Mathf.Exp(-avgError * 20f) - 0.7f * avgError;
        return rew;
    }

    private float RewardPosTraceThetaY() {
        float currentYaw = baseLink.transform.rotation.eulerAngles.y * Mathf.Deg2Rad;
        currentYaw = Mathf.Atan2(Mathf.Sin(currentYaw), Mathf.Cos(currentYaw)); // Normalize to -PI, PI

        // baseThetaYDes is now in WORLD frame (calculated in UpdateDesiredState)
        float desiredYaw = Mathf.Atan2(Mathf.Sin(baseThetaYDes), Mathf.Cos(baseThetaYDes)); // Normalize

        // Angular difference (handle wrap-around)
        float yawError = currentYaw - desiredYaw;
        yawError = Mathf.Atan2(Mathf.Sin(yawError), Mathf.Cos(yawError)); // Normalize to -PI, PI
        float yawErrorAbs = Mathf.Abs(yawError);

        // Error to FINAL target heading (also in WORLD frame)
        float startYawWorld = startRot.eulerAngles.y * Mathf.Deg2Rad;
        float finalYawWorld = startYawWorld + targetHeading;
        float finalError = currentYaw - finalYawWorld;
        finalError = Mathf.Atan2(Mathf.Sin(finalError), Mathf.Cos(finalError));
        float finalHeadingError = Mathf.Abs(finalError);

        // Paper formula: exp(-error * sigma) - 1.5 * error + 0.5 * exp(-end_error * 10)
        float rew = Mathf.Exp(-yawErrorAbs * 5f) - 1.5f * yawErrorAbs + 0.5f * Mathf.Exp(-finalHeadingError * 10f);
        return rew;
    }

    private float RewardPosTraceThetaY_Vel() {
        // Desired Angular Velocity (calculated in UpdateDesiredState)
        float targetAngVel = baseAngVelYDes;
        float currentAngVel = baseLink.angularVelocity.y;

        float angVelError = Mathf.Abs(targetAngVel - currentAngVel);
        
        // Paper formula: exp(-error * sigma) - 0.5 * error
        float rew = Mathf.Exp(-angVelError * 20f) - 0.5f * angVelError;
        return rew;
    }

    private float RewardPosStop() {
        // Only active when task is complete (phi >= 1) or in stand mode
        float linVel = baseLink.linearVelocity.magnitude;
        float angVel = baseLink.angularVelocity.magnitude;
        
        // Joint pose regularization (return to default pose)
        float jointDiffNorm = 0f;
        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;
            float diff = joints[i].jointPosition[0] * Mathf.Rad2Deg - defaultJointAngles[i];
            jointDiffNorm += Mathf.Abs(diff);
        }

        // Paper formula
        float rew_lin = Mathf.Exp(-linVel * 20f);
        float rew_ang = Mathf.Exp(-angVel * 20f);
        float rew_joint = Mathf.Exp(-jointDiffNorm * 0.01f);
        
        // Orientation bonus
        Vector3 euler = baseLink.transform.rotation.eulerAngles;
        float pitch = NormalizeAngle(euler.x) * Mathf.Deg2Rad;
        float roll = NormalizeAngle(euler.z) * Mathf.Deg2Rad;
        float orientation = Mathf.Exp(-Mathf.Sqrt(pitch*pitch + roll*roll) * 40f);

        return (rew_lin + rew_ang) * 0.5f + rew_joint + 1.5f * orientation;
    }

    private float RewardFeetContactNumber() {
        // Gait phase determines which foot should be on ground
        // phi_gait < 0.5: Right foot stance (Left swing)
        // phi_gait >= 0.5: Left foot stance (Right swing)
        bool leftShouldBeStance = (m_GaitPhase >= 0.5f);
        bool rightShouldBeStance = (m_GaitPhase < 0.5f);

        bool leftIsContact = (leftFoot != null && leftFoot.isGrounded);
        bool rightIsContact = (rightFoot != null && rightFoot.isGrounded);

        // Paper: reward = 2 if match, -0.3 if mismatch
        float rew = 0f;
        rew += (leftIsContact == leftShouldBeStance) ? 2f : -0.3f;
        rew += (rightIsContact == rightShouldBeStance) ? 2f : -0.3f;

        return rew * 0.5f; // Average
    }


    private float RewardFeetSwingZ() {
        // Reward swing foot for reaching target height (Paper: World Y coordinate)
        bool leftIsSwing = (m_GaitPhase < 0.5f);
        bool rightIsSwing = (m_GaitPhase >= 0.5f);

        float rew = 0f;

        if (leftIsSwing && leftAnkleJoint != null) {
            float footY = leftAnkleJoint.transform.position.y;
            float refY = GetSwingFootTargetHeight();
            float err = Mathf.Abs(footY - refY);
            rew += Mathf.Exp(-err * 80f) - 20f * err * err;

            // Debug: 스윙 높이 확인 (주기적으로)
            if (Time.frameCount % 100 == 0) {
                Debug.Log($"[SwingZ] Left ankleY={footY:F3}, refY={refY:F3}, err={err:F3}, phase={m_GaitPhase:F2}");
            }
        }

        if (rightIsSwing && rightAnkleJoint != null) {
            float footY = rightAnkleJoint.transform.position.y;
            float refY = GetSwingFootTargetHeight();
            float err = Mathf.Abs(footY - refY);
            rew += Mathf.Exp(-err * 80f) - 20f * err * err;
        }

        return rew;
    }
    private float RewardFeetDSwingY() {
        float rew = 0f;

        bool leftIsSwing = (m_GaitPhase < 0.5f);
        bool rightIsSwing = (m_GaitPhase >= 0.5f);

        float swingProgress = leftIsSwing ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);

        float refDY;
        if (swingProgress < 0.5f) {
            refDY = dfeY_swing_max; 
        }
        else {
            refDY = -dfeY_swing_max;
        }
        if (leftIsSwing && leftFootBody != null) {
            float footDY = leftFootBody.linearVelocity.y;
            float err = Mathf.Abs(refDY - footDY);
            rew += Mathf.Exp(-err * 5f) - 0.1f * err * err;
        }

        if (rightIsSwing && rightFootBody != null) {
            float footDY = rightFootBody.linearVelocity.y;
            float err = Mathf.Abs(refDY - footDY);
            rew += Mathf.Exp(-err * 5f) - 0.1f * err * err;
        }

        return rew;
    }
    
private float RewardFeetDSwingZ() {
    float rew = 0f;
    bool leftIsSwing = (m_GaitPhase < 0.5f);
    bool rightIsSwing = (m_GaitPhase >= 0.5f);

    if (leftIsSwing && leftFootBody != null) {
        Vector3 worldVel = leftFootBody.linearVelocity;
        Vector3 localVel = baseLink.transform.InverseTransformDirection(worldVel);
        float sq = localVel.z * localVel.z;
        rew += Mathf.Exp(-sq * 20f);
    }
    if (rightIsSwing && rightFootBody != null) {
        Vector3 worldVel = rightFootBody.linearVelocity;
        Vector3 localVel = baseLink.transform.InverseTransformDirection(worldVel);
        float sq = localVel.z * localVel.z;
        rew += Mathf.Exp(-sq * 20f);
    }
    return rew;
}

    // 무릎 굽힘 보상 - 스윙 다리의 무릎이 적절히 굽혀지도록
    private float RewardKneeBend() {
        float rew = 0f;
        bool leftIsSwing = (m_GaitPhase < 0.5f);
        bool rightIsSwing = (m_GaitPhase >= 0.5f);

        // 스윙 진행도에 따른 목표 무릎 굽힘 (스윙 중간에 최대 굽힘)
        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        float bendMultiplier = 4f * swingProgress * (1f - swingProgress); // 0→1→0 parabola
        float targetBendRad = 45f * Mathf.Deg2Rad * bendMultiplier; // 최대 45도 굽힘

        if (leftIsSwing && leftKneeJoint != null) {
            float kneeAngle = leftKneeJoint.jointPosition[0]; // 라디안
            float err = Mathf.Abs(kneeAngle - targetBendRad);
            rew += Mathf.Exp(-err * 5f);
        }
        if (rightIsSwing && rightKneeJoint != null) {
            float kneeAngle = rightKneeJoint.jointPosition[0]; // 라디안
            float err = Mathf.Abs(kneeAngle - targetBendRad);
            rew += Mathf.Exp(-err * 5f);
        }
        return rew;
    }

    // 전진 속도 보상 - 월드 Z축 방향으로 목표 속도로 이동하도록
    private float RewardForwardVelocity() {
        Vector3 vel = baseLink.linearVelocity;
        float forwardSpeed = vel.z;  // 월드 Z축 방향 속도
        float speedError = Mathf.Abs(forwardSpeed - targetWalkingSpeed);

        return Mathf.Exp(-speedError * 5f);
    }

    // 측면 이동 페널티 - 월드 X축 이탈 페널티
    private float RewardLateralPenalty() {
        Vector3 vel = baseLink.linearVelocity;
        float lateralSpeed = Mathf.Abs(vel.x);  // 월드 X축 방향 속도
        return -lateralSpeed;
    }

    // Yaw 회전 페널티 - 초기 방향 유지
    private float RewardYawPenalty() {
        float currentYaw = baseLink.transform.eulerAngles.y;
        float initialYaw = startRot.eulerAngles.y;
        float yawError = Mathf.Abs(Mathf.DeltaAngle(currentYaw, initialYaw));
        return -yawError;  // 도 단위, 스케일로 조절
    }

    private float GetSwingFootTargetHeight() {
        // Parabolic trajectory for swing foot (Paper: World Y coordinate)
        // Peak at mid-swing
        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        // 0.5보다 작으면 -> 왼발 스윙, 0.5보다 크면 -> 오른발 스윙
        float heightMultiplier = 4f * swingProgress * (1f - swingProgress); // Parabola 0→1→0
        // Paper: feZ_swing * swing_mask + ankle_height
        float ankleHeight = 0.06f; // 지면 기준 발목 높이
        return targetFeetSwingHeight * heightMultiplier + ankleHeight;
    }

    private float RewardOrientation() {
        Vector3 euler = baseLink.transform.rotation.eulerAngles;
        float pitch = NormalizeAngle(euler.x) * Mathf.Deg2Rad;
        float roll = NormalizeAngle(euler.z) * Mathf.Deg2Rad;

        // Paper uses two terms:
        // 1. quat_mismatch: exp(-|pitch| + |roll|) * 30)
        // 2. projected_gravity: exp(-sqrt(pitch^2 + roll^2) * 40)
        float quat_mismatch = Mathf.Exp(-(Mathf.Abs(pitch) + Mathf.Abs(roll)) * 30f);
        float proj_gravity = Mathf.Exp(-Mathf.Sqrt(pitch*pitch + roll*roll) * 40f);
        
        return (quat_mismatch + proj_gravity) * 0.5f;
    }

    private float RewardBaseHeight() {
        float currentHeight = baseLink.transform.position.y;
        float err = Mathf.Abs(currentHeight - targetBaseHeight);

        float rew = Mathf.Exp(-err * 20f) - 15f * err * err;
        return rew;
    }

    private float RewardActionSmoothness(float[] currentActions) {
        float term1 = 0f; // Difference from last action
        float term2 = 0f; // Second derivative

        for (int i = 0; i < 6; i++) {
            float diff1 = currentActions[i] - lastActions[i];
            term1 += diff1 * diff1;

            float diff2 = currentActions[i] + lastLastActions[i] - 2f * lastActions[i];
            term2 += diff2 * diff2;
        }

        return term1 + 0.1f * term2;
    }
    
    private float RewardBaseAcc() {
        Vector3 currentVel = baseLink.linearVelocity;
        Vector3 acc = (currentVel - lastBaseVelocity) / Time.fixedDeltaTime;
        float accNorm = acc.magnitude;
        float rew = Mathf.Exp(-accNorm * 6f);;
        return rew;
    }

    private float RewardDofVel() {
        float sumSq = 0f;
        foreach (var j in joints) {
            if (j == null) continue;
            float vel = j.jointVelocity[0];
            sumSq += vel * vel;
        }
        return sumSq;
    }

    private float RewardFeetSlip() {
        float slipReward = 0f;

        // Left foot slip
        if (leftFoot != null && leftFoot.isGrounded && leftFootBody != null) {
            Vector3 footVel = leftFootBody.linearVelocity;
            Vector3 footAngVel = leftFootBody.angularVelocity;
            // Paper: norm of [linear_vel(3) + angular_vel(3)]
            float speedNorm = Mathf.Sqrt(footVel.x * footVel.x + footVel.y * footVel.y + footVel.z * footVel.z
                                         + footAngVel.x * footAngVel.x + footAngVel.y * footAngVel.y + footAngVel.z * footAngVel.z);
            slipReward += Mathf.Sqrt(speedNorm);  // Paper: sqrt(norm)
        }

        // Right foot slip
        if (rightFoot != null && rightFoot.isGrounded && rightFootBody != null) {
            Vector3 footVel = rightFootBody.linearVelocity;
            Vector3 footAngVel = rightFootBody.angularVelocity;
            float speedNorm = Mathf.Sqrt(footVel.x * footVel.x + footVel.y * footVel.y + footVel.z * footVel.z
                                         + footAngVel.x * footAngVel.x + footAngVel.y * footAngVel.y + footAngVel.z * footAngVel.z);
            slipReward += Mathf.Sqrt(speedNorm);  // Paper: sqrt(norm)
        }

        return slipReward;
    }

    private Vector3 GetVirtualLeg(Transform hip, Transform ankle){ 
        Vector3 worldLeg = ankle.position - hip.position;
        Vector3 localLeg = baseLink.transform.InverseTransformDirection(worldLeg);
        return localLeg;
    }

    private void UpdateVirtualLegEvents() {
        bool leftGrounded = (leftFoot != null && leftFoot.isGrounded);
        bool rightGrounded = (rightFoot != null && rightFoot.isGrounded);

        //Lift-Off 감지
        eventLF_L = !leftGrounded && lastLeftFootGrounded;
        eventLF_R = !rightGrounded && lastRightFootGrounded;

        //Touch-Down 감지
        eventTD_L = leftGrounded && !lastLeftFootGrounded;
        eventTD_R = rightGrounded && !lastRightFootGrounded;

        if (eventLF_L) {
            virtualLegL_LF = GetVirtualLeg(leftHipJoint.transform, leftAnkleJoint.transform);
        }
        if (eventTD_L) {
            virtualLegL_TD = GetVirtualLeg(leftHipJoint.transform, leftAnkleJoint.transform);
        }
        if (eventLF_R) {
            virtualLegR_LF = GetVirtualLeg(rightHipJoint.transform, rightAnkleJoint.transform);
        }
        if (eventTD_R) {
            virtualLegR_TD = GetVirtualLeg(rightHipJoint.transform, rightAnkleJoint.transform);
        }

        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        if (Mathf.Abs(swingProgress - 0.5f) < 0.05f) {
            if (m_GaitPhase < 0.5f) {
                virtualLegL_Mid = GetVirtualLeg(leftHipJoint.transform, leftAnkleJoint.transform);
            } else {
                virtualLegR_Mid = GetVirtualLeg(rightHipJoint.transform, rightAnkleJoint.transform);
            }
        }
        //이전 프레임 기록
        lastLeftFootGrounded = leftGrounded;
        lastRightFootGrounded = rightGrounded;
        
    }
    private float RewardVirtualLegSym() {
        float rew = 0f;

        if(eventLF_L) {
            rew += Mathf.Exp(-Mathf.Abs(virtualLegL_LF.z + virtualLegL_TD.z) * 10f);
        }
        if(eventLF_R) {
            rew += Mathf.Exp(-Mathf.Abs(virtualLegR_LF.z + virtualLegR_TD.z) * 10f);
        }

        if(eventTD_L) {
            rew += 2f * Mathf.Exp(-Mathf.Abs(virtualLegL_Mid.z) * 30f);
            rew += 3f * Mathf.Exp(-Mathf.Abs(virtualLegL_TD.x - virtualLegR_TD.x) * 10f);
        }
        if(eventTD_R) {
            rew += 2f * Mathf.Exp(-Mathf.Abs(virtualLegR_Mid.z) * 30f);
            rew += 3f * Mathf.Exp(-Mathf.Abs(virtualLegR_TD.x - virtualLegL_TD.x) * 10f);
        }
        return rew;
    }

    private float RewardVirtualLegSymCont(){
        float rew = 0f;
        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        bool firstHalf = swingProgress < 0.5f;
        bool secondHalf = swingProgress >= 0.5f;

        if(m_GaitPhase < 0.5f) {
            Vector3 vLeg = GetVirtualLeg(leftHipJoint.transform, leftAnkleJoint.transform);
            if (firstHalf && vLeg.z > 0) rew -= 1f;
            if (secondHalf && vLeg.z < 0) rew -= 1f;
        }
        else {
            Vector3 vLeg = GetVirtualLeg(rightHipJoint.transform, rightAnkleJoint.transform);
            if (firstHalf && vLeg.z > 0) rew -= 1f;
            if (secondHalf && vLeg.z < 0) rew -= 1f;
        }
        return rew;
    }

    // flat_orientation_l2: L2 penalty for body tilt (gravity projection)
    private float RewardFlatOrientationL2() {
        // Project gravity into body frame
        Vector3 gravityWorld = Vector3.down;
        Vector3 gravityBody = baseLink.transform.InverseTransformDirection(gravityWorld);
        // L2 penalty on x,z components (should be 0 if upright)
        // Paper: -6 * (gx^2 + gz^2)
        return gravityBody.x * gravityBody.x + gravityBody.z * gravityBody.z;
    }

    // vel_mismatch_exp: Exponential penalty for velocity mismatch
    private float RewardVelMismatchExp() {
        Vector3 currentVel = baseLink.linearVelocity;
        Vector3 velError = baseLinVelDes - currentVel;
        // Paper: exp(-|error| * sigma) - 1
        return Mathf.Exp(-velError.magnitude * 5f) - 1f;
    }       



    // ==================== COLLISION HANDLING ====================

    public void HandleGroundCollision() {
        AddReward(scale_collision);
        EndEpisode();
    }

    // ==================== HEURISTIC ====================

    public override void Heuristic(in ActionBuffers actionsOut) {
        var cont = actionsOut.ContinuousActions;
        cont[0] = Input.GetKey(KeyCode.Alpha1) ? 1f : (Input.GetKey(KeyCode.Alpha2) ? -1f : 0f);
        cont[1] = Input.GetKey(KeyCode.Alpha3) ? 1f : (Input.GetKey(KeyCode.Alpha4) ? -1f : 0f);
        cont[2] = Input.GetKey(KeyCode.Alpha5) ? 1f : (Input.GetKey(KeyCode.Alpha6) ? -1f : 0f);
        cont[3] = Input.GetKey(KeyCode.W) ? 1f : (Input.GetKey(KeyCode.S) ? -1f : 0f);
        cont[4] = Input.GetKey(KeyCode.Q) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        cont[5] = Input.GetKey(KeyCode.E) ? 1f : (Input.GetKey(KeyCode.D) ? -1f : 0f);
    }
}
