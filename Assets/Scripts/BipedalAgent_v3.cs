using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

public class BipedalAgent_v3 : Agent {
    // =====================================================================
    // ====== LEFT JOINTS =====
    // =====================================================================
    public ArticulationBody leftHipYaw;
    public ArticulationBody leftHipRoll;
    public ArticulationBody leftHipPitch;
    public ArticulationBody leftKneePitch;
    public ArticulationBody leftAnklePitch;
    public ArticulationBody leftAnkleRoll;
    // ===================================================================
    // ====== Right JOINTS =====
    // ===================================================================
    public ArticulationBody rightHipYaw;
    public ArticulationBody rightHipRoll;
    public ArticulationBody rightHipPitch;
    public ArticulationBody rightKneePitch;
    public ArticulationBody rightAnklePitch;
    public ArticulationBody rightAnkleRoll;
    // ===================================================================
    // ===== Base Link =====
    // ===================================================================
     public ArticulationBody baseLink;   
    // ===================================================================
    // ===== Foot Contact =====
    // ===================================================================
    public FootContact leftFoot;
    public FootContact rightFoot;
    private Vector3[] footCheckpoints = new Vector3[] {new Vector3(0f, 0f, 0.125f), new Vector3(0f, 0f, -0.125f), new Vector3(0.06f, 0f, 0f), new Vector3(-0.06f, 0f, 0f)};
    // 위, 아래, 오른쪽, 왼쪽
    // ===================================================================
    // ===== Target =====
    // ===================================================================
    public Transform target;
    // ===================================================================
    // ===== Joint Array =====
    // ===================================================================
    private ArticulationBody[] joints;
    private float[] defaultJointAngles;
    // ===================================================================
    // ===== Training Area =====
    // ===================================================================
    public Transform Ground;
    // ===================================================================
    // ===== CONFIGURATION =====
    // ===================================================================
    private float initialHipPitch = 30f;
    private float initialKneePitch = -60f;
    private float initialAnklePitch = 30f;
    private float initialBaseHeight = 0.65962f;
    private float targetFeetSwingHeight = 0.2f;
    private float targetWalkingSpeed = 0.767f; //보폭 0.69m, 프루드 수 0.1 세팅
    private float m_GaitPeriod = 2f; // 한 다리당 1sec 소요

    // ===================================================================
    // ===== Clock =====
    // ===================================================================
    private float m_GaitPhase = 0f;
    private float movementDuration = 10f;
    private float movementTimer = 0f;

    // ===================================================================
    // ===== State Tracking =====
    // ===================================================================
    private float[] lastActions = new float[12];
    private float[] lastlastActions = new float[12];
    private Vector3 lastBaseVelocity;

    private Quaternion startRot;
    // ===================================================================
    // ===== Pose =====
    // ===================================================================
    private Vector3 targetPos;
    // ===================================================================
    // ===== Virtual Leg =====
    // ===================================================================
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
    // ===================================================================
    // ===== Reward Scale =====
    // ===================================================================
    private float scale_forward_velocity = 25f;  // 10 -> 25 증가
    private float scale_latera_velocity = 5f;
    private float scale_yaw = 3f;
    private float scale_feet_orientation = 2f;
    private float scale_feet_contact_number = 3f;
    private float scale_grounded_feet_flat = -2f;
    private float scale_feet_swingY = 10f;
    private float scale_feet_dswingY = 1.5f;
    private float scale_feet_dswingX = 3f;
    private float scale_virtual_leg_sym = 55f;
    private float scale_virtual_leg_sym_cont = 1.2f;
    private float scale_orientation = 3f;
    private float scale_orientation_l2 = -6f;
    private float scale_base_height = 4f;
    private float scale_action_smoothness = -0.2f;
    private float scale_base_acc = 0.2f;
    private float scale_feet_slip = -0.1f;
    private float scale_dof_vel = -5e-4f;
    private float scale_standing_penalty = 0f;  // 제자리 서있기 penalty (5 -> 2 완화)
    private float scale_default_joint_pos = 4f;
    // ===================================================================
    // ===== Swing Parameters =====
    // ===================================================================
    private float dfeY_swing_max = 1.0f;  // 스윙 발의 최대 수직 속도
    // ===================================================================
    // ===== Initialization ======
    // ===================================================================
    public override void Initialize() {
        joints = new ArticulationBody[] {
            leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll,
            rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll
        };

        // HipYaw, HipRoll, HipPitch, KneePitch, AnklePitch, AnkleRoll
        defaultJointAngles = new float[] {0f, 0f, initialHipPitch, initialKneePitch, initialAnklePitch, 0f, 0f, 0f, initialHipPitch, initialKneePitch, initialAnklePitch, 0f};

        lastActions = new float[12];
        lastlastActions = new float[12];

    }
    // ===================================================================
    // ===== Episode Begin =====
    // ===================================================================
    public override void OnEpisodeBegin() {
        // ===== Reset Initial Rocation =====
        Vector3 centerPos = Ground.position;
        centerPos.y = initialBaseHeight;
        baseLink.TeleportRoot(centerPos, Quaternion.identity);
        startRot = baseLink.transform.rotation;
        // ===== Reset Action History =====
        for (int i = 0; i < 12; i ++) {
            lastActions[i] = 0f;
            lastlastActions[i] = 0f;
        }
        // ===== Reset Velocity =====
        baseLink.linearVelocity = Vector3.zero;
        baseLink.angularVelocity = Vector3.zero;
        lastBaseVelocity = Vector3.zero;
        // ===== Reset Pose =====
        for (int i = 0; i < joints.Length; i++){
            if (joints[i] == null) continue;
            joints[i].jointPosition = new ArticulationReducedSpace(defaultJointAngles[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);
            var dr = joints[i].xDrive;
            dr.stiffness = 10000f;
            dr.damping = 100f;
            dr.forceLimit = 1000f;
            dr.target = defaultJointAngles[i];
            joints[i].xDrive = dr;
        }
        // ===== Virtual Leg =====
        lastLeftFootGrounded = false;
        lastRightFootGrounded = false;
        virtualLegL_LF = Vector3.zero;
        virtualLegL_TD = Vector3.zero;
        virtualLegL_Mid = Vector3.zero;
        virtualLegR_LF = Vector3.zero;
        virtualLegR_TD = Vector3.zero;
        virtualLegR_Mid = Vector3.zero;

        // ===== Target =====
        float localAngle = 0f;
        float dist = 10f;

        Vector3 localDisplacement = new Vector3(Mathf.Sin(localAngle) * dist, 0, Mathf.Cos(localAngle) * dist);
        targetPos = centerPos + localDisplacement;
        target.position = new Vector3(targetPos.x, 0.15f, targetPos.z);

        // ===== Reset State =====
        m_GaitPhase = 0f;
        movementTimer = 0f;
    }

    // ===== Collect Observations (49 dims) =====
    public override void CollectObservations(VectorSensor sensor) {
        // ===== 1. Gait Phase [2 dims] =====
        float phase_gait = m_GaitPhase * 2f * Mathf.PI; ///
        sensor.AddObservation(Mathf.Sin(phase_gait));
        sensor.AddObservation(Mathf.Cos(phase_gait));

        // ===== 2. Target Info (Body Frame) [3 dims] =====
        Vector3 toTarget = targetPos - baseLink.transform.position;
        Vector3 localTarget = baseLink.transform.InverseTransformDirection(toTarget);
        Vector3 localDir = localTarget.normalized;
        float distToTarget = toTarget.magnitude;
        sensor.AddObservation(localDir.x);
        sensor.AddObservation(localDir.z);
        sensor.AddObservation(Mathf.Clamp(distToTarget, 0f, 10f) / 10f);

        // ===== 3. Base Orientation (roll, pitch) [2 dims] =====
        Vector3 euler = baseLink.transform.eulerAngles;
        float roll = NormalizeAngle(euler.z) * Mathf.Deg2Rad;
        float pitch = NormalizeAngle(euler.x) * Mathf.Deg2Rad;
        sensor.AddObservation(roll);
        sensor.AddObservation(pitch);

        // ===== 4. Joint Positions [12 dims] =====
        foreach (var j in joints) {
            float pos = (j != null) ? j.jointPosition[0] : 0f;
            sensor.AddObservation(pos);
        }

        // ===== 5. Joint Velocities [12 dims] =====
        foreach (var j in joints) {
            float vel = (j != null) ? j.jointVelocity[0] * 0.05f : 0f;
            sensor.AddObservation(vel);
        }

        // ===== 6. Last Actions [12 dims] =====
        foreach (float a in lastActions) {
            sensor.AddObservation(a);
        }

        // ===== 7. Base Angular Velocity (Body Frame) [3 dims] =====
        Vector3 angVelWorld = baseLink.angularVelocity;
        Vector3 angVelBody = baseLink.transform.InverseTransformDirection(angVelWorld);
        sensor.AddObservation(angVelBody * 0.25f);

        // ===== 8. Base Linear Velocity (Body Frame) [3 dims] =====
        Vector3 linVelWorld = baseLink.linearVelocity;
        Vector3 linVelBody = baseLink.transform.InverseTransformDirection(linVelWorld);
        sensor.AddObservation(linVelBody * 0.5f);
    }

    private float NormalizeAngle(float angle) {
        if (angle > 180f) angle -= 360f;
        return angle;
    }

    // ===== Control Parameters =====
    [Header("Control")]
    public float actionScale = 25f;

    public override void OnActionReceived(ActionBuffers actions) {
        float dt = Time.fixedDeltaTime;
        movementTimer += dt;
        
        // ===== Update Gait Phase =====
        m_GaitPhase += dt / m_GaitPeriod;
        m_GaitPhase %= 1f;

        // ===== Update Virtual Leg Events =====
        UpdateVirtualLegEvents();

        // ===== Apply Actions =====
        var continuousActions = actions.ContinuousActions;
        float[] currentActions = new float[12];

        for (int i = 0; i < 12; i++) {
            currentActions[i] = continuousActions[i];
        }

        for (int i = 0; i < 12; i++) {
            if (joints[i] == null) continue;
            float targetAngle = defaultJointAngles[i] + currentActions[i] * actionScale;
            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }

        // ===== Calculate Rewards =====
        CalculateRewards(currentActions);

        // ===== Update Last Actions =====
        for (int i = 0; i < 12; i++) {
            lastlastActions[i] = lastActions[i];
            lastActions[i] = currentActions[i];
        }
        lastBaseVelocity = baseLink.linearVelocity;
    }

    private void CalculateRewards(float[] currentActions) {
        float totalReward = 0;
        // ====== 1. Survival Check =====
        RewardSurvival();
        // ===== 2. Target Direction Following =====
        totalReward += RewardForwardVelocity() * scale_forward_velocity;
        totalReward += RewardLateralVelocity() * scale_latera_velocity;
        totalReward += RewardYaw() * scale_yaw;

        // ===== 2. Feet Option  =====
        totalReward += RewardFeetOrientation() * scale_feet_orientation;
        totalReward += RewardFeetContactNumber() * scale_feet_contact_number;
        totalReward += RewardGroundFeetFlat() * scale_grounded_feet_flat;
        totalReward += RewardFeetSwingY() * scale_feet_swingY;
        totalReward += RewardFeetDSwingY() * scale_feet_dswingY;
        totalReward += RewardFeetDSwingX() * scale_feet_dswingX;
        //totalReward += RewardKneeBend()
        totalReward += RewardVirtualLegSym() * scale_virtual_leg_sym;
        totalReward += RewardVirtualLegSymCont() * scale_virtual_leg_sym_cont;

        // ===== 3. Stability =====
        totalReward += RewardOrientation() * scale_orientation;
        totalReward += RewardFlatOrientationL2() * scale_orientation_l2;
        totalReward += RewardActionSmoothness(currentActions) * scale_action_smoothness;
        totalReward += RewardBaseHeight() * scale_base_height;
        totalReward += RewardBaseAcc() * scale_base_acc;
        totalReward += RewardDofVel() * scale_dof_vel;
        totalReward += RewardFeetSlip() * scale_feet_slip;

        // ===== 4. Standing Penalty (제자리 서있기 방지) =====
        totalReward += RewardStandingPenalty() * scale_standing_penalty;
        totalReward += RewardDefaultJointPos() * scale_default_joint_pos;

        AddReward(totalReward);

        // ===== TIMEOUT CHECK =====
        if (movementTimer > movementDuration) {
            EndEpisode();
        }
    }
    // ===================================================================
    // ===== Reward Functions ======
    // ===================================================================
    private void RewardSurvival() {
        float upright = Vector3.Dot(baseLink.transform.up, Vector3.up);
        if(upright < 0.5f) {
            AddReward(-5f);
            EndEpisode();
            return;
        }
    }

    private float RewardForwardVelocity() {
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        Vector3 targetDir = toTarget.normalized;

        Vector3 currentVel = baseLink.linearVelocity;
        currentVel.y = 0;

        // 타겟 방향 속도 스칼라
        float velTowardTarget = Vector3.Dot(currentVel, targetDir);
        float velError = velTowardTarget - targetWalkingSpeed;

        float reward = Mathf.Exp(-velError * velError * 4f);
        float penalty = velError * velError * 0.5f;
        
        return reward - penalty;
    }
    private float RewardLateralVelocity() {
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        Vector3 targetDir = toTarget.normalized;

        Vector3 currentVel = baseLink.linearVelocity;
        currentVel.y = 0;

        Vector3 lateralVel = currentVel - Vector3.Dot(currentVel, targetDir) * targetDir;
        float lateralSpeed = lateralVel.magnitude;
        return Mathf.Exp(-lateralSpeed * 3f);

    }

    private float RewardYaw() {
        float currentYaw = baseLink.transform.eulerAngles.y;
        float initialYaw = startRot.eulerAngles.y;
        float yawError = Mathf.Abs(Mathf.DeltaAngle(currentYaw, initialYaw));
        return Mathf.Exp(-yawError * 0.05f);
    }

    private float GetSwingFootTargetHeight() {
        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        float heightMul = 4f * swingProgress * (1f - swingProgress);
        float ankleHeight = 0.02f;
        return targetFeetSwingHeight * heightMul + ankleHeight;
    }

    private float RewardGroundFeetFlat() {
        float penalty = 0;

        if (leftFoot != null && leftFoot.isGrounded) {
            Vector3 footup = leftAnkleRoll.transform.up;
            float dot = Vector3.Dot(footup, Vector3.up);
            float angle = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f));
            penalty += angle * angle;
        }

        if (rightFoot != null && rightFoot.isGrounded) {
            Vector3 footup = rightAnkleRoll.transform.up;
            float dot = Vector3.Dot(footup, Vector3.up);
            float angle = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f));
            penalty += angle * angle;
            
        }
        return penalty;
    }
    //발 사이즈 (x, y, z) (0.12, 0.04, 0.25)
    private float RewardFeetOrientation() {
        float reward = 0;
        bool leftContact = (leftFoot != null && leftFoot.isGrounded);
        bool rightContact = (rightFoot != null && rightFoot.isGrounded);
        if(!leftContact && rightContact) {
            reward += GetFootFlatnessReward(leftAnkleRoll);
        }
        if(leftContact && !rightContact) {
            reward += GetFootFlatnessReward(rightAnkleRoll);
        }
        return reward;
    }
    
    private float GetFootFlatnessReward(ArticulationBody foot) {
        float totalVariance = 0f;
        float[] heights = new float[footCheckpoints.Length];

        for (int i = 0; i < footCheckpoints.Length; i ++ ) {
            //foot.transform.TransfromPoint
            Vector3 worldPos = foot.transform.TransformPoint(footCheckpoints[i]);
            heights[i] = worldPos.y;
        }
        float avgHeight = (heights[0] + heights[1] + heights[2] + heights[3]) / 4f;
        for (int i = 0; i < heights.Length; i++) {
            float diff = heights[i] - avgHeight;
            totalVariance += diff * diff;
        }
        return Mathf.Exp(-totalVariance * 100f);
    }
    private float RewardFeetContactNumber() {
        bool leftStance = (m_GaitPhase >= 0.5f);
        bool rightStance = (m_GaitPhase < 0.5f);

        bool leftContact = (leftFoot != null && leftFoot.isGrounded);
        bool rightContact = (rightFoot != null && rightFoot.isGrounded);

        float reward = 0f;
        reward += (leftContact == leftStance) ? 2f : -0.3f;
        reward += (rightContact == rightStance) ? 2f : -0.3f;

        return reward * 0.5f;
    }

    private float RewardFeetSwingY() {
        bool leftSwing = (m_GaitPhase < 0.5f);
        bool rightSwing = (m_GaitPhase >= 0.5f);

        float reward = 0f;

        if(leftSwing && leftAnkleRoll != null) {
            float footY = leftAnkleRoll.transform.position.y;
            float refY = GetSwingFootTargetHeight();
            float err = Mathf.Abs(footY - refY);
            reward += Mathf.Exp(-err * 80f) - 20f * err * err;
        }

        if(rightSwing && rightAnkleRoll != null) {
            float footY = rightAnkleRoll.transform.position.y;
            float refY = GetSwingFootTargetHeight();
            float err = Mathf.Abs(footY - refY);
            reward += Mathf.Exp(-err * 80f) - 20f * err * err;
        }
        return reward;
    }
    private float RewardFeetDSwingY() {
        float rew = 0f;
        bool leftSwing = (m_GaitPhase < 0.5f);
        bool rightSwing = (m_GaitPhase >= 0.5f);

        // 각 발의 스윙 진행도 (0.0 ~ 1.0)
        float swingProgress = leftSwing ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);

        // 스윙 전반기: 발을 들어올림 (+Y 속도) / 후반기: 발을 내림 (-Y 속도)
        float refDY = (swingProgress < 0.5f) ? dfeY_swing_max : -dfeY_swing_max;

        if (leftSwing && leftAnkleRoll != null) {
            float footDY = leftAnkleRoll.linearVelocity.y;
            float err = Mathf.Abs(refDY - footDY);
            rew += Mathf.Exp(-err * 5f) - 0.1f * err * err;
        }
        if (rightSwing && rightAnkleRoll != null) {
            float footDY = rightAnkleRoll.linearVelocity.y;
            float err = Mathf.Abs(refDY - footDY);
            rew += Mathf.Exp(-err * 5f) - 0.1f * err * err;
        }
        return rew;
    }

    private float RewardFeetDSwingX() {
        // 스윙 발의 측면(X) 속도를 최소화 - 발이 옆으로 흔들리지 않도록
        float reward = 0f;
        bool leftSwing = (m_GaitPhase < 0.5f);
        bool rightSwing = (m_GaitPhase >= 0.5f);

        if (leftSwing && leftAnkleRoll != null) {
            Vector3 worldVel = leftAnkleRoll.linearVelocity;
            Vector3 localVel = baseLink.transform.InverseTransformDirection(worldVel);
            float sqX = localVel.x * localVel.x;
            reward += Mathf.Exp(-sqX * 20f);
        }
        if (rightSwing && rightAnkleRoll != null) {
            Vector3 worldVel = rightAnkleRoll.linearVelocity;
            Vector3 localVel = baseLink.transform.InverseTransformDirection(worldVel);
            float sqX = localVel.x * localVel.x;
            reward += Mathf.Exp(-sqX * 20f);
        }
        return reward;
    }

    private Vector3 GetVirtualLeg(Transform hip, Transform ankle) {
        Vector3 worldLeg = ankle.position - hip.position;
        Vector3 localLeg = baseLink.transform.InverseTransformDirection(worldLeg);
        return localLeg;
    }

    private void UpdateVirtualLegEvents() {
        bool leftGrounded = (leftFoot != null && leftFoot.isGrounded);
        bool rightGrounded = (rightFoot != null && rightFoot.isGrounded);
        eventLF_L = !leftGrounded && lastLeftFootGrounded;
        eventLF_R = !rightGrounded && lastRightFootGrounded;
        
        eventTD_L = leftGrounded && !lastLeftFootGrounded;
        eventTD_R = rightGrounded && ! lastRightFootGrounded;

        if (eventLF_L) {
            virtualLegL_LF = GetVirtualLeg(leftHipPitch.transform, leftAnkleRoll.transform);
        }
        if (eventTD_L) {
            virtualLegL_TD = GetVirtualLeg(leftHipPitch.transform, leftAnkleRoll.transform);
        }
        if (eventLF_R) {
            virtualLegR_LF = GetVirtualLeg(rightHipPitch.transform, rightAnkleRoll.transform);
        }
        if (eventTD_R) {
            virtualLegR_TD = GetVirtualLeg(rightHipPitch.transform, rightAnkleRoll.transform);
        }

        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        if (Mathf.Abs(swingProgress - 0.5f) < 0.05f) {
            if (m_GaitPhase < 0.5f) {
                virtualLegL_Mid = GetVirtualLeg(leftHipPitch.transform, leftAnkleRoll.transform);
            } else {
                virtualLegR_Mid = GetVirtualLeg(rightHipPitch.transform, rightAnkleRoll.transform);
            }
        }
        lastLeftFootGrounded = leftGrounded;
        lastRightFootGrounded = rightGrounded;
    }

    private float RewardVirtualLegSym() {
        float reward = 0f;
        if (eventLF_L) {
            float symError = Mathf.Abs(virtualLegL_LF.z + virtualLegL_TD.z);
            reward += Mathf.Exp(-symError * 10f);
        }
        if (eventLF_R) {
            float symError = Mathf.Abs(virtualLegR_LF.z + virtualLegR_TD.z);
            reward += Mathf.Exp(-symError * 10f);
        }
        if (eventTD_L) {
            float midError = Mathf.Abs(virtualLegL_Mid.z);
            reward += 2f * Mathf.Exp(-midError * 30f);

            float balError = Mathf.Abs(virtualLegL_TD.z - virtualLegR_TD.z);
            reward += 3f * Mathf.Exp(-balError * 30f);   
        }
        if (eventTD_R) {
            float midError = Mathf.Abs(virtualLegR_Mid.z);
            reward += 2f * Mathf.Exp(-midError * 30f);

            float balError = Mathf.Abs(virtualLegL_TD.z - virtualLegR_TD.z);
            reward += 3f * Mathf.Exp(-balError * 30f);               
        }
        return reward;
    }

    private float RewardVirtualLegSymCont() {
        float reward = 0f;
        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        bool firstHalf = swingProgress < 0.5f; //발이 뒤에 있어야함
        bool secondHalf = swingProgress >= 0.5f; //발이 앞에 있어야함

        if(m_GaitPhase < 0.5f) {
            Vector3 vLeg = GetVirtualLeg(leftHipPitch.transform, leftAnkleRoll.transform);
            if (firstHalf && vLeg.z > 0) reward -= 1f;
            if (secondHalf && vLeg.z < 0) reward -= 1f;
        }
        else {
            Vector3 vLeg = GetVirtualLeg(rightHipPitch.transform, rightAnkleRoll.transform);
            if (firstHalf && vLeg.z > 0) reward -= 1f;
            if (secondHalf && vLeg.z < 0) reward -= 1f;
                   
        }
        return reward;
    }

    //목표 근처에서 부드러운 그라디언트
    private float RewardOrientation() {
        Vector3 euler = baseLink.transform.rotation.eulerAngles;
        float pitch = NormalizeAngle(euler.x) * Mathf.Deg2Rad;
        float roll = NormalizeAngle(euler.z) * Mathf.Deg2Rad;

        float quat_mismatch = Mathf.Exp(-(Mathf.Abs(pitch) + Mathf.Abs(roll)) * 30f);
        float proj_gravity = Mathf.Exp(-Mathf.Sqrt(pitch*pitch + roll*roll) * 40f);

        return (quat_mismatch + proj_gravity) * 0.5f;
    }
    //직접적 적용
    private float RewardFlatOrientationL2() {
        Vector3 euler = baseLink.transform.eulerAngles;
        float pitch = NormalizeAngle(euler.x) * Mathf.Deg2Rad;
        float roll = NormalizeAngle(euler.z) * Mathf.Deg2Rad;
        return pitch * pitch + roll * roll;
    }

    private float RewardBaseHeight() {
        float currentHeight = baseLink.transform.position.y;
        float err = Mathf.Abs(currentHeight - initialBaseHeight);

        float reward = Mathf.Exp(-err * 20f) - 15f * err * err;
        return reward;
    }

    private float RewardActionSmoothness(float[] currentActions) {
        float term1 = 0f;
        float term2 = 0f;

        for (int i = 0; i < 12; i++) {
            float diff1 = currentActions[i] - lastActions[i];
            term1 += diff1 * diff1;

            float diff2 = currentActions[i] - 2f * lastActions[i] + lastlastActions[i];
            term2 += diff2 * diff2;
        }
        return term1 + 0.1f * term2;
    }
    private float RewardBaseAcc() {
        Vector3 currentVel = baseLink.linearVelocity;
        Vector3 acc = (currentVel - lastBaseVelocity) / Time.fixedDeltaTime;
        float accNorm = acc.magnitude;
        float reward = Mathf.Exp(-accNorm * 6f);
        return reward;
    }

    private float RewardDofVel() {
        float sumSq = 0f;
        foreach(var j in joints) {
            float vel = j.jointVelocity[0];
            sumSq += vel * vel;
        }
        return sumSq;
    }

    private float RewardFeetSlip() {
        float reward = 0;
        if(leftFoot != null && leftFoot.isGrounded) {
            Vector3 footVel = leftAnkleRoll.linearVelocity;
            //Vector3 footAngVel = leftAnkleRoll.angularVelocity;
            float speedNorm = Mathf.Sqrt(footVel.x * footVel.x + footVel.y * footVel.y + footVel.z * footVel.z );
            reward += Mathf.Sqrt(speedNorm);  // Paper: sqrt(norm)
        }

        if(rightFoot != null && rightFoot.isGrounded) {
            Vector3 footVel = rightAnkleRoll.linearVelocity;
            //Vector3 footAngVel = rightAnkleRoll.angularVelocity;
            float speedNorm = Mathf.Sqrt(footVel.x * footVel.x + footVel.y * footVel.y + footVel.z * footVel.z);
            reward += Mathf.Sqrt(speedNorm);  // Paper: sqrt(norm)
        }
        return reward;
    }

    private float RewardStandingPenalty() {
        // 타겟 방향 속도가 목표 속도의 30% 미만이면 penalty
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        Vector3 targetDir = toTarget.normalized;

        Vector3 currentVel = baseLink.linearVelocity;
        currentVel.y = 0;

        float velTowardTarget = Vector3.Dot(currentVel, targetDir);
        float minSpeed = targetWalkingSpeed * 0.3f;  // 최소 0.23 m/s

        if (velTowardTarget < minSpeed) {
            // 속도가 낮을수록 더 큰 penalty
            float deficit = minSpeed - velTowardTarget;
            return -deficit * 3f;  // 음수 반환
        }
        return 0f;
    }
    private float RewardDefaultJointPos(){
        float error = 0f;
        for (int i = 0; i < 12; i ++) {
            if(joints[i] == null) continue;
            float diff = joints[i].jointPosition[0] - defaultJointAngles[i] * Mathf.Deg2Rad;
            error += diff * diff;
        }
        return Mathf.Exp(-error * 0.5f);   
    }
    // ===================================================================
    // ===== Debug Visualization ======
    // ===================================================================
    void Update() {
        // baseLink의 forward 방향 (파란색)
        Debug.DrawRay(baseLink.transform.position, baseLink.transform.forward * 3f, Color.blue);

        // baseLink에서 타겟으로의 방향 (빨간색)
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;  // 수평 방향만
        Debug.DrawRay(baseLink.transform.position, toTarget.normalized * 3f, Color.red);

        float maxJointVel = 0f;
        foreach (var j in joints) {
            if (j != null) maxJointVel = Mathf.Max(maxJointVel, Mathf.Abs(j.jointVelocity[0]));
        }
        Debug.Log($"JointVel: {maxJointVel:F1}, AngVel: {baseLink.angularVelocity.magnitude:F1}, LinVel: {baseLink.linearVelocity.magnitude:F1}");
    }


    // ===== Collision Handling =====
    public void HandleGroundCollision() {
        AddReward(-10f);
        EndEpisode();
    }

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