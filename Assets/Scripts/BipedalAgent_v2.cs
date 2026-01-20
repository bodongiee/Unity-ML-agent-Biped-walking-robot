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

    // ===== REWARD SCALES (Paper: cfg_p2p_s1.py) =====
    [Header("Reward Scales")]
    public float scale_pos_trace_pXZ = 4f;
    public float scale_pos_trace_pXZ_vel = 6f;
    public float scale_pos_trace_thetaY = 5f;
    public float scale_pos_trace_thetaY_vel = 5f;
    public float scale_pos_stop = 5f;
    public float scale_feet_contact_number = 3f;
    public float scale_feet_swingZ = 10f;
    public float scale_feet_orientation = 2.5f;
    public float scale_feet_slip = -0.1f;
    public float scale_orientation = 3f;
    public float scale_base_height = 4f;
    public float scale_action_smoothness = -0.2f;
    public float scale_dof_vel = -5e-4f;
    public float scale_collision = -1f;
    public float scale_survival = 0.01f;

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
    private float m_GaitPeriod = 0.9f;  // Paper: 0.9s per cycle

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

    // ===== DESIRED STATE (for velocity tracking) =====
    private Vector3 basePosDes;
    private Vector3 baseLinVelDes;
    private float baseThetaYDes;
    private float baseAngVelYDes;

    // ===== CONFIGURATION =====
    private float initialHipAngle = -30f;
    private float initialKneeAngle = 60f;
    private float initialAnkleAngle = -30f;
    private float targetBaseHeight = 0.65f;
    private float targetFeetSwingHeight = 0.05f;

    // ===== OBSERVATION SCALING =====
    private float obs_scale_dof_pos = 1f;
    private float obs_scale_dof_vel = 0.05f;
    private float obs_scale_ang_vel = 0.2f;
    private float obs_scale_quat = 1f;
    private float obs_scale_feet_pos = 1f;

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

        if (baseLink != null) {
            targetBaseHeight = baseLink.transform.position.y;
        }
    }

    public override void OnEpisodeBegin() {
        // Reset Clocks
        m_MovementPhase = 0f;
        m_GaitPhase = 0f;
        movementTimer = 0f;

        // Reset Actions History (6)
        for (int i = 0; i < 6; i++) {
            lastActions[i] = 0f;
            lastLastActions[i] = 0f;
        }

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
        startPos.y = targetBaseHeight; // Ensure start height is correct
        startRot = Quaternion.identity;
        baseLink.TeleportRoot(startPos, startRot);
        baseLink.linearVelocity = Vector3.zero;
        baseLink.angularVelocity = Vector3.zero;
        lastBaseVelocity = Vector3.zero;

        // Generate Random Task
        float angle = Random.Range(-Mathf.PI, Mathf.PI);
        float dist = Random.Range(1f, 3f);
        targetPos = startPos + new Vector3(Mathf.Cos(angle) * dist, 0, Mathf.Sin(angle) * dist);
        targetHeading = Random.Range(-0.5f, 0.5f); // Radians
        movementDuration = Random.Range(3f, 6f);

        if (target != null) target.position = targetPos;

        // Setup Command Vector
        commands[0] = targetPos.x - startPos.x; // dx
        commands[1] = targetPos.z - startPos.z; // dy (Unity Z = forward)
        commands[2] = targetHeading;            // dtheta
        commands[3] = movementDuration;         // T

        previousDistanceToTarget = Vector3.Distance(startPos, targetPos);
        
        velErrorHistory.Clear();
    }

    public override void CollectObservations(VectorSensor sensor) {
        // ===== 1. CLOCK SIGNALS (4 dims) =====
        float phase_mov = m_MovementPhase * 2 * Mathf.PI;
        float phase_gait = m_GaitPhase * 2 * Mathf.PI;
        sensor.AddObservation(Mathf.Sin(phase_mov));
        sensor.AddObservation(Mathf.Cos(phase_mov));
        sensor.AddObservation(Mathf.Sin(phase_gait));
        sensor.AddObservation(Mathf.Cos(phase_gait));

        // ===== 2. COMMANDS (4 dims, scaled) =====
        for (int i = 0; i < 4; i++) {
            sensor.AddObservation(commands[i] * commandsScale[i]);
        }

        // ===== 3. JOINT POSITIONS (6 dims) =====
        foreach (var j in joints) {
            float pos = (j != null) ? j.jointPosition[0] : 0f;
            sensor.AddObservation(pos * obs_scale_dof_pos);
        }

        // ===== 4. JOINT VELOCITIES (6 dims) =====
        foreach (var j in joints) {
            float vel = (j != null) ? j.jointVelocity[0] : 0f;
            sensor.AddObservation(vel * obs_scale_dof_vel);
        }

        // ===== 5. LAST ACTIONS (6 dims) =====
        foreach (float a in lastActions) {
            sensor.AddObservation(a);
        }

        // ===== 6. BASE ANGULAR VELOCITY (3 dims) =====
        Vector3 angVel = baseLink.angularVelocity * obs_scale_ang_vel;
        sensor.AddObservation(angVel);

        // ===== 7. BASE EULER XY - Pitch/Roll (2 dims) =====
        Vector3 euler = baseLink.transform.rotation.eulerAngles;
        float pitch = NormalizeAngle(euler.x) * Mathf.Deg2Rad * obs_scale_quat;
        float roll = NormalizeAngle(euler.z) * Mathf.Deg2Rad * obs_scale_quat;
        sensor.AddObservation(pitch);
        sensor.AddObservation(roll);

        // ===== 8. FEET POSITION (Body Frame) (6 dims) =====
        if (leftFootTransform != null && rightFootTransform != null) {
            Vector3 lPosB = baseLink.transform.InverseTransformPoint(leftFootTransform.position) * obs_scale_feet_pos;
            Vector3 rPosB = baseLink.transform.InverseTransformPoint(rightFootTransform.position) * obs_scale_feet_pos;
            sensor.AddObservation(lPosB);
            sensor.AddObservation(rPosB);
        } else {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(Vector3.zero);
        }

        // TOTAL: 4 + 4 + 6 + 6 + 6 + 3 + 2 + 6 = 37 dims
    }

    private float NormalizeAngle(float angle) {
        if (angle > 180f) angle -= 360f;
        return angle;
    }

    [Header("Control Parameters")]
    public float actionScale = 60f; // Scale for joint target adjustment

    public override void OnActionReceived(ActionBuffers actions) {
        float dt = Time.fixedDeltaTime;

        // ===== UPDATE CLOCKS =====
        movementTimer += dt;
        m_MovementPhase = Mathf.Clamp01(movementTimer / movementDuration);

        m_GaitPhase += dt / m_GaitPeriod;
        m_GaitPhase %= 1f;

        // ===== UPDATE DESIRED STATE (for velocity tracking) =====
        UpdateDesiredState();

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
        baseThetaYDes = Mathf.Lerp(0, targetHeading, m_MovementPhase);

        // Desired velocity = displacement / time_per_step
        float delta_phi = Time.fixedDeltaTime / movementDuration;
        if (delta_phi > 0) {
            baseLinVelDes = (targetPos - startPos) * delta_phi / Time.fixedDeltaTime;
            baseAngVelYDes = targetHeading * delta_phi / Time.fixedDeltaTime;
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

        // ===== 2. POSITION TRACKING (XY) =====
        if (m_MovementPhase < 1f) {
            totalReward += RewardPosTracePXZ() * scale_pos_trace_pXZ;
        }

        // ===== 3. VELOCITY TRACKING (XY) =====
        if (m_MovementPhase < 1f) {
            totalReward += RewardPosTracePXZ_Vel() * scale_pos_trace_pXZ_vel;
        }

        // ===== 4. HEADING TRACKING =====
        if (m_MovementPhase < 1f) {
            totalReward += RewardPosTraceThetaY() * scale_pos_trace_thetaY;
            totalReward += RewardPosTraceThetaY_Vel() * scale_pos_trace_thetaY_vel;
        }

        // ===== 5. STOPPING REWARD =====
        if (m_MovementPhase >= 1f) {
            totalReward += RewardPosStop() * scale_pos_stop;
        }

        // ===== 6. FEET CONTACT NUMBER =====
        if (m_MovementPhase < 1f) {
            totalReward += RewardFeetContactNumber() * scale_feet_contact_number;
        }
        // ===== 7. FEET SWING Z =====
        if (m_MovementPhase < 1f) {
            totalReward += RewardFeetSwingZ() * scale_feet_swingZ;
        }

        // ===== 8. ORIENTATION =====
        if (m_MovementPhase < 1f) {
            totalReward += RewardOrientation() * scale_orientation;
        }

        // ===== 9. BASE HEIGHT =====
        totalReward += RewardBaseHeight() * scale_base_height;

        // ===== 10. ACTION SMOOTHNESS =====
        totalReward += RewardActionSmoothness(currentActions) * scale_action_smoothness;

        // ===== 11. DOF VELOCITY PENALTY =====
        totalReward += RewardDofVel() * scale_dof_vel;

        // ===== 12. FEET SLIP PENALTY =====
        totalReward += RewardFeetSlip() * scale_feet_slip;

        AddReward(totalReward);

        // ===== TIMEOUT CHECK =====
        if (movementTimer > movementDuration + 3f) {
            EndEpisode();
        }
    }

    // ==================== REWARD FUNCTIONS ====================

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

        float desiredYaw = baseThetaYDes;
        float yawErrorAbs = Mathf.Abs(currentYaw - desiredYaw);
        
        // Error to FINAL target heading
        float finalHeadingError = Mathf.Abs(currentYaw - targetHeading);
        
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
        // Reward swing foot for reaching target height
        bool leftIsSwing = (m_GaitPhase < 0.5f);
        bool rightIsSwing = (m_GaitPhase >= 0.5f);

        float rew = 0f;

        if (leftIsSwing && leftFootTransform != null) {
            float footZ = leftFootTransform.position.y;
            float refZ = GetSwingFootTargetZ();
            float err = Mathf.Abs(footZ - refZ);
            rew += Mathf.Exp(-err * 80f) - 20f * err * err;
        }

        if (rightIsSwing && rightFootTransform != null) {
            float footZ = rightFootTransform.position.y;
            float refZ = GetSwingFootTargetZ();
            float err = Mathf.Abs(footZ - refZ);
            rew += Mathf.Exp(-err * 80f) - 20f * err * err;
        }

        return rew;
    }

    private float GetSwingFootTargetZ() {
        // Parabolic trajectory for swing foot
        // Peak at mid-swing (phase = 0.25 or 0.75)
        float swingProgress = (m_GaitPhase < 0.5f) ? (m_GaitPhase * 2f) : ((m_GaitPhase - 0.5f) * 2f);
        //0.5보다 작으면 -> 왼발 스윙 0.5보다 크면 -> 오른발 스윙
        float heightMultiplier = 4f * swingProgress * (1f - swingProgress); // Parabola 0→1→0
        return targetFeetSwingHeight * heightMultiplier + 0.02f; // Base ankle height
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
