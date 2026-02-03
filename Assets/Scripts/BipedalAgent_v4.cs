using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

[System.Serializable]
public class TrajectoryData {
    public List<List<float>> q;    // 관절 위치
    public List<List<float>> v;    // 관절 속도
    public List<List<float>> tau;  // 토크
    public float dt;
}

public enum WalkingState { Standing, StartWalkingLF, StartWalkingRF, WalkingLF, WalkingRF, StoppingLF, StoppingRF, StandingComplete }

public class BipedalAgent_v4 : Agent {
    // =====================================================================
    // ====== Joints =====
    // =====================================================================
    public ArticulationBody leftHipYaw;
    public ArticulationBody leftHipRoll;
    public ArticulationBody leftHipPitch;
    public ArticulationBody leftKneePitch;
    public ArticulationBody leftAnklePitch;
    public ArticulationBody leftAnkleRoll;
    public ArticulationBody rightHipYaw;
    public ArticulationBody rightHipRoll;
    public ArticulationBody rightHipPitch;
    public ArticulationBody rightKneePitch;
    public ArticulationBody rightAnklePitch;
    public ArticulationBody rightAnkleRoll;
    public ArticulationBody baseLink;

    public FootContact leftFoot;
    public FootContact rightFoot;

    public Transform target;
    public Transform Ground;

    private ArticulationBody[] joints;
    private float[] defaultJointAngles = new float[12] { 0f, 0f, 30f, -60f, 30f, 0f, 0f, 0f, 30f, -60f, 30f, 0f };

    // =====================================================================
    // ===== Imitation Learning Data =====
    // =====================================================================
    [Header("Imitation Learning Trajectories")]
    public string standingSeqJsonPath = "Assets/IL/Actions/standing.json";
    public string startLfJsonPath = "Assets/IL/Actions/trajectory_start_rf_support_step.json";
    public string startRfJsonPath = "Assets/IL/Actions/trajectory_start_lf_support_step.json";
    public string walkingLfJsonPath = "Assets/IL/Actions_hard/trajectory_extract_cycle_lf_start_98.json";
    public string walkingRfJsonPath = "Assets/IL/Actions_hard/trajectory_extract_cycle_rf_start_98.json";
    public string stopLfJsonPath = "Assets/IL/Actions/trajectory_stop_rf_support_step.json";
    public string stopRfJsonPath = "Assets/IL/Actions/trajectory_stop_lf_support_step.json";

    private TrajectoryData standingSeq;
    private TrajectoryData startLF;
    private TrajectoryData startRF;
    private TrajectoryData walkingLF;
    private TrajectoryData walkingRF;
    private TrajectoryData stopLF;
    private TrajectoryData stopRF;

    private WalkingState currentState = WalkingState.Standing;
    private TrajectoryData currentTraj;
    private int currentFrame = 0;
    private float trajectoryFrameTimer = 0f;
    private float trajectoryDt = 0.02f;
    private float standingTimer = 0f;
    private float minStandingTime = 1.0f;
    private float standingCompleteTimer = 0f;
    private float minStandingCompleteTime = 1.0f;
    private bool startWithLeftFoot = true;

    [Header("Imitation Learning Settings")]
    public bool useImitation = true;  
    public float manualImitationWeight = 1.0f;
    private float imitationWeight = 1.0f;
    public bool debugMode = true;

    public int graceSteps = 50;
    public float earlyTerminationThreshold = 5.0f;
    // =====================================================================
    // ===== Configuration =====
    // =====================================================================
    [Header("Control Parameters")]
    public float actionScale = 5f; 
    
    private float[] jointMinAngles = new float[12] { -45f, -30f, -45f, -100f, -45f, -30f, -45f, -30f, -45f, -100f, -45f, -30f };
    private float[] jointMaxAngles = new float[12] { 45f, 30f, 100f, 0f, 45f, 30f, 45f, 30f, 100f, 0f, 45f, 30f };

    private float initialBaseHeight = 0.67f;

    [Header("Reward Weights")]
    public float scale_imitation_pose = 20.0f;
    public float scale_imitation_velocity = 3.0f;
    public float scale_com_tracking = 8.0f;
    public float scale_orientation = 1.0f;
    public float scale_base_height = 1.0f;
    public float scale_action_smoothness = 0.3f;
    public float scale_forward_progress = 8.0f;
    public float scale_stop_velocity = 2.0f;
    public float scale_trajectory_direction = 2.0f;

    // =====================================================================
    // ===== State Tracking =====
    // =====================================================================
    private float[] lastActions = new float[12];
    private float[] lastlastActions = new float[12];
    private Vector3 lastBaseVelocity;
    private float movementTimer = 0f;
    private float movementDuration = 120f;

    private Vector3 targetPos;
    private float stopRadius = 0.5f;
    private float stopRadiusTight = 0.1f;
    private float stopSpeedThreshold = 0.1f;
    private float lastDistToTarget;

    // =====================================================================
    // ===== Initialize =====
    // =====================================================================
    public override void Initialize() {
        joints = new ArticulationBody[12] {
            leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll,
            rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll
        };

        LoadTrajectoryData();

        float curriculumWeight = Academy.Instance.EnvironmentParameters.GetWithDefault("imitation_weight", -1.0f);
        if (curriculumWeight >= 0) {
            imitationWeight = curriculumWeight;
        } else {
            imitationWeight = manualImitationWeight;
        }
    }
    // =====================================================================
    // ===== Data Setting =====
    // =====================================================================
    private void LoadTrajectoryData() {
        try {
            standingSeq = LoadJsonTrajectory(standingSeqJsonPath);
            startLF = LoadJsonTrajectory(startLfJsonPath);
            startRF = LoadJsonTrajectory(startRfJsonPath);
            walkingLF = LoadJsonTrajectory(walkingLfJsonPath);
            walkingRF = LoadJsonTrajectory(walkingRfJsonPath);
            stopLF = LoadJsonTrajectory(stopLfJsonPath);
            stopRF = LoadJsonTrajectory(stopRfJsonPath);

            currentState = WalkingState.Standing;
            currentTraj = standingSeq;

        } catch (System.Exception e) {
            Debug.LogError($"Failed to load trajectory data: {e.Message}");
            useImitation = false;
        }
    }

    private TrajectoryData LoadJsonTrajectory(string path) {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string fullPath = Path.Combine(projectRoot, path);
        string jsonContent = File.ReadAllText(fullPath);
        return JsonConvert.DeserializeObject<TrajectoryData>(jsonContent);
    }

    // =====================================================================
    // ===== Episode Beigin =====
    // =====================================================================
    public override void OnEpisodeBegin()
    {
        // ===== Reset Initial Position =====
        Vector3 centerPos = Ground.position;
        centerPos.y = initialBaseHeight;
        baseLink.TeleportRoot(centerPos, Quaternion.identity);

        // ===== Reset Velocity =====
        baseLink.linearVelocity = Vector3.zero;
        baseLink.angularVelocity = Vector3.zero;

        // ===== Reset Action History =====
        for (int i = 0; i < 12; i++) {
            lastActions[i] = 0f;
            lastlastActions[i] = 0f;
        }

        // ===== Reset Pose =====
        for (int i = 0; i < joints.Length; i++) {
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

        // ===== Trajectory Initialization (Standing으로 시작) =====
        currentTraj = standingSeq;
        currentFrame = 0;
        currentState = WalkingState.Standing;
        trajectoryFrameTimer = 0f;
        standingTimer = 0f;
        standingCompleteTimer = 0f;
        movementTimer = 0f;

        // ===== Random Foot Selection (처음 출발할 발 랜덤 선택) =====
        startWithLeftFoot = (Random.value > 0.5f);

        // ===== Target =====

        float distance = Random.Range(3, 9);
        targetPos = centerPos + Vector3.forward * distance;
        target.position = new Vector3(targetPos.x, 0.15f, targetPos.z);
        lastDistToTarget = distance;

        lastBaseVelocity = Vector3.zero;
    }
    // =====================================================================
    // ===== Observations =====
    // =====================================================================
    public override void CollectObservations(VectorSensor sensor) {
        // ===== 1. Target Info [3 dims] =====
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        sensor.AddObservation(toTarget.normalized);

        // ===== 2. Base Orientation [2 dims] =====
        Vector3 euler = baseLink.transform.eulerAngles;
        sensor.AddObservation(NormalizeAngle(euler.z) / 180f);
        sensor.AddObservation(NormalizeAngle(euler.x) / 180f);

        // ===== 3. Current Joint Positions [12 dims] =====
        for (int i = 0; i < 12; i++) {
            float angle = joints[i].jointPosition[0] * Mathf.Rad2Deg;
            sensor.AddObservation(angle / 180f);
        }

        // ===== 4. Joint Velocities [12 dims] =====
        for (int i = 0; i < 12; i++) {
            float vel = joints[i].jointVelocity[0];
            sensor.AddObservation(vel * 0.1f);
        }

        // ===== 5. Last Actions [12 dims] =====
        for (int i = 0; i < 12; i++) {
            sensor.AddObservation(lastActions[i]);
        }

        // ===== 6. Base Angular Velocity [3 dims] =====
        Vector3 angVelBody = baseLink.transform.InverseTransformDirection(baseLink.angularVelocity);
        sensor.AddObservation(angVelBody * 0.25f);

        // ===== 7. Base Linear Velocity [3 dims] =====
        Vector3 linVelBody = baseLink.transform.InverseTransformDirection(baseLink.linearVelocity);
        sensor.AddObservation(linVelBody * 0.5f);

        // ===== 8. Phase [1 dim] =====
        float phase = 0f;
        if (currentTraj != null && currentTraj.q.Count > 0)
            phase = (float)currentFrame / (float)currentTraj.q.Count;
        sensor.AddObservation(phase);

        // ===== 9. Delta (Target - Current) [12 dims] =====
        // Current는 이미 관찰에 포함되어 있으므로 Delta만 추가
        if (useImitation && currentTraj != null && currentTraj.q != null && currentTraj.q.Count > 0) {
            int frameIdx = currentFrame % currentTraj.q.Count;
            List<float> refPose = currentTraj.q[frameIdx];
            List<float> unityPose = ConvertJsonToUnity(refPose);

            for (int i = 0; i < 12; i++) {
                float targetAngleRad = unityPose[7 + i];
                float targetAngleDeg = targetAngleRad * Mathf.Rad2Deg;
                float currentAngleDeg = joints[i].jointPosition[0] * Mathf.Rad2Deg;
                float delta = targetAngleDeg - currentAngleDeg;

                // Delta only (Target - Current)
                sensor.AddObservation(delta / 180f);
            }
        } else {
            // No trajectory data - use default angles as target
            for (int i = 0; i < 12; i++) {
                float currentAngleDeg = joints[i].jointPosition[0] * Mathf.Rad2Deg;
                float delta = defaultJointAngles[i] - currentAngleDeg;

                sensor.AddObservation(delta / 180f);
            }
        }
    }

    private float NormalizeAngle(float angle) {
        if (angle > 180f) angle -= 360f;
        return angle;
    }
    // =====================================================================
    // ===== Action Received =====
    // =====================================================================
    public override void OnActionReceived(ActionBuffers actions)
    {
        movementTimer += Time.fixedDeltaTime;

        // ===== Extract Actions =====
        var act = actions.ContinuousActions;
        float[] a = new float[12];
        for (int i = 0; i < 12; i++) a[i] = act[i];

        // ===== Apply Actions =====
        ApplyActionsWithImitation(a);

        // ===== Update State Machine =====
        UpdateStateMachine();

        // ===== Calculate Rewards =====
        CalculateRewards(a);

        // ===== Check Termination =====
        CheckTermination();

        // ===== Update History =====
        for (int i = 0; i < 12; i++)
        {
            lastlastActions[i] = lastActions[i];
            lastActions[i] = a[i];
        }
        lastBaseVelocity = baseLink.linearVelocity;

        // ===== Update Trajectory Frame =====
        if (currentState != WalkingState.StandingComplete)
        {
            trajectoryFrameTimer += Time.fixedDeltaTime;
            if (trajectoryFrameTimer >= trajectoryDt)
            {
                trajectoryFrameTimer -= trajectoryDt;
                currentFrame++;
            }
        }
    }

    private void ApplyActionsWithImitation(float[] policyActions) {
        for (int i = 0; i < 12; i++) {
            if (joints[i] == null) continue;
            float targetAngle = defaultJointAngles[i] + policyActions[i] * actionScale;
            targetAngle = Mathf.Clamp(targetAngle, jointMinAngles[i], jointMaxAngles[i]);
            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }
    }


    private List<float> ConvertJsonToUnity(List<float> jsonPose) {
        List<float> unityPose = new List<float>(jsonPose);
        float json_x = jsonPose[0];  // Front
        float json_y = jsonPose[1];  // Side
        float json_z = jsonPose[2];  // Up

        unityPose[0] = json_y; // Unity X <= Side
        unityPose[1] = json_z; // Unity Y <= Up
        unityPose[2] = json_x; // Unity Z <= Front
        
        // Quaternion (x,y,z,w) -> (y,z,x,w)
        float qx = jsonPose[3];
        float qy = jsonPose[4];
        float qz = jsonPose[5];
        float qw = jsonPose[6];
        
        unityPose[3] = qy; // Unity X
        unityPose[4] = qz; // Unity Y
        unityPose[5] = qx; // Unity Z
        unityPose[6] = qw;
        
        return unityPose;
    }

    private void UpdateStateMachine() {
        // [Sequence Fix] Standing -> Start -> Walking 순서 준수
        
        switch (currentState) {
            case WalkingState.Standing:
                standingTimer += Time.fixedDeltaTime;
                // 1초 후 출발 (안정화 시간 확보)
                if (standingTimer >= 1.0f) {
                    // 랜덤으로 선택된 발로 시작
                    if (startWithLeftFoot) {
                        SwitchState(WalkingState.StartWalkingLF, startLF);
                    } else {
                        SwitchState(WalkingState.StartWalkingRF, startRF);
                    }
                    standingTimer = 0f;
                }
                break;

            case WalkingState.StartWalkingLF:
                // 왼발로 시작 -> 오른발 사이클 반복
                if (currentFrame >= currentTraj.q.Count - 1) {
                    SwitchState(WalkingState.WalkingRF, walkingRF);
                }
                break;

            case WalkingState.StartWalkingRF:
                // 오른발로 시작 -> 왼발 사이클 반복
                if (currentFrame >= currentTraj.q.Count - 1) {
                    SwitchState(WalkingState.WalkingLF, walkingLF);
                }
                break;

            case WalkingState.WalkingLF:
                // WalkingLF 사이클 반복 (자기 자신)
                if (currentFrame >= currentTraj.q.Count - 1) {
                    SwitchState(WalkingState.WalkingLF, walkingLF);
                }
                break;

            case WalkingState.WalkingRF:
                // WalkingRF 사이클 반복 (자기 자신)
                if (currentFrame >= currentTraj.q.Count - 1) {
                    SwitchState(WalkingState.WalkingRF, walkingRF);
                }
                break;
                
            default:
                // 그 외 상태면 그냥 Standing으로
                SwitchState(WalkingState.Standing, standingSeq);
                break;
        }
    }

    private void SwitchState(WalkingState newState, TrajectoryData newTraj) {
        currentState = newState;
        currentTraj = newTraj;
        currentFrame = 0;
        trajectoryFrameTimer = 0f;
    }

    private void CalculateRewards(float[] currentActions) {
        float totalReward = 0f;
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        float distToTarget = toTarget.magnitude;
        bool nearTarget = distToTarget < stopRadius;

        if (!nearTarget) {
            float progressReward = (lastDistToTarget - distToTarget) / Time.fixedDeltaTime;
            totalReward += progressReward * scale_forward_progress;
            totalReward += RewardTrajectoryDirection() * scale_trajectory_direction;
        }

        if (useImitation && currentTraj != null && imitationWeight > 0f) {
            totalReward += RewardImitationPose() * scale_imitation_pose * imitationWeight;
            totalReward += RewardImitationVelocity() * scale_imitation_velocity * imitationWeight;
            totalReward += RewardCoMTracking() * scale_com_tracking * imitationWeight;
        }

        totalReward += RewardOrientation() * scale_orientation;
        totalReward += RewardBaseHeight() * scale_base_height;
        totalReward += RewardActionSmoothness(currentActions) * scale_action_smoothness;

        if (nearTarget) {
            float speed = baseLink.linearVelocity.magnitude;
            float stopReward = Mathf.Exp(-speed * 10f);
            totalReward += stopReward * scale_stop_velocity;
        }

        AddReward(totalReward);
        lastDistToTarget = distToTarget;
    }


    private float RewardImitationPose() {
        if (currentTraj == null || currentTraj.q == null || currentTraj.q.Count == 0) return 0f;
        int frameIdx = currentFrame % currentTraj.q.Count;
        List<float> refPose = currentTraj.q[frameIdx];
        List<float> unityPose = ConvertJsonToUnity(refPose);

        float poseError = 0f;
        for (int i = 0; i < 12; i++) {
            float currentQ = joints[i].jointPosition[0];
            float refQ = unityPose[7 + i];
            float diff = currentQ - refQ;
            poseError += diff * diff;
        }
        return Mathf.Exp(-poseError * 2f); 
    }

    private float RewardImitationVelocity() {
        if (currentTraj == null || currentTraj.v == null || currentTraj.v.Count == 0) return 0f;
        int frameIdx = currentFrame % currentTraj.v.Count;
        List<float> refVel = currentTraj.v[frameIdx];

        float velError = 0f;
        for (int i = 0; i < 12; i++) {
            float currentV = joints[i].jointVelocity[0];
            float refV = refVel[6 + i];
            float diff = currentV - refV;
            velError += diff * diff;
        }
        return Mathf.Exp(-velError * 0.2f); 
    }

    private float RewardCoMTracking() {
        if (currentTraj == null || currentTraj.q == null || currentTraj.q.Count == 0) return 0f;
        int frameIdx = currentFrame % currentTraj.q.Count;
        List<float> refPose = currentTraj.q[frameIdx];
        List<float> unityPose = ConvertJsonToUnity(refPose);

        float refX = unityPose[0];
        float refZ = unityPose[2];
        Vector3 currentPos = baseLink.transform.position;
        float errorX = refX - currentPos.x;
        float errorZ = refZ - currentPos.z;

        float lateralError = errorX * errorX * 2f;
        float forwardError = errorZ * errorZ;
        
        return Mathf.Exp(-(lateralError + forwardError) * 1f); 
    }

    private float RewardOrientation() {
        Vector3 euler = baseLink.transform.eulerAngles;
        float roll = NormalizeAngle(euler.z);
        float pitch = NormalizeAngle(euler.x);
        float orientError = (roll * roll + pitch * pitch) / (90f * 90f);
        return Mathf.Exp(-orientError * 5f);
    }

    private float RewardTrajectoryDirection() {
        // 로봇의 forward 방향과 타겟 방향의 일치도 측정
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;

        if (toTarget.magnitude < 0.1f) return 1f; // 타겟에 매우 가까우면 최대 보상

        Vector3 targetDir = toTarget.normalized;
        Vector3 robotForward = baseLink.transform.forward;
        robotForward.y = 0;
        robotForward.Normalize();

        // Dot product: 1 (완전히 일치) ~ -1 (완전히 반대)
        float alignment = Vector3.Dot(robotForward, targetDir);

        // Exponential reward: 방향이 일치할수록 높은 보상
        return Mathf.Exp(-(1f - alignment) * 3f);
    }

    private float RewardBaseHeight() {
        float heightError = Mathf.Abs(baseLink.transform.position.y - initialBaseHeight);
        return Mathf.Exp(-heightError * 10f);
    }

    private float RewardActionSmoothness(float[] currentActions) {
        float smoothness = 0f;
        for (int i = 0; i < 12; i++) {
            float diff = currentActions[i] - lastActions[i];
            smoothness += diff * diff;
        }
        return Mathf.Exp(-smoothness * 5f);
    }

    private void CheckTermination() {
        if(StepCount < graceSteps)
            return;
        float baseHeight = baseLink.transform.position.y;
        if (baseHeight < 0.3f) {
            AddReward(-1f); // 패널티 감소
            EndEpisode();
            return;
        }

        Vector3 euler = baseLink.transform.eulerAngles;
        float roll = NormalizeAngle(euler.z);
        float pitch = NormalizeAngle(euler.x);
        if (Mathf.Abs(roll) > 60f || Mathf.Abs(pitch) > 60f) {
            AddReward(-1f);
            EndEpisode();
            return;
        }

        if (useImitation && currentTraj != null && currentTraj.q != null) {
            float poseError = CalculateTrajectoryDistance();
            // [수정 3 적용] Threshold 사용
            if (poseError > earlyTerminationThreshold) {
                AddReward(-1f);
                EndEpisode();
                return;
            }
        }

        if (movementTimer > movementDuration) {
            EndEpisode();
            return;
        }
    }

    private float CalculateTrajectoryDistance() {
        if (currentTraj == null || currentTraj.q == null || currentTraj.q.Count == 0) return 0f;
        int frameIdx = currentFrame % currentTraj.q.Count;
        List<float> refPose = currentTraj.q[frameIdx];
        List<float> unityPose = ConvertJsonToUnity(refPose);

        float poseError = 0f;
        for (int i = 0; i < 12; i++) {
            float currentQ = joints[i].jointPosition[0];
            float refQ = unityPose[7 + i];
            float diff = currentQ - refQ;
            poseError += diff * diff;
        }
        return Mathf.Sqrt(poseError);
    }

    public void HandleGroundCollision() {
        if (StepCount < graceSteps) return;
        AddReward(-1f);
        EndEpisode();
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        var cont = actionsOut.ContinuousActions;
        for (int i = 0; i < 12; i++) cont[i] = 0f;
    }
}