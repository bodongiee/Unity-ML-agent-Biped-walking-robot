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
    public string walkingLfJsonPath = "Assets/IL/Actions/trajectory_extract_cycle_lf_start_98.json";
    public string walkingRfJsonPath = "Assets/IL/Actions/trajectory_extract_cycle_rf_start_98.json";
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

    public int graceSteps = 200;  
    public float earlyTerminationThreshold = 50.0f;  // 궤적 오차 허용치 증가
    // =====================================================================
    // ===== Configuration =====
    // =====================================================================
    [Header("Control Parameters")]
    public float actionScale = 15f;

    public bool playbackMode = false;
    
    private float[] jointMinAngles = new float[12] { -45f, -30f, -45f, -100f, -45f, -30f, -45f, -30f, -45f, -100f, -45f, -30f };
    private float[] jointMaxAngles = new float[12] { 45f, 30f, 100f, 0f, 45f, 30f, 45f, 30f, 100f, 0f, 45f, 30f };

    private float initialBaseHeight = 0.67f;

    [Header("Reward Weights")]
    public float scale_imitation_pose = 6.0f;        // Learn walking pattern
    public float scale_imitation_velocity = 1.0f;    // Learn movement dynamics
    public float scale_com_tracking = 0.0f;          // DISABLED
    public float scale_yaw = 15.0f;                  // Direction control
    public float scale_orientation = 2.0f;           // Keep upright
    public float scale_base_height = 1.0f;
    public float scale_action_smoothness = 0.3f;
    public float scale_forward_progress = 0.0f;      // REMOVED
    public float scale_stop_velocity = 2.0f;
    public float scale_trajectory_direction = 0.0f;  // REMOVED

    // =====================================================================
    // ===== State Tracking =====
    // =====================================================================
    private float[] lastActions = new float[12];
    private float[] lastlastActions = new float[12];
    private Vector3 lastBaseVelocity;
    private float movementTimer = 0f;
    private float movementDuration = 480f;

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
    // ===== Start: ML-Agent 없이 실행할 때 초기화 =====
    // =====================================================================
    private void Start() {

        if (joints == null || joints.Length == 0) {
            joints = new ArticulationBody[12] {
                leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll,
                rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll
            };
        }

        // 궤적 데이터 로드
        if (currentTraj == null) {
            LoadTrajectoryData();
        }

        // 초기화 수행
        OnEpisodeBegin();
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

        // ===== Reset Pose) =====
        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;

            // 관절 위치와 속도 초기화
            joints[i].jointPosition = new ArticulationReducedSpace(defaultJointAngles[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);

            var dr = joints[i].xDrive;
            dr.driveType = ArticulationDriveType.Force;
            dr.stiffness = 10000f;
            dr.damping = 100f;
            dr.forceLimit = 1000f;
            dr.target = defaultJointAngles[i];
            joints[i].xDrive = dr;

            // 디버그: 관절 설정 확인
            //if (debugMode && i == 3) {
            //    Debug.Log($"<color=yellow>Joint[3] Setup: JointType={joints[i].jointType}, " +
            //              $"TwistLock={joints[i].twistLock}, DriveType={dr.driveType}, " +
            //              $"Stiffness={dr.stiffness}, ForceLimit={dr.forceLimit}</color>");
            //}
        }

        Physics.SyncTransforms();

        // ===== 모든 관절 속도 다시 0으로 설정 (물리 동기화 후) =====
        baseLink.linearVelocity = Vector3.zero;
        baseLink.angularVelocity = Vector3.zero;
        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] != null) {
                joints[i].jointVelocity = new ArticulationReducedSpace(0f);
            }
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
        startWithLeftFoot = Random.value > 0.5f;

        // ===== Target =====
        float distance = Random.Range(3f, 9f);
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
        Vector3 toTarget = target.position - baseLink.transform.position;
        toTarget.y = 0;
        // CRITICAL: Convert to robot's LOCAL coordinate frame!
        // This makes learning much easier - "forward" is always (0,0,1) regardless of robot's world orientation
        Vector3 localTarget = baseLink.transform.InverseTransformDirection(toTarget.normalized);
        sensor.AddObservation(localTarget);

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
        if (playbackMode) return;
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
        Vector3 toTarget = target.position - baseLink.transform.position;  // Use target.position for consistency
        toTarget.y = 0;
        float distToTarget = toTarget.magnitude;
        bool nearTarget = distToTarget < stopRadius;

        if (!nearTarget) {
            // Progress reward with clamping to prevent domination
            float progressReward = (lastDistToTarget - distToTarget) / Time.fixedDeltaTime;
            progressReward = Mathf.Clamp(progressReward, -2f, 2f);  // Clamp to reasonable range
            totalReward += progressReward * scale_forward_progress;
            totalReward += RewardYaw() * scale_yaw;  // Use angle-based Yaw reward instead
        }

        if (useImitation && currentTraj != null && imitationWeight > 0f) {
            totalReward += RewardImitationPose() * scale_imitation_pose * imitationWeight;
            totalReward += RewardImitationVelocity() * scale_imitation_velocity * imitationWeight;
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
        Vector3 toTarget = target.position - baseLink.transform.position;
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

    private float RewardYaw() {
        Vector3 toTarget = target.position - baseLink.transform.position;
        toTarget.y = 0;

        if (toTarget.magnitude < 0.1f) return 1f;

        // 목표 Yaw 각도 계산
        float desiredYaw = Mathf.Atan2(toTarget.x, toTarget.z) * Mathf.Rad2Deg;
        float currentYaw = baseLink.transform.eulerAngles.y;

        // 각도 차이 계산 (-180 ~ 180)
        float yawError = Mathf.Abs(Mathf.DeltaAngle(currentYaw, desiredYaw));

        // 각도 오차에 따른 지수 감쇠 보상 (0.05 계수로 부드러운 그라디언트)
        return Mathf.Exp(-yawError * 0.05f);
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
        bool isInitialState = (currentState == WalkingState.Standing ||
                               currentState == WalkingState.StartWalkingLF ||
                               currentState == WalkingState.StartWalkingRF);

        if (isInitialState) {
            return;  // 초기 상태에서는 절대 종료 안함
        }

        // Grace period 체크
        if(StepCount < graceSteps)
            return;

        // 높이 체크
        float baseHeight = baseLink.transform.position.y;
        if (baseHeight < 0.3f) {
            if (debugMode) Debug.LogWarning($"Termination: Height={baseHeight:F2}");
            AddReward(-1f);
            EndEpisode();
            return;
        }

        // 각도 체크
        Vector3 euler = baseLink.transform.eulerAngles;
        float roll = NormalizeAngle(euler.z);
        float pitch = NormalizeAngle(euler.x);
        if (Mathf.Abs(roll) > 60f || Mathf.Abs(pitch) > 60f) {
            if (debugMode) Debug.LogWarning($"Termination: Roll={roll:F1}°, Pitch={pitch:F1}°");
            AddReward(-1f);
            EndEpisode();
            return;
        }

        // 궤적 오차 체크
        if (useImitation && currentTraj != null && currentTraj.q != null) {
            float poseError = CalculateTrajectoryDistance();
            if (poseError > earlyTerminationThreshold) {
                if (debugMode) Debug.LogWarning($"Termination: PoseError={poseError:F2}");
                AddReward(-1f);
                EndEpisode();
                return;
            }
        }

        // 시간 초과
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
        cont[3] = Input.GetKey(KeyCode.W) ? 1f : (Input.GetKey(KeyCode.S) ? -1f : 0f);
        cont[4] = Input.GetKey(KeyCode.Q) ? 1f : (Input.GetKey(KeyCode.A) ? -1f : 0f);
        cont[9] = Input.GetKey(KeyCode.I) ? 1f : (Input.GetKey(KeyCode.K) ? -1f : 0f);
        cont[10] = Input.GetKey(KeyCode.U) ? 1f : (Input.GetKey(KeyCode.J) ? -1f : 0f);
    }

    // ===== Debug Visualization =====
    private void Update() {
        if (baseLink == null || target == null) return;

        Vector3 basePos = baseLink.transform.position;

        // 1. 로봇의 forward 방향 (파란색 화살표)
        Debug.DrawRay(basePos, baseLink.transform.forward * 3f, Color.blue);

        // 2. targetPos로의 방향 (빨간색 화살표 - 현재 리워드에서 사용 중)
        Vector3 toTargetPos = targetPos - basePos;
        toTargetPos.y = 0;
        // targetPos가 Vector3.zero가 아니고 충분히 멀면 그리기
        if (targetPos != Vector3.zero && toTargetPos.magnitude > 0.1f) {
            Debug.DrawRay(basePos, toTargetPos.normalized * 3f, Color.red);
            Debug.DrawRay(basePos, toTargetPos, Color.red); // 실제 거리까지도 표시
        }

        // 3. target.position으로의 방향 (초록색 화살표 - 실제 Transform)
        Vector3 toTargetTransform = target.position - basePos;
        toTargetTransform.y = 0;
        if (toTargetTransform.magnitude > 0.1f) {
            Debug.DrawRay(basePos, toTargetTransform.normalized * 3.5f, Color.green);
        }

        // 4. 로봇에서 실제 target까지 직선 (노란색)
        Debug.DrawLine(basePos, target.position, Color.yellow);

        // 5. 타겟 위치 표시 (마젠타 십자가)
        Vector3 targetWorldPos = target.position;
        Debug.DrawLine(targetWorldPos + Vector3.left * 0.3f, targetWorldPos + Vector3.right * 0.3f, Color.magenta);
        Debug.DrawLine(targetWorldPos + Vector3.forward * 0.3f, targetWorldPos + Vector3.back * 0.3f, Color.magenta);

        // 6. 디버그 정보 출력 (100프레임마다)
        if (debugMode && Time.frameCount % 100 == 0) {
            float distToTargetPos = toTargetPos.magnitude;
            float distToTargetTransform = toTargetTransform.magnitude;
            float positionDiff = Vector3.Distance(targetPos, target.position);

            Vector3 robotForward = baseLink.transform.forward;
            robotForward.y = 0;
            robotForward.Normalize();

            float angleToTarget = Vector3.SignedAngle(robotForward, toTargetTransform.normalized, Vector3.up);

            Debug.Log($"<color=cyan>[Target Debug]</color> " +
                      $"Robot={basePos:F2}, " +
                      $"targetPos={targetPos:F2}, " +
                      $"target.position={target.position:F2}\n" +
                      $"Dist(targetPos)={distToTargetPos:F2}m, " +
                      $"Dist(target.position)={distToTargetTransform:F2}m, " +
                      $"Diff={positionDiff:F3}m, " +
                      $"Angle={angleToTarget:F1}°");
        }
    }

    // ===== FixedUpdate: ML-Agent 없이 실행할 때 궤적 자동 재생 =====
    private void FixedUpdate() {

        if(!playbackMode) return;

        // ML-Agent 없이 실행 중 - 궤적을 직접 따라감
        if (debugMode && Time.frameCount % 100 == 0) {
            Debug.Log($"FixedUpdate: State={currentState}, Frame={currentFrame}, StandingTimer={standingTimer:F2}");
        }

        UpdateStateMachine();

        // 궤적 프레임 업데이트
        if (currentState != WalkingState.StandingComplete) {
            trajectoryFrameTimer += Time.fixedDeltaTime;
            if (trajectoryFrameTimer >= trajectoryDt) {
                trajectoryFrameTimer -= trajectoryDt;
                currentFrame++;
            }
        }

        // 궤적의 타겟 각도를 직접 적용
        if (currentTraj != null && currentTraj.q != null && currentTraj.q.Count > 0) {
            int frameIdx = currentFrame % currentTraj.q.Count;
            List<float> refPose = currentTraj.q[frameIdx];
            List<float> unityPose = ConvertJsonToUnity(refPose);

            for (int i = 0; i < 12; i++) {
                if (joints[i] == null) continue;
                float targetAngleRad = unityPose[7 + i];
                float targetAngleDeg = targetAngleRad * Mathf.Rad2Deg;
                targetAngleDeg = Mathf.Clamp(targetAngleDeg, jointMinAngles[i], jointMaxAngles[i]);

                var drive = joints[i].xDrive;
                drive.target = targetAngleDeg;
                joints[i].xDrive = drive;

                // 디버그: 왼쪽 무릎 (관절 3번) 확인
                if (debugMode && i == 3 && Time.frameCount % 100 == 0) {
                    float currentAngleDeg = joints[i].jointPosition[0] * Mathf.Rad2Deg;
                    Debug.Log($"LKnee[3]: Current={currentAngleDeg:F1}°, Target={targetAngleDeg:F1}°, " +
                              $"Traj={targetAngleRad:F3}rad, Stiffness={drive.stiffness}");
                }
            }
        } else if (debugMode && Time.frameCount % 100 == 0) {
            Debug.LogWarning("FixedUpdate: currentTraj is null or empty!");
        }
    }
}