using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

[System.Serializable]
// =====================================================================
// ====== Trajectory Data =====
// =====================================================================
public class TrajectoryData {
    public List<List<float>> q;    // 관절 위치
    public List<List<float>> v;    // 관절 속도
    public List<List<float>> tau;  // 토크
    public float dt;
}
// 행동 상태
public enum WalkingState { Standing, StartWalkingLF, StartWalkingRF, WalkingLF, WalkingRF, StoppingLF, StoppingRF, StandingComplete}

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

    // =====================================================================
    // ===== Foot Contact =====
    // =====================================================================
    public FootContact leftFoot;
    public FootContact rightFoot;

    // =====================================================================
    // ===== Target =====
    // =====================================================================
    public Transform target;
    public Transform Ground;

    // =====================================================================
    // ===== Joint Array =====
    // =====================================================================
    private ArticulationBody[] joints;
    private float[] defaultJointAngles = new float[12] {0f, 0f, 30f, -60f, 30f, 0f, 0f, 0f, 30f, -60f, 30f, 0f};
    //Left Hip (Yaw, Roll, Pitch), Left Knee (Pitch), Left Ankle (Pitch, Roll), Right

    // =====================================================================
    // ===== Imitation Learning Data =====
    // =====================================================================
    [Header("Imitation Learning Trajectories")]
    public string standingSeqJsonPath = "Assets/IL/Actions/standing.json";
    public string startLfJsonPath = "Assets/IL/Actions/trajectory_start_rf_support_step.json";
    public string startRfJsonPath = "Assets/IL/Actions/trajectory_start_lf_support_step.json";
    public string walkingLfJsonPath = "Assets/IL/Actions/trajectory_extract_cycle_lf_start.json";
    public string walkingRfJsonPath = "Assets/IL/Actions/trajectory_extract_cycle_rf_start.json";
    public string stopLfJsonPath = "Assets/IL/Actions/trajectory_stop_rf_support_step.json";
    public string stopRfJsonPath = "Assets/IL/Actions/trajectory_stop_lf_support_step.json";

    private TrajectoryData standingSeq;
    private TrajectoryData startLF;
    private TrajectoryData startRF;
    private TrajectoryData walkingLF;
    private TrajectoryData walkingRF;
    private TrajectoryData stopLF;
    private TrajectoryData stopRF;

    // 현재 상태
    private WalkingState currentState = WalkingState.Standing;
    private TrajectoryData currentTraj;
    private int currentFrame = 0;
    private float trajectoryFrameTimer = 0f;
    private float trajectoryDt = 0.02f;  // Unity FixedUpdate와 맞춤
    private float standingTimer = 0f;    // Standing 상태 유지 시간
    private float minStandingTime = 1.0f;  // 최소 1초 대기


    [Header("Imitation Learning Settings")]
    public bool useImitation = true;  // 모방학습 활성화
    public float manualImitationWeight = 1.0f;  // Inspector에서 수동 조정 가능
    private float imitationWeight = 1.0f;
    public bool debugMode = true;  // 디버그 로그 활성화

    // =====================================================================
    // ===== Configuration =====
    // =====================================================================
    [Header("Control Parameters")]
    public float actionScale = 60f;
    private float[] jointMinAngles = new float[12] {-45f, -30f, -45f, -100f, -45f, -30f, -45f, -30f, -45f, -100f, -45f, -30f};
    private float[] jointMaxAngles = new float[12] {45f, 30f, 100f, 0f, 45f, 30f, 45f, 30f, 100f, 0f, 45f, 30f};

    private float initialBaseHeight = 0.67f;

    [Header("Reward Weights")]
    public float scale_imitation_pose = 2.0f;
    public float scale_imitation_velocity = 0.5f;
    public float scale_orientation = 0.5f;
    public float scale_base_height = 0.5f;
    public float scale_action_smoothness = 0.1f;
    public float scale_forward_progress = 1.0f;
    public float scale_stop_velocity = 2.0f;

    // =====================================================================
    // ===== State Tracking =====
    // =====================================================================
    private float[] lastActions = new float[12];
    private float[] lastlastActions = new float[12];
    private Vector3 lastBaseVelocity;
    private float movementTimer = 0f;
    private float movementDuration = 40f;

    private Vector3 targetPos;
    private float stopRadius = 1.0f;           // 정지 시작 거리
    private float stopRadiusTight = 0.1f;      // 성공 판정 거리
    private float stopSpeedThreshold = 0.1f;   // 성공 판정 속도

    // =====================================================================
    // ===== Initialization =====
    // =====================================================================
    public override void Initialize() {
        joints = new ArticulationBody[12] {
            leftHipYaw, leftHipRoll, leftHipPitch, leftKneePitch, leftAnklePitch, leftAnkleRoll,
            rightHipYaw, rightHipRoll, rightHipPitch, rightKneePitch, rightAnklePitch, rightAnkleRoll
        };

        LoadTrajectoryData();

        // Curriculum learning에서 imitation_weight 가져오기 (학습 시)
        // Inspector의 manualImitationWeight 사용 (테스트 시)
        float curriculumWeight = Academy.Instance.EnvironmentParameters.GetWithDefault("imitation_weight", -1.0f);
        if (curriculumWeight >= 0) {
            imitationWeight = curriculumWeight;
            Debug.Log($"Using curriculum imitation weight: {imitationWeight}");
        } else {
            imitationWeight = manualImitationWeight;
            Debug.Log($"Using manual imitation weight: {imitationWeight}");
        }
    }

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
        }
        catch (System.Exception e) {
            Debug.LogError($"Failed to load trajectory data: {e.Message}");
            useImitation = false;
        }
    }

    private TrajectoryData LoadJsonTrajectory(string path) {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string fullPath = Path.Combine(projectRoot, path);
        Debug.Log($"Loading trajectory from: {fullPath}");
        string jsonContent = File.ReadAllText(fullPath);
        return JsonConvert.DeserializeObject<TrajectoryData>(jsonContent);
    }

    // =====================================================================
    // ===== Episode Begin =====
    // =====================================================================
    public override void OnEpisodeBegin() {
        // Reset base position
        Vector3 centerPos = Ground.position;
        centerPos.y = initialBaseHeight;
        baseLink.TeleportRoot(centerPos, Quaternion.identity);

        // Reset velocities
        baseLink.linearVelocity = Vector3.zero;
        baseLink.angularVelocity = Vector3.zero;
        lastBaseVelocity = Vector3.zero;

        // Reset joints to trajectory's first frame (if using imitation)
        if (useImitation && standingSeq != null && standingSeq.q != null && standingSeq.q.Count > 0) {
            List<float> refPose = standingSeq.q[0];
            List<float> unityPose = ConvertJsonToUnity(refPose);

            if (debugMode) {
                Debug.Log("=== Initializing with trajectory first frame ===");
            }
            for (int i = 0; i < joints.Length && i < 12; i++) {
                if (joints[i] == null) continue;

                // 궤적의 첫 프레임 관절 각도
                float trajAngleRad = unityPose[7 + i];
                float trajAngleDeg = trajAngleRad * Mathf.Rad2Deg;

                joints[i].jointPosition = new ArticulationReducedSpace(trajAngleRad);
                joints[i].jointVelocity = new ArticulationReducedSpace(0f);

                var dr = joints[i].xDrive;
                dr.stiffness = 10000f;
                dr.damping = 100f;
                dr.forceLimit = 1000f;
                dr.target = trajAngleDeg;
                joints[i].xDrive = dr;
            }
        } else {
            // 모방학습 비활성화 시 기본 자세
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

                if (debugMode) {
                    Debug.Log($"Joint {i}: defaultAngle={defaultJointAngles[i]:F2}°");
                }
            }
        }

        // Reset actions
        for (int i = 0; i < 12; i++) {
            lastActions[i] = 0f;
            lastlastActions[i] = 0f;
        }

        // Reset target
        float dist = 5f;
        targetPos = centerPos + new Vector3(0, 0, dist);
        target.position = new Vector3(targetPos.x, 0.15f, targetPos.z);

        // Reset state machine
        currentState = WalkingState.Standing;
        currentTraj = standingSeq;
        currentFrame = 0;
        trajectoryFrameTimer = 0f;
        standingTimer = 0f;  // Standing 타이머 리셋

        movementTimer = 0f;
        lastDistToTarget = dist;

        // Update imitation weight
        float curriculumWeight = Academy.Instance.EnvironmentParameters.GetWithDefault("imitation_weight", -1.0f);
        if (curriculumWeight >= 0) {
            imitationWeight = curriculumWeight;
        } else {
            imitationWeight = manualImitationWeight;
        }
    }

    // =====================================================================
    // ===== Observations =====
    // =====================================================================
    public override void CollectObservations(VectorSensor sensor) {
        // 1. Target Info [3]
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        sensor.AddObservation(toTarget.normalized);

        // 2. Base Orientation [2]
        Vector3 euler = baseLink.transform.eulerAngles;
        sensor.AddObservation(NormalizeAngle(euler.z) / 180f);  // Roll
        sensor.AddObservation(NormalizeAngle(euler.x) / 180f);  // Pitch

        // 3. Joint Positions [12]
        for (int i = 0; i < 12; i++) {
            float angle = joints[i].jointPosition[0] * Mathf.Rad2Deg;
            sensor.AddObservation(angle / 180f);
        }

        // 4. Joint Velocities [12]
        for (int i = 0; i < 12; i++) {
            float vel = joints[i].jointVelocity[0];
            sensor.AddObservation(vel * 0.1f);
        }

        // 5. Last Actions [12]
        for (int i = 0; i < 12; i++) {
            sensor.AddObservation(lastActions[i]);
        }

        // 6. Base Angular Velocity [3]
        Vector3 angVelBody = baseLink.transform.InverseTransformDirection(baseLink.angularVelocity);
        sensor.AddObservation(angVelBody * 0.25f);

        // 7. Base Linear Velocity [3]
        Vector3 linVelBody = baseLink.transform.InverseTransformDirection(baseLink.linearVelocity);
        sensor.AddObservation(linVelBody * 0.5f);
    }

    private float NormalizeAngle(float angle) {
        if (angle > 180f) angle -= 360f;
        return angle;
    }

    // =====================================================================
    // ===== Action Received =====
    // =====================================================================
    public override void OnActionReceived(ActionBuffers actions) {
        float dt = Time.fixedDeltaTime;
        movementTimer += dt;

        // Apply actions with imitation learning
        var continuousActions = actions.ContinuousActions;
        float[] currentActions = new float[12];

        for (int i = 0; i < 12; i++) {
            currentActions[i] = continuousActions[i];
        }

        // Blend policy actions with trajectory
        ApplyActionsWithImitation(currentActions);

        // Update state machine
        UpdateStateMachine();

        // Calculate rewards
        CalculateRewards(currentActions);

        // Check termination
        CheckTermination();

        // Update last actions
        for (int i = 0; i < 12; i++) {
            lastlastActions[i] = lastActions[i];
            lastActions[i] = currentActions[i];
        }
        lastBaseVelocity = baseLink.linearVelocity;

        // Update trajectory frame
        trajectoryFrameTimer += dt;
        if (trajectoryFrameTimer >= trajectoryDt) {
            trajectoryFrameTimer -= trajectoryDt;
            currentFrame++;
        }
    }

    private void ApplyActionsWithImitation(float[] policyActions) {
        // ===== No Use of Imitation Learning =====
        if (!useImitation || currentTraj == null || currentTraj.q == null || currentTraj.q.Count == 0) {
            for (int i = 0; i < 12; i++) {
                if (joints[i] == null) continue;
                float targetAngle = defaultJointAngles[i] + policyActions[i] * actionScale;
                targetAngle = Mathf.Clamp(targetAngle, jointMinAngles[i], jointMaxAngles[i]);
                var drive = joints[i].xDrive;
                drive.target = targetAngle;
                joints[i].xDrive = drive;
            }
            return;
        }
        // ===== Apply Imitation Learning =====
        // Trajectory and Policy Blending
        int frameIdx = currentFrame % currentTraj.q.Count;
        List<float> refPose = currentTraj.q[frameIdx];

        // 좌표계 변환 적용
        List<float> unityPose = ConvertJsonToUnity(refPose);

        for (int i = 0; i < 12; i++) {
            if (joints[i] == null) continue;
            // 궤적 타겟 (도 단위)
            float trajTarget = unityPose[7 + i] * Mathf.Rad2Deg;
            float targetAngle;

            if (imitationWeight >= 0.99f) {
                targetAngle = trajTarget;
            }
            else {
                float policyTarget = defaultJointAngles[i] + policyActions[i] * actionScale;
                targetAngle = Mathf.Lerp(policyTarget, trajTarget, imitationWeight);
            }

            targetAngle = Mathf.Clamp(targetAngle, jointMinAngles[i], jointMaxAngles[i]);

            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }
    }

    // =====================================================================
    // ===== Coordinate Conversion =====
    // =====================================================================
    // JSON: x=전진, y=좌우, z=위
    // Unity: x=좌우, y=위, z=전진
    private List<float> ConvertJsonToUnity(List<float> jsonPose) {
        List<float> unityPose = new List<float>(jsonPose);

        // FreeFlyer 위치 변환 [0:3]
        float json_x = jsonPose[0];  // 전진
        float json_y = jsonPose[1];  // 좌우
        float json_z = jsonPose[2];  // 위

        unityPose[0] = json_y;  // Unity x = JSON y (좌우)
        unityPose[1] = json_z;  // Unity y = JSON z (위)
        unityPose[2] = json_x;  // Unity z = JSON x (전진)
        return unityPose;
    }

    // =====================================================================
    // ===== State Machine =====
    // =====================================================================
    private void UpdateStateMachine() {
        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        float distToTarget = toTarget.magnitude;
        float speed = baseLink.linearVelocity.magnitude;

        // 발 위치 확인 (어느 발이 앞인지)
        bool leftFootForward = leftFoot.transform.position.z > rightFoot.transform.position.z;

        switch (currentState) {
            case WalkingState.Standing:
                // Standing → Start Walking (최소 대기 시간 후)
                standingTimer += Time.fixedDeltaTime;
                if (distToTarget > stopRadius && standingTimer >= minStandingTime) {
                    // 왼발로 시작 (기본)
                    SwitchState(WalkingState.StartWalkingLF, startLF);
                    standingTimer = 0f;
                }
                break;

            case WalkingState.StartWalkingLF:
                // Start → Walking (궤적 끝까지 재생)
                if (currentFrame >= currentTraj.q.Count - 1) {
                    SwitchState(WalkingState.WalkingLF, walkingLF);
                }
                break;

            case WalkingState.WalkingLF:
                // Walking → Stop or Continue
                if (distToTarget < stopRadius) {
                    // 왼발 앞이면 왼발로 정지, 오른발 앞이면 오른발로 정지
                    if (leftFootForward) {
                        SwitchState(WalkingState.StoppingLF, stopLF);
                    } else {
                        SwitchState(WalkingState.StoppingRF, stopRF);
                    }
                }
                // 계속 걷기 (궤적 반복)
                break;

            case WalkingState.StoppingLF:
            case WalkingState.StoppingRF:
                // Stop → Standing Complete
                if (currentFrame >= currentTraj.q.Count - 1) {
                    SwitchState(WalkingState.StandingComplete, standingSeq);
                }
                break;

            case WalkingState.StandingComplete:
                // 성공 판정
                if (distToTarget < stopRadiusTight && speed < stopSpeedThreshold) {
                    float timeBonus = Mathf.Clamp01(1f - movementTimer / movementDuration);
                    AddReward(30f + timeBonus * 20f);
                    EndEpisode();
                }
                break;
        }
    }

    private void SwitchState(WalkingState newState, TrajectoryData newTraj) {
        currentState = newState;
        currentTraj = newTraj;
        currentFrame = 0;
        trajectoryFrameTimer = 0f;
    }

    // =====================================================================
    // ===== Rewards =====
    // =====================================================================
    private float lastDistToTarget;

    private void CalculateRewards(float[] currentActions) {
        float totalReward = 0f;

        Vector3 toTarget = targetPos - baseLink.transform.position;
        toTarget.y = 0;
        float distToTarget = toTarget.magnitude;
        bool nearTarget = distToTarget < stopRadius;

        // 1. Forward progress (타겟으로 이동)
        if (!nearTarget) {
            float progressReward = (lastDistToTarget - distToTarget) / Time.fixedDeltaTime;
            totalReward += progressReward * scale_forward_progress;
        }

        // 2. Imitation rewards (useImitation이 true이고 궤적이 있을 때만)
        if (useImitation && currentTraj != null && imitationWeight > 0f) {
            totalReward += RewardImitationPose() * scale_imitation_pose * imitationWeight;
            totalReward += RewardImitationVelocity() * scale_imitation_velocity * imitationWeight;
        }

        // 3. Stability rewards
        totalReward += RewardOrientation() * scale_orientation;
        totalReward += RewardBaseHeight() * scale_base_height;
        totalReward += RewardActionSmoothness(currentActions) * scale_action_smoothness;

        // 4. Stop rewards (타겟 근처에서)
        if (nearTarget) {
            float speed = baseLink.linearVelocity.magnitude;
            float stopReward = Mathf.Exp(-speed * 10f);
            totalReward += stopReward * scale_stop_velocity;
        }

        AddReward(totalReward);
        lastDistToTarget = distToTarget;
    }

    // =====================================================================
    // ===== Imitation Learning Rewards =====
    // =====================================================================
    private float RewardImitationPose() {
        if (currentTraj == null || currentTraj.q == null || currentTraj.q.Count == 0) {
            return 0f;
        }

        int frameIdx = currentFrame % currentTraj.q.Count;
        List<float> refPose = currentTraj.q[frameIdx];

        if (refPose == null || refPose.Count < 19) {
            return 0f;
        }

        // 좌표계 변환
        List<float> unityPose = ConvertJsonToUnity(refPose);

        float poseError = 0f;
        for (int i = 0; i < 12; i++) {
            float currentQ = joints[i].jointPosition[0];  // 라디안
            float refQ = unityPose[7 + i];  // 라디안
            float diff = currentQ - refQ;
            poseError += diff * diff;
        }

        return Mathf.Exp(-poseError * 10f);
    }

    private float RewardImitationVelocity() {
        if (currentTraj == null || currentTraj.v == null || currentTraj.v.Count == 0) {
            return 0f;
        }

        int frameIdx = currentFrame % currentTraj.v.Count;
        List<float> refVel = currentTraj.v[frameIdx];

        if (refVel == null || refVel.Count < 18) {
            return 0f;
        }

        float velError = 0f;
        for (int i = 0; i < 12; i++) {
            float currentV = joints[i].jointVelocity[0];
            float refV = refVel[6 + i];  // FreeFlyer 속도 6개 제외
            float diff = currentV - refV;
            velError += diff * diff;
        }

        return Mathf.Exp(-velError * 2f);
    }

    // =====================================================================
    // ===== Stability Rewards =====
    // =====================================================================
    private float RewardOrientation() {
        Vector3 euler = baseLink.transform.eulerAngles;
        float roll = NormalizeAngle(euler.z);
        float pitch = NormalizeAngle(euler.x);
        float orientError = (roll * roll + pitch * pitch) / (90f * 90f);
        return Mathf.Exp(-orientError * 5f);
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

    // =====================================================================
    // ===== Termination =====
    // =====================================================================
    private void CheckTermination() {
        // Base 높이 체크
        float baseHeight = baseLink.transform.position.y;
        if (baseHeight < 0.3f) {
            if (debugMode) {
                Debug.LogWarning($"Episode ended: Base too low (height={baseHeight:F3}m)");
            }
            AddReward(-10f);
            EndEpisode();
            return;
        }

        // Orientation 체크
        Vector3 euler = baseLink.transform.eulerAngles;
        float roll = NormalizeAngle(euler.z);
        float pitch = NormalizeAngle(euler.x);
        if (Mathf.Abs(roll) > 60f || Mathf.Abs(pitch) > 60f) {
            if (debugMode) {
                Debug.LogWarning($"Episode ended: Fell over (roll={roll:F1}°, pitch={pitch:F1}°)");
                Debug.LogWarning($"Base height: {baseHeight:F3}m, Position: {baseLink.transform.position}");
            }
            AddReward(-10f);
            EndEpisode();
            return;
        }

        // Timeout
        if (movementTimer > movementDuration) {
            Vector3 toTarget = targetPos - baseLink.transform.position;
            toTarget.y = 0;
            float distToTarget = toTarget.magnitude;

            if (distToTarget > stopRadius) {
                AddReward(-5f);
            }
            EndEpisode();
            return;
        }
    }

    public void HandleGroundCollision() {
        AddReward(-10f);
        EndEpisode();
    }

    // =====================================================================
    // ===== Heuristic =====
    // =====================================================================
    public override void Heuristic(in ActionBuffers actionsOut) {
        var cont = actionsOut.ContinuousActions;
        for (int i = 0; i < 12; i++) {
            cont[i] = 0f;
        }
    }
}
