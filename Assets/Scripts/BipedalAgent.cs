using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class BipedalAgent : Agent {
    [Header("Joint References")]
    public ArticulationBody leftHipJoint;
    public ArticulationBody leftKneeJoint;
    public ArticulationBody leftAnkleJoint;
    public ArticulationBody rightHipJoint;
    public ArticulationBody rightKneeJoint;
    public ArticulationBody rightAnkleJoint;
    private ArticulationBody[] joints;
    //private ArticulationDrive[] drives;
    private float[] lowerLimits;
    private float[] upperLimits;
    private float[] previousVelocities = new float[6];  // 관절 가속도 계산용

    private float leftFootAirTime = 0f;
    private float rightFootAirTime = 0f;
    private bool leftFootGrounded = true;
    private bool rightFootGrounded = true;
    private int targetStaySteps = 0;  // 타겟 영역 체류 스텝 수

    [Header("Foot Contact")]
    public FootContact leftFoot;
    public FootContact rightFoot;

    [Header("Target & Base")]
    public Transform target;
    public ArticulationBody baseLink;

    [Header("Initial Pose (Offset)")]
    private float initialAnkleAngle;
    private float initialHipAngle;
    private float initialKneeAngle;
    private Vector3 startPosition;
    private float previousDistanceToTarget;
 
    // URDF 관절 제한값 (도 단위) - 오리걸음용
    // Hip: 0~-90 (앞으로만), Knee: 0~115, Ankle: -65~65
    private readonly float[] jointLowerDeg = { -90f, 0f, -65f, -90f, 0f, -65f };
    private readonly float[] jointUpperDeg = { 0f, 115f, 65f, 0f, 115f, 65f };

    public override void Initialize() {
        joints = new ArticulationBody[] { leftHipJoint, leftKneeJoint, leftAnkleJoint, rightHipJoint, rightKneeJoint, rightAnkleJoint };
        //drives = new ArticulationDrive[6];

        // 오리걸음 초기 자세
        initialAnkleAngle = -30f;
        initialHipAngle = -30f;   
        initialKneeAngle = 60f;  
        lowerLimits = jointLowerDeg;
        upperLimits = jointUpperDeg;

        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;
            
            // 물리 설정 튜닝, 기본 설정
            joints[i].jointFriction = 0.1f;
            joints[i].angularDamping = 1f;

            var drive = joints[i].xDrive;
            drive.stiffness = 10000f; 
            drive.damping = 100f;     
            drive.forceLimit = 1000f; 
            drive.lowerLimit = lowerLimits[i];
            drive.upperLimit = upperLimits[i];
            joints[i].xDrive = drive;
        }
        
        //유니티 환경에서의 로봇 위치를 START 포지션으로 설정 (오리걸음 자세에 맞게 낮춤) 
        if (baseLink != null) {
            Vector3 pos = baseLink.transform.position;
            pos.y = 0.65f;  // 오리걸음 자세 높이
            startPosition = pos;
        }
    }

    private void ResetRobotPose() {
        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;

            float targetDeg = 0f;
            // 오리걸음 초기 자세 설정
            if (i == 0 || i == 3) targetDeg = initialHipAngle;      
            else if (i == 1 || i == 4) targetDeg = initialKneeAngle;
            else if (i == 2 || i == 5) targetDeg = initialAnkleAngle;

            var drive = joints[i].xDrive;
            drive.target = targetDeg;
            joints[i].xDrive = drive;

            joints[i].jointPosition = new ArticulationReducedSpace(targetDeg * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);

        }

        if (baseLink != null) {
            baseLink.TeleportRoot(startPosition, Quaternion.identity);
            baseLink.linearVelocity = Vector3.zero;
            baseLink.angularVelocity = Vector3.zero;
        }
    }

    public override void OnEpisodeBegin() {
        ResetRobotPose();

        // 이전 속도 초기화
        for (int i = 0; i < 6; i++) {
            previousVelocities[i] = 0f;
        }
        // Time in Air 초기화
        leftFootAirTime = 0f;
        rightFootAirTime = 0f;
        leftFootGrounded = true;
        rightFootGrounded = true;
        targetStaySteps = 0;

        if (target != null && baseLink != null) {
            float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
            
            float distance = Random.Range(2f, 10f);
            Vector3 randomOffset = new Vector3(Mathf.Cos(angle) * distance, 0f, Mathf.Sin(angle) * distance);
            target.position = startPosition + randomOffset;

            previousDistanceToTarget = Vector3.Distance(baseLink.transform.position, target.position);
        }
    }

//UnityEditor.TransformWorldPlacementJSON:{"position":{"x":0.0,"y":0.7670000195503235,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"scale":{"x":0.5,"y":0.20000000298023225,"z":0.25}}
    public override void CollectObservations(VectorSensor sensor) {
        
        //타겟과의 거리 -> 1
        sensor.AddObservation(Vector3.Distance(baseLink.transform.position, target.position) / 10f);

        //타겟 방향 (로봇 기준 로컬 좌표) -> 2
        Vector3 toTarget = (target.position - baseLink.transform.position).normalized;
        Vector3 localDir = baseLink.transform.InverseTransformDirection(toTarget);
        sensor.AddObservation(localDir.x);  // 로봇 기준 앞뒤
        sensor.AddObservation(localDir.z);  // 로봇 기준 좌우

        //로봇이 서있는 정도 -> 1
        sensor.AddObservation(Vector3.Dot(baseLink.transform.up, Vector3.up)); 

        //각 관절 현재 각도 -> 6
        foreach (var j in joints) {
            sensor.AddObservation(j.jointPosition[0]);
        }

        //각 관절 현재 속도 -> 6
        foreach (var j in joints) {
            sensor.AddObservation(j.jointVelocity[0] / 30f);
        }

        //각 발 현재 상태 -> 2
        sensor.AddObservation(leftFoot != null && leftFoot.isGrounded ? 1f : 0f);
        sensor.AddObservation(rightFoot != null && rightFoot.isGrounded ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions) {
        var cont = actions.ContinuousActions;


        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;

            //기준 각도(Bias) 설정 - 오리걸음 자세 기준
            float bias = 0f;
            if (i == 0 || i == 3) bias = initialHipAngle;       // Hip
            else if (i == 1 || i == 4) bias = initialKneeAngle; // Knee
            else if (i == 2 || i == 5) bias = initialAnkleAngle; // Ankle

            float targetAngle = bias + (cont[i] * 40f);
            targetAngle = Mathf.Clamp(targetAngle, lowerLimits[i], upperLimits[i]);

            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }

        CheckRewards();
    }

    private void CheckRewards() {
        float currentDistance = Vector3.Distance(baseLink.transform.position, target.position);
        float upright = Vector3.Dot(baseLink.transform.up, Vector3.up);

        // 1. 넘어짐 체크
        if (upright < 0.5f) {
            AddReward(-5f);
            EndEpisode();
            return;
        }

        // 2. 타겟 방향 계산
        Vector3 toTarget = (target.position - baseLink.transform.position).normalized;
        Vector3 forward = baseLink.transform.forward;
        float facingReward = Vector3.Dot(new Vector3(forward.x, 0, forward.z).normalized, toTarget);

        // 3. 전진 보상 (앞을 향할 때만)
        float progress = previousDistanceToTarget - currentDistance;
        if (facingReward > 0.7f) {
            AddReward(progress * 10f);
        // 백스텝으로 가면 페널티
        } else if (facingReward < -0.7f && progress > 0) {
            AddReward(-0.1f);
        }
        previousDistanceToTarget = currentDistance;


        // 4. 가만히 있으면 작은 페널티 (타겟 근처가 아닐 때만)
        if (currentDistance > 2f) {
            float speed = baseLink.linearVelocity.magnitude;
            if (speed < 0.1f) {
                AddReward(-0.01f);
            }
        }

        // 5. 감속 보상 (타겟 2m 이내에서 거리에 비례해 감속 유도)
        float baseSpeed = baseLink.linearVelocity.magnitude;
        if (currentDistance < 2f) {
            // 거리가 가까울수록 낮은 속도 요구
            float allowedSpeed = currentDistance;
            if (baseSpeed > allowedSpeed) {
                AddReward((allowedSpeed - baseSpeed) * 0.1f);  // 0.01f → 0.1f 강화
            }
        }

        // Stability Assist at Target
        if (currentDistance < 0.5f) {
            if(upright > 0.8f) AddReward(0.05f); 
            else AddReward(-0.05f); // Force stability when stopping
        }

        // 6. 목표 도달
        if (currentDistance < 0.5f) {
            targetStaySteps++;

            float jointSpeed = 0f;
            foreach (var j in joints) {
                jointSpeed += Mathf.Abs(j.jointVelocity[0]);
            }

            // 완전히 정지했으면 큰 보상 + 종료
            // Guidance to stop: Reward for low velocity when close
            AddReward(Mathf.Clamp01(1f - baseSpeed) * 0.1f);

            // STOPPING LOGIC FIX:
            // If stayed for 50 steps (approx 1s), count as success regardless of perfect zero velocity.
            if (targetStaySteps > 50) {
                 AddReward(50f);
                 EndEpisode();
            }
            // Strict stop bonus (optional)
            else if (baseSpeed < 0.1f && jointSpeed < 1f) {
                AddReward(50f);
                EndEpisode();
            }
        } else {
            targetStaySteps = 0;  // 타겟 영역 벗어나면 리셋
        }
        
        //UnityEditor.TransformWorldPlacementJSON:{"position":{"x":0.0,"y":0.7670000195503235,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"scale":{"x":1.0,"y":1.0,"z":1.0}}
        //============================================================================================
        // Duck Walk Curriculum Learning
        //============================================================================================
        int currentStep = Academy.Instance.StepCount;

        // Phase 2: Crouch Height Maintenance (After 5M steps)
        if(currentStep == 5000000) {
            Debug.Log("=====================================================");
            Debug.Log("Phase 2: Crouch Height Maintenance (After 5M steps)");
            Debug.Log("=====================================================");
        }
        if (currentStep > 5000000) {
            float targetHeight = 0.65f;
            float currentHeight = baseLink.transform.position.y;
            float heightError = Mathf.Abs(currentHeight - targetHeight);
            AddReward(Mathf.Exp(-5f * heightError) * 0.1f);
            
            // Explicit Penalty for being too high (Prevent lifting/tiptoes)
            if(currentHeight > 0.75f) {
                AddReward((0.75f - currentHeight) * 0.1f);
            }
        }

        // Phase 3: High Step & Waddle (After 10M steps)
        if(currentStep == 10000000) {
            Debug.Log("=====================================================");
            Debug.Log("Phase 3: High Step & Waddle (After 10M steps)");
            Debug.Log("=====================================================");
        }
        if (currentStep > 10000000) {
            // High Step Reward
            float targetStepHeight = 0.25f;
            
            if (leftFoot != null && !leftFoot.isGrounded) {
                 float footH = leftAnkleJoint.transform.position.y;
                 if(footH > 0.05f) {
                     float range = Mathf.Clamp01(footH / targetStepHeight);
                     AddReward(range * 0.05f);
                 }
            }
            if (rightFoot != null && !rightFoot.isGrounded) {
                 float footH = rightAnkleJoint.transform.position.y;
                 if(footH > 0.05f) {
                     float range = Mathf.Clamp01(footH / targetStepHeight);
                     AddReward(range * 0.05f);
                 }
            }

            // Hip Usage / Shuffle Prevention
            float hipVelocitySum = Mathf.Abs(leftHipJoint.jointVelocity[0]) + Mathf.Abs(rightHipJoint.jointVelocity[0]);
            if (baseLink.linearVelocity.magnitude > 0.1f) {
                 AddReward(Mathf.Clamp01(hipVelocitySum) * 0.02f);
            }
        }

    }

    public void HandleGroundCollision() {
        AddReward(-1f);
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
