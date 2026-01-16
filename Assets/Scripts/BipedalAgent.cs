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
    public float initialAnkleAngle;
    public float initialHipAngle;
    private Vector3 startPosition;
    private float previousDistanceToTarget;
 
    // URDF 관절 제한값 (도 단위)
    private readonly float[] jointLowerDeg = { -90f, 0f, -65f, -90f, 0f, -65f };
    private readonly float[] jointUpperDeg = { 90f, 115f, 65f, 90f, 115f, 65f };

    public override void Initialize() {
        joints = new ArticulationBody[] { leftHipJoint, leftKneeJoint, leftAnkleJoint, rightHipJoint, rightKneeJoint, rightAnkleJoint };
        //drives = new ArticulationDrive[6];
        initialAnkleAngle = -8f;
        initialHipAngle = 0f;
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
        //유니티 환경에서의 로봇 위치를 START 포지션으로 설정
        if (baseLink != null) startPosition = baseLink.transform.position;
    }

    private void ResetRobotPose() {
        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;

            float targetDeg = 0f;
            //처음 로봇이 기울어지게 시작하도록 하는 설정
            if (i == 2 || i == 5) targetDeg = initialAnkleAngle; 
            else if (i == 0 || i == 3) targetDeg = initialHipAngle;

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
        sensor.AddObservation(localDir.x);  // 로봇 기준 좌우
        sensor.AddObservation(localDir.z);  // 로봇 기준 앞뒤

        //로봇이 서있는 정도 -> 1
        sensor.AddObservation(Vector3.Dot(baseLink.transform.up, Vector3.up)); // Uprightness
        //각 관절 현재 각도 -> 6
        foreach (var j in joints) {
            sensor.AddObservation(j.jointPosition[0]);
        }

        foreach (var j in joints) {
            sensor.AddObservation(j.jointVelocity[0] / 30f);
        }

        sensor.AddObservation(leftFoot != null && leftFoot.isGrounded ? 1f : 0f);
        sensor.AddObservation(rightFoot != null && rightFoot.isGrounded ? 1f : 0f);
    }
    public override void OnActionReceived(ActionBuffers actions) {
        var cont = actions.ContinuousActions;


        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;

            //기준 각도(Bias) 설정
            float bias = 0f;
            if (i == 2 || i == 5) bias = initialAnkleAngle;
            else if (i == 0 || i == 3) bias = initialHipAngle;

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

        // 2. 전진 보상
        float progress = previousDistanceToTarget - currentDistance;
        AddReward(progress * 10f);
        previousDistanceToTarget = currentDistance;

        // 3. 타겟 방향 보상 (타겟을 향할수록 보상)
        Vector3 toTarget = (target.position - baseLink.transform.position).normalized;
        Vector3 forward = baseLink.transform.forward;
        float facingReward = Vector3.Dot(new Vector3(forward.x, 0, forward.z).normalized, toTarget);
        AddReward(facingReward * 0.01f);

        // 4. 가만히 있으면 작은 페널티
        //AddReward(-0.001f);

        // 5. 감속 보상 (타겟 2m 이내에서 거리에 비례해 감속 유도)
        float baseSpeed = baseLink.linearVelocity.magnitude;
        if (currentDistance < 2f) {
            // 거리가 가까울수록 낮은 속도 요구
            float allowedSpeed = currentDistance;
            if (baseSpeed > allowedSpeed) {
                AddReward((allowedSpeed - baseSpeed) * 0.01f);  // 초과 속도에 페널티
            }
        }

        // 6. 목표 도달
        if (currentDistance < 0.5f) {
            targetStaySteps++;

            float jointSpeed = 0f;
            foreach (var j in joints) {
                jointSpeed += Mathf.Abs(j.jointVelocity[0]);
            }

            // 완전히 정지했으면 큰 보상 + 종료
            if (baseSpeed < 0.1f && jointSpeed < 1f) {
                AddReward(50f);
                EndEpisode();
            }
            // 50스텝 (약 3초) 내에 정지 못하면 작은 보상 + 종료
            else if (targetStaySteps > 50) {
                AddReward(10f);
                EndEpisode();
            }
        } else {
            targetStaySteps = 0;  // 타겟 영역 벗어나면 리셋
        }
        
        //UnityEditor.TransformWorldPlacementJSON:{"position":{"x":0.0,"y":0.7670000195503235,"z":0.0},"rotation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0},"scale":{"x":1.0,"y":1.0,"z":1.0}}
        //============================================================================================
        //Curriculum Learning
        //============================================================================================
        //BipedalAgent-20028066.pt -> Just walking, Tensorboard graph -> plateau
        
        //============================================================================================
        //1차 커리큘럼 업데이트 : 2,000,000+ 스텝에서 사용
        //============================================================================================
        //무릎 사용
        float minKneeAngle = 35f;
        float maxKneeAngle = 60f;
        float currentLeftKnee = leftKneeJoint.jointPosition[0] * Mathf.Rad2Deg;
        float currentRightKnee = rightKneeJoint.jointPosition[0] * Mathf.Rad2Deg;

        float leftKneePenalty = 0f;
        float rightKneePenalty = 0f;
        if(currentLeftKnee < minKneeAngle)
            leftKneePenalty += (minKneeAngle - currentLeftKnee);
        else if(currentLeftKnee > maxKneeAngle)
            leftKneePenalty += (currentLeftKnee - maxKneeAngle);

        if(currentRightKnee < minKneeAngle)
            rightKneePenalty += (minKneeAngle - currentRightKnee);
        else if(currentRightKnee > maxKneeAngle)
            rightKneePenalty += (currentRightKnee - maxKneeAngle);                        

        AddReward((leftKneePenalty + rightKneePenalty) * (-0.01f));

        //로봇 높이 조정
        float maxHeight = 0.6f;
        float minHeight = 0.3f;
        float currentPoisition = baseLink.transform.position.y;
        if(currentPoisition > maxHeight) {
            AddReward((currentPoisition - maxHeight) * (-0.01f));
        }
        else if(currentPoisition < minHeight) {
            AddReward((minHeight - currentPoisition) * (-0.01f));
        }


        //로봇 베이스 기울기 (수평 유지)
        float baseAngle = Mathf.Abs(Vector3.Dot(baseLink.transform.forward, Vector3.up));
        AddReward(-baseAngle * 0.01f);

        /*        
        //============================================================================================
        //2차 커리큘럼 업데이트 : 
        //============================================================================================

        //발 높이 보상 (Expressive Whole-Body Control for Humanoid Robots #1)
        float targetFootHeight = 0.15f;
        if(leftFoot != null) {
            if (!leftFoot.isGrounded) {
                float footHeight = leftAnkleJoint.transform.position.y;
                float footHeightDiff = Mathf.Abs(footHeight - targetFootHeight);
                AddReward(footHeightDiff * (-0.01f));

            }
        }
        if(rightFoot != null) {
            if (!rightFoot.isGrounded) {
                float footHeight = rightAnkleJoint.transform.position.y;
                float footHeightDiff = Mathf.Abs(footHeight - targetFootHeight);
                AddReward(footHeightDiff * (-0.01f));

            }
        }

        //발 공중  체류 보상 (Expressive Whole-Body Control for Humanoid Robots #2)
        if(leftFoot != null) {
            if (!leftFoot.isGrounded) {
                leftFootAirTime += Time.fixedDeltaTime;
            }
            else if (leftFoot.isGrounded && !leftFootGrounded) {
                float effectiveAirTime = Mathf.Min(leftFootAirTime, 0.6f);
                AddReward(effectiveAirTime * 0.5f);
                leftFootAirTime = 0f;
            }
            leftFootGrounded = leftFoot.isGrounded;
        }

        if(rightFoot != null) {
            if (!rightFoot.isGrounded) {
                rightFootAirTime += Time.fixedDeltaTime;
            }
            else if (rightFoot.isGrounded && !rightFootGrounded) {
                float effectiveAirTime = Mathf.Min(rightFootAirTime, 0.6f);
                AddReward(rightFootAirTime * 0.5f);
                rightFootAirTime = 0f;
            }
            rightFootGrounded = rightFoot.isGrounded;
        }
        //Drag 페널티 (Expressive Whole-Body Control for Humanoid Robots #3)
        float dragPenalty = 0f;
        if(leftFoot != null) {
            if(leftFoot.isGrounded) {
                Vector3 footV = leftAnkleJoint.linearVelocity;
                float horizontalSpeed = new Vector2(footV.x, footV.z).magnitude;
                dragPenalty += horizontalSpeed;
            }
        }

        if(rightFoot != null) {
            if(rightFoot.isGrounded) {
                Vector3 footV = rightAnkleJoint.linearVelocity;
                float horizontalSpeed = new Vector2(footV.x, footV.z).magnitude;
                dragPenalty += horizontalSpeed;
            }
        }
        AddReward(dragPenalty * (-0.01f));

        //관절 가속도 페널티 (Expressive Whole-Body Control for Humanoid Robots #6)
        float accPenalty = 0f;
        for (int i = 0; i < 6; i++) {
            float acc = (joints[i].jointVelocity[0] - previousVelocities[i]) / Time.fixedDeltaTime;
            accPenalty += Mathf.Abs(acc);
            previousVelocities[i] = joints[i].jointVelocity[0];
        }
        AddReward(accPenalty * (-0.0001f));

        //수직 선속도 페널티 (Expressive Whole-Body Control for Humanoid Robots #12)
        float verticalVelocity = baseLink.linearVelocity.y;
        AddReward(Mathf.Abs(verticalVelocity) * (-0.01f));

        //수평면 각속도 페널티 (Expressive Whole-Body Control for Humanoid Robots #13)
        float horizontalAngularVelocity = baseLink.angularVelocity.y;
        AddReward(Mathf.Abs(horizontalAngularVelocity) * (-0.001f));
        */

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
