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
    private int stoppingTimer = 0; // Timeout for stopping phase

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
 
    // Gait Parameters
    private float m_GaitPhase;
    private float m_GaitPeriod = 0.8f; // Seconds per cycle

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
        m_GaitPhase = 0f; // Reset phase
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

        // GAIT PHASE Signals -> 2
        float phase = m_GaitPhase * 2 * Mathf.PI;
        sensor.AddObservation(Mathf.Sin(phase)); 
        sensor.AddObservation(Mathf.Cos(phase));

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
        // Update Phase
        m_GaitPhase += Time.fixedDeltaTime / m_GaitPeriod;
        m_GaitPhase %= 1f;

        var cont = actions.ContinuousActions;


        for (int i = 0; i < joints.Length; i++) {
            if (joints[i] == null) continue;

            //기준 각도(Bias) 설정 - 오리걸음 자세 기준
            float bias = 0f;
            if (i == 0 || i == 3) bias = initialHipAngle;       // Hip
            else if (i == 1 || i == 4) bias = initialKneeAngle; // Knee
            else if (i == 2 || i == 5) bias = initialAnkleAngle; // Ankle

            float actionScale = 40f;
            if (i == 0 || i == 3) actionScale = 60f; // Increased Hip Range

            float targetAngle = bias + (cont[i] * actionScale);
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
        Vector3 toTarget = (target.position - baseLink.transform.position).normalized;
        Vector3 forward = baseLink.transform.forward;
        Vector3 forwardFlat = new Vector3(forward.x, 0, forward.z).normalized;

        // 1. Survival & Upright
        if (upright < 0.5f) {
            AddReward(-5f);
            EndEpisode();
            return;
        }
        AddReward(0.01f); // Existing reward

        // 2. Navigation State Machine
        float angleToTarget = Vector3.Angle(forwardFlat, toTarget);
        bool needsTurn = angleToTarget > 15f; // Tightened to 15 degrees (Strict)
        bool isStopping = currentDistance < 0.5f;

        if (isStopping) {
            // STOPPING MODE
            float velocity = baseLink.linearVelocity.magnitude;
            
            // Penalize movement VERY HEAVILY
            AddReward(-0.2f * velocity); 
            
            // Penalize angular velocity too
            AddReward(-0.1f * baseLink.angularVelocity.magnitude);

            // Count stable steps (velocity < 0.2)
            if (velocity < 0.2f) {
                targetStaySteps++;
            } else {
                targetStaySteps = Mathf.Max(0, targetStaySteps - 2); // Decay faster
            }
            
            // ABSOLUTE TIMEOUT: Force end after 200 steps in zone regardless of state
            stoppingTimer++;
            if (stoppingTimer > 200) {
                // Partial reward based on how stable it was
                float partialReward = Mathf.Clamp(targetStaySteps * 0.5f, 0f, 30f);
                AddReward(partialReward - 2f); // Small penalty for not fully stopping
                EndEpisode();
                return;
            }

            // SUCCESS: Full reward for staying completely still
            if (targetStaySteps > 50) {
                AddReward(50f);
                EndEpisode();
                return;
            }
        } 
        else {
            // Reset ALL counters when leaving zone (prevents re-entry farming)
            if (stoppingTimer > 0) {
                // Penalty for leaving the zone after entering
                AddReward(-0.5f);
            }
            stoppingTimer = 0;
            targetStaySteps = 0;
            
            if (needsTurn) {
                // TURNING MODE (Priority)
                float turnReward = Vector3.Dot(forwardFlat, toTarget); 
                AddReward(turnReward * 0.1f);
                
                // Strict penalty for moving forward while turning
                float fwdVel = Vector3.Dot(baseLink.linearVelocity, forwardFlat);
                if(fwdVel > 0.1f) AddReward(-0.05f); 
            } 
            else {
                // WALKING MODE
                // Progress Reward: Only reward reducing distance
                float progress = previousDistanceToTarget - currentDistance;
                if (progress > 0) {
                    AddReward(progress * 30f); 
                }
            }
        }

        // 3. Posture & Gait (ONLY ACTIVE WHEN NOT STOPPING)
        if (!isStopping) {
            // Posture
            Vector3 localUp = baseLink.transform.InverseTransformDirection(Vector3.up);
            if (localUp.z < -0.1f) AddReward(localUp.z * 0.1f); 
            AddReward(-0.001f * baseLink.angularVelocity.magnitude);

            // Cyclic Gait
            bool leftShouldSupport = (m_GaitPhase >= 0.5f);
            bool rightShouldSupport = (m_GaitPhase < 0.5f);

            float contactReward = 0f;
            if (leftFoot != null && leftFoot.isGrounded == leftShouldSupport) contactReward += 0.1f;
            if (rightFoot != null && rightFoot.isGrounded == rightShouldSupport) contactReward += 0.1f;
            AddReward(contactReward);

            // Hip Swing Reward
            if (!leftShouldSupport && leftFoot != null) {
                float swingVel = Vector3.Dot(leftAnkleJoint.linearVelocity, forwardFlat);
                if(swingVel > 0) AddReward(swingVel * 0.05f);
            }
            if (!rightShouldSupport && rightFoot != null) {
                float swingVel = Vector3.Dot(rightAnkleJoint.linearVelocity, forwardFlat);
                if(swingVel > 0) AddReward(swingVel * 0.05f);
            }
        }

        previousDistanceToTarget = currentDistance;
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
