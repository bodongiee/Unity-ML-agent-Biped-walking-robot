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
    private ArticulationDrive[] drives;
    private float[] lowerLimits;
    private float[] upperLimits;

    [Header("Target & Base")]
    public Transform target;
    public ArticulationBody baseLink;

    [Header("Initial Pose (Offset)")]
    public float initialAnkleAngle;
    public float initialHipAngle;
    private Vector3 startPosition;
    private float previousDistanceToTarget;
 
    // URDF 관절 제한값 (도 단위)
    private readonly float[] jointLowerDeg = { -90f, 0f, -46f, -90f, 0f, -46f };
    private readonly float[] jointUpperDeg = { 90f, 115f, 46f, 90f, 115f, 46f };

    public override void Initialize() {
        joints = new ArticulationBody[] { leftHipJoint, leftKneeJoint, leftAnkleJoint, rightHipJoint, rightKneeJoint, rightAnkleJoint };
        drives = new ArticulationDrive[6];
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
        if (upright < 0.3f) {
            AddReward(-2f);
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
        AddReward(facingReward * 0.005f);

        // 4. 가만히 있으면 작은 페널티
        AddReward(-0.001f);

        // 5. 목표 도달
        if (currentDistance < 0.5f) {
            AddReward(50f);
            EndEpisode();
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
