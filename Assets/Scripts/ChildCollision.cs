using UnityEngine;

public class ChildCollision : MonoBehaviour {
    private BipedalAgent agent_v1;
    private BipedalAgent_v2 agent_v2;
    private BipedalAgent_v3 agent_v3;

    private void Awake() {
        // Try finding v3 first (Priority)
        agent_v3 = GetComponentInParent<BipedalAgent_v3>();
        if (agent_v3 != null) return;

        // Try finding v2
        agent_v2 = GetComponentInParent<BipedalAgent_v2>();
        if (agent_v2 != null) return;

        // Fallback to v1
        agent_v1 = GetComponentInParent<BipedalAgent>();
    }
//======================================================================================
// Collision
//======================================================================================
    private void OnCollisionEnter(Collision collision) {
        if (collision.gameObject.CompareTag("Ground")) {
            if (agent_v3 != null) {
                agent_v3.HandleGroundCollision();
            }
            else if (agent_v2 != null) {
                agent_v2.HandleGroundCollision();
            }
            else if (agent_v1 != null) {
                agent_v1.HandleGroundCollision();
            }
        }
    }

    private void OnCollisionStay(Collision collision) {
        OnCollisionEnter(collision);
    }
    
}