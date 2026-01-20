using UnityEngine;

public class ChildCollision : MonoBehaviour {
    private BipedalAgent agent_v1;
    private BipedalAgent_v2 agent_v2;

    private void Awake() {
        // Try finding v2 first (Priority)
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
            if (agent_v2 != null) {
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