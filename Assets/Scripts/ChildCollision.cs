using UnityEngine;

public class ChildCollision : MonoBehaviour {
    private BipedalAgent agent;

    private void Awake() {
        agent = GetComponentInParent<BipedalAgent>();
        if (agent != null) {
            Debug.Log($"[{transform.name}] Found BipedalAgent in parent: {agent.name}");
            return;
        }
        
    }
//======================================================================================
// Collision
//======================================================================================
    private void OnCollisionEnter(Collision collision) {
        if (collision.gameObject.CompareTag("Ground")) {
            if (agent != null) {
                agent.HandleGroundCollision();
            }
        }
    }

    private void OnCollisionStay(Collision collision) {
        OnCollisionEnter(collision);
    }
    
}