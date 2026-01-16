using UnityEngine;

public class FootContact : MonoBehaviour {
    public bool isGrounded = false;

    void OnCollisionEnter(Collision collision) {
        if (collision.gameObject.CompareTag("Ground")) {
            isGrounded = true;
        }
    }

    void OnCollisionStay(Collision collision) {
        if (collision.gameObject.CompareTag("Ground")) {
            isGrounded = true;
        }
    }

    void OnCollisionExit(Collision collision) {
        if (collision.gameObject.CompareTag("Ground")) {
            isGrounded = false;
        }
    }
}