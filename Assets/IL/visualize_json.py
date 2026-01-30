import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import sys
import os
import json
import time

# Paths
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(CUR_DIR, "bipedal_v3_stable.URDF")
ACTIONS_DIR = os.path.join(CUR_DIR, "Actions")

def load_json_trajectory(json_path):
    print(f"Opening {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert list to numpy arrays
    if 'q' not in data:
        print("Error: JSON does not contain 'q' key")
        sys.exit(1)
        
    q_list = np.array(data['q'])
    dt = data.get('dt', 0.01)
    
    return q_list, dt

def main():
    target_file = None
    
    # Check if file provided argument
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.exists(arg_path):
            target_file = arg_path
        elif os.path.exists(os.path.join(ACTIONS_DIR, arg_path)):
            target_file = os.path.join(ACTIONS_DIR, arg_path)
        else:
            print(f"File not found: {arg_path}")

    # If no valid file yet, prompt user
    if not target_file:
        if not os.path.exists(ACTIONS_DIR):
            print(f"Actions directory not found: {ACTIONS_DIR}")
            return
            
        files = [f for f in os.listdir(ACTIONS_DIR) if f.endswith('.json')]
        files.sort()
        
        if not files:
            print(f"No JSON files found in {ACTIONS_DIR}")
            return

        print(f"\nFound {len(files)} trajectory files in {ACTIONS_DIR}:")
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
        
        try:
            selection = input(f"\nEnter number (0-{len(files)-1}) or filename: ")
            
            # Check if user typed filename
            if selection in files:
                target_file = os.path.join(ACTIONS_DIR, selection)
            else:
                idx = int(selection)
                if 0 <= idx < len(files):
                    target_file = os.path.join(ACTIONS_DIR, files[idx])
                else:
                    print("Invalid index")
                    return
        except ValueError:
            print("Invalid input")
            return

    print(f"\nTarget File: {target_file}")
    
    # Load Data
    q_traj, dt = load_json_trajectory(target_file)
    print(f"Loaded {len(q_traj)} frames. dt={dt}s. Duration={len(q_traj)*dt:.2f}s")

    # Load Model
    if not os.path.exists(URDF_PATH):
        print(f"URDF not found at {URDF_PATH}")
        return

    print(f"Loading URDF: {URDF_PATH}")
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    visual_model = pin.buildGeomFromUrdf(
        model, URDF_PATH, pin.GeometryType.VISUAL,
        package_dirs=[CUR_DIR]
    )
    
    # Init Visualizer
    try:
        v = MeshcatVisualizer(model, visual_model, visual_model)
        v.initViewer(open=True)
        v.loadViewerModel("pinocchio")
    except Exception as e:
        print(f"Failed to initialize Meshcat visualizer: {e}")
        print("Make sure 'meshcat' is installed (pip install meshcat)")
        return
    
    print("\n=== Starting Visualization ===")
    print(f"Open your browser at the URL shown above (usually http://127.0.0.1:7000/static/)")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            for i, q in enumerate(q_traj):
                v.display(q)
                # Playback speed control
                time.sleep(dt) 
            
            print("Replaying sequence...")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
