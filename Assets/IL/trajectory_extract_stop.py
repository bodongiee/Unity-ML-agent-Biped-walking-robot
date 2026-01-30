import numpy as np
import json
files = {
    'trajectory_stop_lf_support.npy': 'trajectory_stop_lf_support',
    'trajectory_stop_rf_support.npy': 'trajectory_stop_rf_support',
}

# Parameters
step_length = 0.15   
step_height = 0.05  
dt = 0.01
# Timing
T_init = 0.8            # 초기 안정화
T_shift = 0.6           # Weight shift
T_swing = 0.8           # Swing phase
T_ds = 0.4              # Double support

# Steps
num_steps = 3

#One Cycle : ShiftL + SwingR + DS + ShiftR + SwingL + DS
one_cycle_time = T_shift + T_swing + T_ds


walking_time = T_init + one_cycle_time * num_steps + one_cycle_time
print(f"Total walking time: {walking_time:.2f} seconds")

total_frames = int(walking_time / dt) + 1
print(f"Total frames: {total_frames} at dt={dt}s")

for input_file, output_name in files.items():
    traj = np.load(input_file, allow_pickle=True).item()
    total_length = len(traj['q'])
    print(f"Total length of trajectory in {input_file}: {total_length}")

    stop_start_idx = int((T_init + one_cycle_time * (num_steps - 1) + T_shift + T_swing) / dt) + 1
    print(f"Stop start index for {input_file}: {stop_start_idx}")

    stop_sequence = {
        'q' : traj['q'][stop_start_idx : ],
        'v' : traj['v'][stop_start_idx : ],
        'tau' : traj['tau'][stop_start_idx : ],
        'dt' : traj['dt'],
    }

    npy_filemname = f'{output_name}_step.npy'
    np.save(npy_filemname, stop_sequence)
    print(f"Saved stop trajectory to {npy_filemname}")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super(NumpyEncoder, self).default(obj)

    json_filename = f'{output_name}_step.json'
    with open(json_filename, 'w') as json_file:
        json.dump(stop_sequence, json_file, cls=NumpyEncoder, indent=4)
    print(f"Saved stop trajectory to {json_filename}")
