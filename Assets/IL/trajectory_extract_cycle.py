import numpy as np
import json
file = 'walking_trajectory_fix_com.npy'
# Parameters
step_length = 0.15   
step_height = 0.05  
dt = 0.01
# Timing
T_init = 0.8            # 초기 안정화
T_shift = 0.6           # Weight shift
T_swing = 0.8           # Swing phase
T_ds = 0.8              # Double support

# Steps
num_steps = 5
one_cycle_time = T_shift + T_swing + T_ds

walking_start_time = int((T_init + one_cycle_time + T_shift + T_swing) / dt) 

walking_stop_time = walking_start_time + int(one_cycle_time * 2 / dt) + 1

traj = np.load(file, allow_pickle=True).item()

print(f"Walking start frame: {walking_start_time}, Stop one cycle frame: {walking_stop_time}, Total frames: {len(traj['q'])}")
sequence = {
    'q' : traj['q'][ walking_start_time : walking_stop_time ],
    'v' : traj['v'][walking_start_time : walking_stop_time ],
    'tau' : traj['tau'][walking_start_time : walking_stop_time ],
    'dt' : traj['dt'],
}

npy_filename = 'trajectory_extract_cycle.npy'
np.save(npy_filename, sequence)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

json_filename = 'trajectory_extract_cycle.json'
with open(json_filename, 'w') as json_file:
    json.dump(sequence, json_file, cls=NumpyEncoder, indent=4)
print(f"Saved cycle trajectory to {npy_filename} and {json_filename}")