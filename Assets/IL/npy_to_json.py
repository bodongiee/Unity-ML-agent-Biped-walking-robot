import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

# 데이터 로드
data = np.load('./trajectory_start_rf_support.npy', allow_pickle=True)

data_list = data.tolist()

# JSON 저장
with open('./trajectory_start_rf_support_step.json', 'w') as json_file:
    json.dump(data_list, json_file, cls=NumpyEncoder, indent=4)

print("Successfully converted npy to json!")