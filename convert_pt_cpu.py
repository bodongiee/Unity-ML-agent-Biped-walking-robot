import torch
import sys

input_path = "results/bipedal_walk_v46/BipedalAgent_v4/BipedalAgent_v4-30830620.pt"
output_path = "results/bipedal_walk_v46/BipedalAgent_v4/checkpoint_cpu.pt"

print(f"Loading: {input_path}")
checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
print(f"Saving to CPU: {output_path}")
torch.save(checkpoint, output_path)
print("Done!")


#python convert_pt_cpu.py
#cp results/bipedal_walk_v6/BipedalAgent/checkpoint_cpu.pt results/bipedal_walk_v6/BipedalAgent/checkpoint.pt
#rm -rf results/export_final
#mlagents-learn config.yaml --run-id=export_final --initialize-from=bipedal_walk_v6