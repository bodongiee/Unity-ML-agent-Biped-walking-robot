# Unity ML-agent Bipedal Robot Walking Project

## Model1(BipedalAgent_1.onnx)
<img src="./Images/model1.gif" width="100%"></img>
### Training Info
| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Total Steps | 67,400,000 |
| Batch Size | 1024 |
| Buffer Size | 10,240 |
| Learning Rate | 0.0003 |
| Hidden Units | 256 |
| Num Layers | 2 |
| Gamma | 0.99 |
| Time Horizon | 1000 |

### Observations (10)
- 타겟과의 거리 (1)
- 타겟 방향 벡터 x, z (2)
- 로봇 직립도 (1)
- 각 관절 각도 (6)

### Actions (6)
- Left Hip, Knee, Ankle
- Right Hip, Knee, Ankle

### Rewards
- 직립 유지: +0.01 * uprightness
- 높이 유지: +0.01 * height
- 전진: +5.0 * progress
- 목표 도달: +10.0
- 넘어짐: -1.0