# DeepSeek-R1 Implementation 🧠➗

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/)

Advanced implementation of DeepSeek-R1 mathematical reasoning model with Group Relative Policy Optimization (GRPO).

## Key Features ✨

- **GRPO Training**: Novel group-based RL training approach
- **Multi-modal Rewards**: Combined format + accuracy rewards
  - Mathematical verification
  - Secure code execution
- **Safety Distillation**: Knowledge transfer with safety constraints
- **LoRA Support**: Efficient parameter fine-tuning
- **Distributed Training**: Accelerate integration for multi-GPU

## Installation ⚙️

```bash
git clone https://github.com/mkantwala/DeepSeek-R1-TrainingSuite.git
cd deepseek-r1
pip install -r requirements.txt
```

## Quick Start 🚀

### Training Configuration

```yaml
# configs/base_config.yaml
model_config:
  base_model: "deepseek-ai/deepseek-math-7b-base"
  max_length: 2048
  lora:
    r: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]

training_params:
  epochs: 1000
  batch_size: 16
  learning_rate: 2e-5
  group_size: 4
```

### Start Training

```bash
accelerate launch scripts/train.py \
  --config configs/base_config.yaml \
  --dataset_path math_dataset
```

### Distillation

```bash
python scripts/distill.py \
  --teacher_model trained_teacher \
  --student_model deepseek-ai/deepseek-math-7b-base \
  --dataset math_dataset
```

## Project Structure 📂

```bash
deepseek-r1/
├── configs/              # Training configurations
├── data/                 # Data processing modules
├── training/             # Core training logic
├── reward/               # Reward calculation system
├── distillation/         # Safety distillation
├── models/               # Model architectures
├── scripts/              # Operational scripts
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Advanced Features 🔥

### Custom Reward Components

Implement custom reward functions:

```python
from reward.reward_calculator import BaseRewardCalculator

class CustomReward(BaseRewardCalculator):
    def calculate_reward(self, response, ground_truth):
        # Implement custom logic
        return {"total": custom_score, ...}
```

### Multi-GPU Training

Utilize Accelerate for distributed training:

```bash
accelerate config  # Set up distributed environment
accelerate launch scripts/train.py
```

### LoRA Configuration

Edit `configs/base_config.yaml` to modify LoRA parameters:

```yaml
lora:
  r: 16
  lora_alpha: 64
  target_modules: ["q_proj", "v_proj", "output_proj"]
  bias: "none"
```

## Testing 🧪

Run comprehensive test suite:

```bash
pytest tests/ -v
```

## Contribution 🤝

Contributions welcome! Please follow:

1. Fork the repository
2. Create your feature branch
3. Submit a pull request
