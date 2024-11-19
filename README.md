# PrefAcc - A GPU-Accelerated Benchmark for RLHF

PrefAcc is a GPU-accelerated benchmark for Reinforcement Learning from Human Feedback (RLHF), 
achieving 40-50x faster training compared to existing benchmarks. 
Built on Google's Brax physics engine, PrefAcc enables efficient evaluation of RLHF algorithms in control task settings.


## Environment Setup

To set up the development environment (using CUDA):

1. Create a virtual environment:
   ```bash
   python -m venv myenv
   ```

2. Activate the virtual environment:
     ```bash
     source myenv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

In case you want to use a different accelerator than CUDA (or set up from scratch):
```bash
pip install brax 
```
and then follow the install instructions for your device https://github.com/jax-ml/jax#installation.

To get started, open [`training.ipynb`](training.ipynb) in a Jupyter notebook to train and evaluate the models.

## Tested Environment

This project has been tested on the following hardware/software configuration:
- GPU: NVIDIA RTX 4070 SUPER
- CPU: AMD Ryzen 5700X
- OS: Ubuntu 22.04 LTS
- NVIDIA Driver Version: 555

Your results may vary with different hardware configurations.

## Performance

PrefAcc achieves significant speedup compared to existing RLHF benchmarks through full GPU acceleration.

### Training Speed Comparison
| Metric              | PrefAcc | BPref    | Speedup |
|--------------------|----------|-----------|----------|
| Steps/Second       | ~200K    | ~4K       | 50x      |
| Time (4M steps)    | 98s      | 74min     | 45x      |

## Available Oracles

PrefAcc implements three types of synthetic oracles to simulate different patterns of human feedback:

### Perfect Oracle
- Provides optimal feedback based on ground truth rewards
- Returns preference σ₁ ≻ σ₂ whenever sum of rewards for σ₁ exceeds σ₂
- Serves as an upper baseline for oracle comparisons

### Mistake Oracle  
- Simulates human errors and inconsistencies
- Flips the perfect oracle's preference with probability ϵ (default: 0.1)
- Models scenarios where humans provide incorrect feedback

### Myopic Oracle
- Models short-sighted human decision making
- Weights earlier state-action pairs more heavily than later ones
- Uses decay factor γ to reduce influence of later timesteps
- Simulates humans who make quick judgments based on early observations

## Citation

If you use PrefAcc in your research, please cite:

```bibtex
@misc{senftraiss2024prefacc,
    title={PrefAcc: A GPU-Accelerated Benchmark for RLHF},
    author={Senft-Raiß, Jonas Maximilian},
    school={LMU Munich},
    year={2024},
    type={Bachelor's Thesis},
    month={November},
    address={Munich, Germany}
}
```

# License and Attribution

This project builds upon Google's Brax PPO implementation (https://github.com/google/brax)
to create an RLHF (Reinforcement Learning from Human Feedback) benchmark.
Copyright for the original PPO implementation: 2024 The Brax Authors.

Key modifications and additions:
- Extended the PPO implementation to support RLHF
- Implemented synthetic feedback generation

The original Brax code is licensed under Apache License 2.0.

All modifications and additional features are licensed under MIT.