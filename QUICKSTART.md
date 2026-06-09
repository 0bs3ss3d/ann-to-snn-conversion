# Complete ANN-to-SNN Conversion Framework - Quick Start Guide

## 🚀 Installation & Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/0bs3ss3d/ann-to-snn-conversion.git
cd ann-to-snn-conversion

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For Jupyter notebook interface
pip install jupyter jupyterlab ipywidgets
```

---

## 📋 Quick Start Commands

### Option 1: Full Pipeline (Recommended for thesis)
```bash
# Train ANN + Calibrate + Convert + Evaluate (end-to-end)
python main.py --mode full_pipeline \
    --dataset cifar10 \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --time_steps 150 \
    --device cuda \
    --seed 42
```

**Output:**
- `checkpoints/ann_final.pth` - Trained ANN
- `checkpoints/snn_final.pth` - Converted SNN
- `checkpoints/results.json` - Metrics (accuracy, firing rates, etc.)

---

### Option 2: Step-by-Step (For detailed analysis)

#### Step 1: Train ANN only
```bash
python main.py --mode train_ann \
    --dataset cifar10 \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --device cuda
```

#### Step 2: Calibrate & Convert to SNN
```bash
python main.py --mode calibrate_and_convert \
    --checkpoint checkpoints/ann_best.pth \
    --calibration_samples 1000 \
    --vth_percentile 99.0 \
    --device cuda
```

#### Step 3: Evaluate both models
```bash
python main.py --mode evaluate \
    --checkpoint checkpoints/ann_best.pth \
    --time_steps 100 \
    --batch_size 128 \
    --device cuda
```

---

### Option 3: Interactive Jupyter Notebook (Best for thesis documentation)

```bash
# Convert notebook reference to Jupyter format
jupyter notebook

# Then open and run thesis_experiments.ipynb step-by-step
# This provides:
#   - Data visualization
#   - Training history plots
#   - Activation analysis
#   - Side-by-side ANN vs SNN comparison
#   - Publication-ready figures
```

---

## 📊 Expected Results on CIFAR-10

```
ANN Accuracy:        ~87-90%
SNN Accuracy:        ~85-88%
Accuracy Drop:       ~2-3% (excellent conversion!)
Average Sparsity:    ~90-95% (high energy efficiency)
Inference Time (T=100): ~2-3 seconds on GPU
```

---

## 🎯 For Your MRes Thesis

### Recommended Workflow:

1. **Data Preparation** (Week 1)
   ```bash
   python main.py --mode full_pipeline --epochs 5 --time_steps 50
   # Quick test run to verify setup
   ```

2. **Primary Experiments** (CIFAR-10)
   ```bash
   python main.py --mode full_pipeline \
       --epochs 100 \
       --time_steps 100 \
       --vth_percentile 99.0 \
       --device cuda
   ```

3. **Ablation Studies** (Threshold sensitivity)
   ```bash
   # Test different percentiles
   for percentile in 90 95 99 99.5; do
       python main.py --mode full_pipeline \
           --vth_percentile $percentile \
           --checkpoint_dir "checkpoints/ablation_p${percentile}"
   done
   ```

4. **Alternative Datasets** (GTSRB road signs)
   ```bash
   # After implementing GTSRB loader in data.py
   python main.py --mode full_pipeline \
       --dataset road_signs \
       --epochs 100 \
       --device cuda
   ```

5. **GPU Efficiency Analysis**
   ```bash
   # Compare ANN vs SNN computation time
   python analyze_efficiency.py --checkpoint checkpoints/snn_final.pth
   ```

---

## 📈 Command-Line Reference

### Training Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--epochs` | 100 | 10-500 | Training epochs |
| `--batch_size` | 128 | 32-256 | Batch size |
| `--learning_rate` | 0.001 | 1e-4 to 1e-2 | Adam learning rate |
| `--weight_decay` | 1e-4 | 0 to 1e-3 | L2 regularization |

### SNN Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--time_steps` | 100 | Inference time steps (100-500 typical) |
| `--vth_percentile` | 99.0 | Percentile for threshold calibration |
| `--threshold_scale` | 1.0 | Threshold scaling factor (0.8-1.2) |
| `--calibration_samples` | 1000 | Samples for activation statistics |

### System Parameters
| Parameter | Default | Choices |
|-----------|---------|----------|
| `--device` | cuda | `cuda`, `cpu` |
| `--seed` | 42 | Integer (for reproducibility) |
| `--num_workers` | 4 | Number of data loading threads |

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Reduce batch size
python main.py --mode full_pipeline --batch_size 64

# Or reduce time steps for SNN evaluation
python main.py --mode evaluate --time_steps 50
```

### Issue: Slow data loading
```bash
# Increase number of workers
python main.py --mode full_pipeline --num_workers 8
```

### Issue: CPU fallback (no GPU)
```bash
# Explicitly use CPU
python main.py --mode full_pipeline --device cpu --batch_size 32

# Note: Much slower, use for testing only
```

### Issue: Accuracy drop > 5% (poor conversion)
```bash
# Adjust threshold percentile (try lower)
python main.py --mode calibrate_and_convert --vth_percentile 95

# Or adjust threshold scale
python main.py --mode calibrate_and_convert --threshold_scale 0.9
```

---

## 📚 Project Structure

```
ann-to-snn-conversion/
├── data.py                    # Dataset loading (CIFAR-10, extensible to GTSRB)
├── ann_model.py               # ANN architecture (SimpleCNN, DeepCNN)
├── train_ann.py               # Training pipeline (CrossEntropyLoss + Adam)
├── calibration.py             # Activation collection & threshold normalization
├── snn_model.py               # SNN with LIF neurons (snnTorch)
├── evaluate.py                # ANN & SNN evaluation metrics
├── utils.py                   # Device setup, utilities, constants
├── main.py                    # CLI entry point with argparse
├── thesis_experiments.ipynb   # Jupyter notebook for interactive analysis
├── requirements.txt           # Python dependencies
��── README.md                  # Full documentation
├── QUICKSTART.md              # This file
├── checkpoints/               # Auto-created: saved models
└── data/                      # Auto-created: downloaded datasets
```

---

## 🎓 Research Tips for Thesis

### 1. Reproducibility
```bash
# Always specify seed for reproducible results
python main.py --mode full_pipeline --seed 42 --device cuda
```

### 2. Multiple Runs (Statistical Significance)
```bash
for seed in 42 43 44 45 46; do
    python main.py --mode full_pipeline \
        --seed $seed \
        --checkpoint_dir "checkpoints/seed_${seed}"
done
```

### 3. Monitor GPU Usage
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### 4. Save Experimental Logs
```bash
python main.py --mode full_pipeline 2>&1 | tee logs/experiment_$(date +%Y%m%d_%H%M%S).log
```

---

## 📖 References for Thesis

[1] Bu, T., Ding, J., Huang, T., & Yu, Z. (2023). Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks. arXiv:2303.04347

[2] Rathi, N., Srinivasan, G., Panda, P., & Roy, K. (2020). Enabling deep spiking neural networks with hybrid conversion and spike timing dependent backpropagation. ICLR 2020. https://openreview.net/forum?id=B1xSperKvH

[3] Zhang, Q., Wu, C., Kahana, A., Kim, Y., Li, Y., Karniadakis, G. E., & Panda, P. (2023). Artificial to spiking neural networks conversion for scientific machine learning. arXiv:2308.16372

---

## 🤝 Contributing & Extending

### Add Custom Dataset (e.g., GTSRB)

1. Edit `data.py` - add `load_gtsrb()` function
2. Update `get_class_names()` with GTSRB classes
3. Run: `python main.py --mode full_pipeline --dataset gtsrb`

### Add Alternative Architecture

1. Edit `ann_model.py` - create new `MyModel(nn.Module)`
2. Update `create_model()` factory function
3. Run: `python main.py --mode full_pipeline --architecture mymodel`

### Add Custom Metrics

1. Edit `evaluate.py` - add new metric function
2. Call from `print_evaluation_summary()`

---

## ✅ Verification Checklist

- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] GPU detected (`python -c "import torch; print(torch.cuda.is_available())`)
- [ ] Quick test passes (`python main.py --mode train_ann --epochs 1 --batch_size 64`)
- [ ] Checkpoints directory created (`ls -la checkpoints/`)
- [ ] Dataset downloaded automatically (`ls -la data/cifar10/`)

---

**Last Updated:** June 2025  
**Author:** Your Name  
**For:** MRes Thesis - ANN-to-SNN Conversion for FPGA-Oriented Neuromorphic Vision Systems
