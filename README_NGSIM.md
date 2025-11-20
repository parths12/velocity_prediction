# NGSIM Validation Pipeline
Real-world validation of your iTransformer velocity prediction model

## 📁 Project Structure

```
sumo_new/
├── NGSIM_dataset/              # Your NGSIM data folder
│   ├── trajectories-0750-0805.txt
│   └── ... (other trajectory files)
├── ngsim_preprocessing.py       # Step 1: Convert NGSIM to 26 features
├── ngsim_evaluation.py          # Step 2: Zero-shot evaluation
├── ngsim_transfer_learning.py   # Step 3: Fine-tuning
├── velocity_prediction.py       # Your existing model file
├── best_model.pth              # Your SUMO-trained model
└── README_NGSIM.md             # This file
```

## 🚀 Quick Start Guide

### Step 1: Preprocess NGSIM Data (Day 1-2)

```bash
python ngsim_preprocessing.py
```

**What it does:**
- Loads NGSIM trajectory files from `NGSIM_dataset/`
- Converts to your 26-feature format
- Creates 60s lookback + 20s prediction sequences
- Splits into train (10%), validation (0%), test (90%)
- Saves to `ngsim_processed/`

**Output:**
```
ngsim_processed/
├── ngsim_train.pkl      # 10% for fine-tuning
├── ngsim_val.pkl        # 0% (not used)
├── ngsim_test.pkl       # 90% for testing
└── ngsim_metadata.pkl   # Dataset info
```

**Expected time:** 5-10 minutes for 1 trajectory file

---

### Step 2: Zero-Shot Evaluation (Day 3-4)

```bash
python ngsim_evaluation.py
```

**What it does:**
- Loads your SUMO-trained model (`best_model.pth`)
- Tests on NGSIM without any retraining
- Measures domain gap (NGSIM RMSE - SUMO RMSE)
- Creates visualizations

**Output:**
```
ngsim_evaluation_results/
├── ngsim_results.pkl                # Metrics
├── ngsim_results.txt                # Human-readable
├── ngsim_sample_predictions.png     # 10 trajectory samples
├── ngsim_error_vs_horizon.png       # RMSE/MAE over 20s
└── ngsim_scatter_plot.png           # Predicted vs Actual
```

**Expected results:**
- **Best case:** RMSE 0.4-0.5 (domain gap < 0.2)
  → Excellent generalization! Model works on real data.
- **Good case:** RMSE 0.5-0.7 (domain gap 0.2-0.3)
  → Acceptable, fine-tuning will help.
- **Needs work:** RMSE > 0.7 (domain gap > 0.3)
  → Significant domain gap, fine-tuning necessary.

---

### Step 3: Transfer Learning (Day 5-6)

**Only if zero-shot RMSE > 0.5** (otherwise skip to comparison!)

```bash
python ngsim_transfer_learning.py
```

**What it does:**
- Loads SUMO-trained model
- Freezes embedding + first transformer layer (keeps learned patterns)
- Fine-tunes on NGSIM training data (10%)
- Lower learning rate (1e-4 vs 1e-3)
- Saves fine-tuned model

**Output:**
```
ngsim_finetuned_model.pth
ngsim_transfer_learning_curves.png
```

**Expected improvement:**
- Before: RMSE 0.6-0.8
- After: RMSE 0.4-0.5
- Improvement: 20-30%

---

### Step 4: Re-evaluate Fine-tuned Model

After fine-tuning, test again:

1. Edit `ngsim_evaluation.py`:
   ```python
   MODEL_PATH = 'ngsim_finetuned_model.pth'  # Changed from 'best_model.pth'
   ```

2. Run evaluation again:
   ```bash
   python ngsim_evaluation.py
   ```

3. Compare before/after:
   - Zero-shot RMSE: X.XX
   - Fine-tuned RMSE: Y.YY
   - Improvement: ZZ%

---

## 📊 Expected Timeline

| Day | Task | Output | Time |
|-----|------|--------|------|
| 1-2 | Preprocessing | `ngsim_processed/` | 30 min |
| 3-4 | Zero-shot eval | Domain gap measured | 15 min |
| 5-6 | Fine-tuning | Improved model | 1-2 hours |
| 7 | Re-eval + report | Final results | 30 min |

**Total: ~3-4 hours of work over 7 days**

---

## 🎯 Success Metrics

### Goal 1: Real-World Validation
- ✅ NGSIM RMSE < 0.5 → Proves model works beyond simulation
- ✅ Aggregate accuracy > 90% → High-quality predictions

### Goal 2: Beat McMaster Paper
- McMaster 3s RMSE: 0.321
- Your 20s RMSE: 0.356 (SUMO), 0.4-0.5 (NGSIM expected)
- **Key insight:** You're solving a HARDER problem (20s vs 3s)!

### Goal 3: Publication-Ready
- ✅ Validated on real data → Credibility
- ✅ Domain gap < 0.3 → Acceptable generalization
- ✅ Transfer learning works → Shows robustness

---

## 🔧 Troubleshooting

### Issue 1: FileNotFoundError

**Problem:** `FileNotFoundError: NGSIM_dataset/trajectories-0750-0805.txt`

**Solution:**
1. Check folder name: `NGSIM_dataset` (case-sensitive)
2. Check file exists: `ls NGSIM_dataset/`
3. Update filename in `ngsim_preprocessing.py`:
   ```python
   TRAJECTORY_FILES = [
       'your_actual_filename.txt',  # Update here
   ]
   ```

### Issue 2: ImportError

**Problem:** `ImportError: cannot import name 'iTransformer' from 'velocity_prediction'`

**Solution:**
1. Ensure `velocity_prediction.py` is in same folder
2. Check class name matches: `class iTransformer(nn.Module):`
3. If different, update imports in scripts

### Issue 3: CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
1. Reduce batch size:
   ```python
   BATCH_SIZE = 16  # Was 32
   ```
2. Or use CPU:
   ```python
   device='cpu'  # In evaluator/learner initialization
   ```

### Issue 4: High Domain Gap (RMSE > 0.8)

**Problem:** Zero-shot RMSE very high, big domain gap

**Solutions:**
1. **Run transfer learning** (mandatory!)
2. Increase fine-tuning epochs:
   ```python
   EPOCHS = 30  # Was 20
   ```
3. Unfreeze more layers:
   ```python
   freeze_early_layers=False  # Only freeze embedding
   ```

---

## 📈 What to Report in Paper

### Section: NGSIM Validation

**Results:**
- SUMO RMSE: 0.356 m/s
- NGSIM RMSE (zero-shot): X.XX m/s
- NGSIM RMSE (fine-tuned): Y.YY m/s
- Domain gap: Z.ZZ m/s (AA% of SUMO)

**Analysis:**
- "Model successfully transfers to real-world NGSIM data with XX% accuracy"
- "Transfer learning reduces domain gap by YY%, achieving ZZ m/s RMSE"
- "Demonstrates generalization beyond simulation to actual highway driving"

**Comparison to McMaster:**
- "While McMaster achieves 0.321 RMSE on 3s horizon, our model maintains 0.4-0.5 RMSE on 20s horizon (6.7× longer prediction)"
- "Normalized by horizon length, our per-second error rate is competitive"

---

## 🎓 Next Steps After Validation

### Week 2: Multi-Horizon Implementation
Add 3s, 10s, 20s prediction heads to directly compare with McMaster at 3s.

**Expected:**
- Your 3s RMSE: 0.15-0.20 (BEATS McMaster's 0.321!)
- Your 10s RMSE: 0.25-0.30
- Your 20s RMSE: 0.35-0.40

### Week 3: Ablation Studies
Test with fewer features to find optimal configuration.

### Week 4: Paper Writing
Combine all results into 8-page conference paper.

---

## 📞 Need Help?

**Common questions:**

Q: "Which NGSIM file should I use?"
A: Start with `trajectories-0750-0805.txt` (15-minute segment)

Q: "How much data do I need?"
A: 1 file (~5000 sequences) is enough for validation. More is better!

Q: "What if my model doesn't import?"
A: Check `velocity_prediction.py` has `class iTransformer(nn.Module):`

Q: "Should I fine-tune?"
A: Only if zero-shot RMSE > 0.5. Otherwise, you're already good!

---

## ✅ Pre-flight Checklist

Before running scripts:

- [ ] NGSIM data files in `NGSIM_dataset/` folder
- [ ] `velocity_prediction.py` exists (your model file)
- [ ] `best_model.pth` exists (your trained weights)
- [ ] PyTorch installed (`pip install torch`)
- [ ] All dependencies installed (`pip install pandas numpy sklearn matplotlib`)

**Ready to start? Run Step 1: `python ngsim_preprocessing.py`**

---

## 📝 File Descriptions

### ngsim_preprocessing.py (500 lines)
**Purpose:** Convert NGSIM to your format
**Input:** Raw NGSIM trajectory files
**Output:** Processed sequences (600×26 input, 20 output)
**Key functions:**
- `load_trajectory_file()` - Read NGSIM CSV
- `get_surrounding_vehicles()` - Find 13 neighbors
- `extract_features()` - Create 26-feature vector
- `create_sequences()` - Build 60s → 20s pairs
- `split_data()` - Train/val/test split

### ngsim_evaluation.py (400 lines)
**Purpose:** Test model on NGSIM
**Input:** Trained model + processed NGSIM data
**Output:** Metrics + visualizations
**Key functions:**
- `evaluate()` - Run inference on test set
- `compute_metrics()` - RMSE, MAE, MAPE
- `visualize_predictions()` - Create plots
- `print_results()` - Domain gap analysis

### ngsim_transfer_learning.py (350 lines)
**Purpose:** Fine-tune on NGSIM
**Input:** SUMO model + NGSIM train data
**Output:** Fine-tuned model
**Key functions:**
- `freeze_layers()` - Preserve SUMO knowledge
- `fine_tune()` - Train on NGSIM
- `evaluate()` - Measure improvement
- `plot_training_curves()` - Training progress

---

## 🎉 Success!

**When you complete all steps, you'll have:**

1. ✅ Real-world validation (NGSIM)
2. ✅ Domain gap measured and addressed
3. ✅ Transfer learning demonstrated
4. ✅ Publication-ready results
5. ✅ Comparison to McMaster paper
6. ✅ Visualizations and metrics

**You're now 30% done with Week 1 goals! Keep going! 💪**
