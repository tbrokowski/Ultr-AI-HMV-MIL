# HMV-MIL: Hierarchical Multi-View Multiple Instance Learning

Training code for lung ultrasound TB classification using hierarchical multi-view MIL (HMV-MIL) and paper ablation experiments.

## Architecture

HMV-MIL (`model_type: attention_pool`) uses a CLIP ViT backbone with:

1. **Frame-level attention** — learned attention pooling over frames within each anatomical site
2. **Site pathology heads** — auxiliary multi-label pathology detection per site
3. **Patient-level MIL** — gated multiple instance learning for TB classification

## Repository layout

```
ultr_ai/
  train/train_ablation_distributed.py   # Main training entry point
  network_architecture/                 # Models, selectors, backbones
  dataset/                              # Patient-level data loading
  config/multi_task.py                  # YAML configuration loader
configs/                                # 14 experiment families x 5 folds
Data/
  labels/                               # Patient labels
  test_files/                           # 5-fold CV splits
scripts/
  run_ablation_single_node.sh           # Single-node multi-GPU launcher
  submit_parallel_ablations.sh          # Batch launcher (local or SLURM)
```

## Data setup

1. Place lung ultrasound videos under `Data/LusBeninVideos/` (not included in this repo).
2. Provide a video metadata index at `Data/processed_files_2.csv` (not included; required for training).
3. Included metadata:
   - `Data/labels/labels_multidiagnosis.csv`
   - `Data/test_files/Fold_{0-4}.csv`
4. Optional: place local CLIP weights in `CLIP_weights/` (otherwise HuggingFace download is used).

## Environment

Install PyTorch for your CUDA/CPU setup, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Training

Single-node multi-GPU (4 GPUs):

```bash
bash scripts/run_ablation_single_node.sh configs/attention_pool_extra3_full_train2/fold0.yaml
```

Direct invocation:

```bash
torchrun --standalone --nproc_per_node=4 \
  ultr_ai/train/train_ablation_distributed.py \
  --config configs/attention_pool_extra3_full_train2/fold0.yaml \
  --video_folder ./Data/LusBeninVideos
```

Override output location:

```bash
torchrun --standalone --nproc_per_node=4 \
  ultr_ai/train/train_ablation_distributed.py \
  --config configs/attention_pool_extra3_full_train2/fold0.yaml \
  --output_dir ./outputs
```

Submit all paper experiments locally or via SLURM:

```bash
bash scripts/submit_parallel_ablations.sh local   # or: slurm
```

## Experiment mapping

| Config directory | Paper / role |
|------------------|--------------|
| `attention_pool_extra3_full_train2` | HMV-MIL (main model) |
| `uniform_extra3` | UniformSampling |
| `mean_pool_extra3` | NoKeyframe |
| `singletask_extra3` | NoPathology |
| `attention_pool_noInitWeights` | NoPreTraining |
| `attention_pool_extra4_k1` | kEqualOneSelection |
| `attention_pool_extra4_k8` | kEqualEightSelection |
| `attention_pool_cxr` | NoMIL |
| `LeViT-Attention` | LeViT backbone |
| `3dcnn` | 3D-ResNet baseline |
| `cnnlstm` | CNN-LSTM baseline |
| `inception3d` | 3D-Inception baseline |
| `vivit` | Video Transformer baseline |
| `r2plus1d` | R2+1D baseline |

## Outputs

Checkpoints and logs are written under paths in each config YAML (default: `./outputs/ablation_results/<experiment>/fold<N>/`).

## Citation

If you use this code, please cite the accompanying paper (citation TBD).
