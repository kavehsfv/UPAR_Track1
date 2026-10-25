# UPAR 2027 Track 1 — Pedestrian Attribute Recognition: Code Release

This release reproduces our Codabench Track 1 submission
(username `kavehsfv-957256-957256_VID-Docker-GPU`, leaderboard test-set results below).

**Method:** fine-tuned CLIP ViT-B/16 visual tower (OpenAI "openai" weights) with a
40-way linear head, trained with class-balanced weighted BCE, plus a validation-set
per-attribute logit-bias calibration.

## Test-set results (Codabench leaderboard, 36,343 images)

| Metric | Score |
|---|---|
| Challenge_Avg | 0.7599 |
| mA | 0.7716 |
| Label_F1 | 0.5341 |
| Instance_Acc | 0.6257 |
| Instance_Prec | 0.7152 |
| Instance_Rec | 0.8009 |
| Instance_F1 | 0.7486 |

## Layout

```
run.py                      <- exact submission entry point (Codabench ingestion contract)
training/                   <- everything needed to retrain from scratch
  download_datasets.py      <- downloads Market1501 / PA100k / PETA (Google Drive / Dropbox)
  prepare_clip_init.py      <- converts open_clip CLIP ViT-B/16 (openai) visual tower
  data.py                   <- dataset + CSV loaders + augmentation
  vit.py                    <- plain-PyTorch CLIP-compatible ViT (weight-compatible w/ open_clip)
  metrics.py                <- official-style track metrics (mA, Label_F1, Instance_*, Challenge_Avg)
  train.py                  <- fine-tuning driver
  calibrate_export.py       <- per-attribute bias calibration + submission weight export
weights/
  par_model_ep2_calibrated.pt  <- EXACT weights of the standing submission (epoch 2 + t1_bias)
  calib.json                   <- calibration record (biases, val metrics)
submission_zip/
  task1_clip_vitb16_cal.zip    <- the standing Codabench submission zip (byte-identical)
  task1_clip_vitb16_ep4raw.zip <- earlier variant (epoch 4, zero bias), for reference
validate_zip.py             <- validates a submission zip exactly like the ingestion program
Dockerfile                  <- reproducible inference environment (pytorch 2.4.1, as on Codabench)
```

## Reproducing inference

1. Download the datasets (needed for the val split images; see `training/download_datasets.py`)
   or place your own pedestrian crops + annotation CSVs under `data/` following the
   UPAR 2027 starter-kit layout.
2. Weights: `weights/par_model_ep2_calibrated.pt` (also inside the submission zip at
   `assets/par_model.pt`).
3. Quick check (same code path as the Codabench ingestion):

```bash
python validate_zip.py task1 submission_zip/task1_clip_vitb16_cal.zip --limit 64
```

4. Full val-split evaluation:

```bash
python validate_zip.py task1 submission_zip/task1_clip_vitb16_cal.zip
```

Expected val results (33,407 images): Challenge_Avg 0.8835, mA 0.8941,
Label_F1 0.6673, Instance_F1 0.8731 (calibrated; see `weights/calib.json`).

## Reproducing training

Environment: Python 3.11, PyTorch >= 2.0 + CUDA GPU (a single 32 GB GPU was used),
`open_clip_torch` (only for the one-time CLIP weight conversion), Pillow, NumPy, tqdm, gdown<6.

```bash
cd training
python download_datasets.py                    # Market1501, PA100k, PETA -> ../data/
python prepare_clip_init.py                    # downloads CLIP ViT-B/16 (openai), converts -> weights/
python train.py --epochs 6 --batch-size 256    # ~1.8 min/epoch on an RTX 5090
python calibrate_export.py --track 1           # bias calibration + export -> runs/clip_vitb16/export_t1/
```

Key hyper-parameters (all defaults in `train.py` / `calibrate_export.py`):
- Weighted BCE, per-attribute pos-weight `(neg/pos)^0.5`
- AdamW, weight decay 0.05; backbone LR 2e-5, head LR 1e-3 (differential)
- 500-iteration linear warmup + cosine decay; 6 epochs; batch 256; bf16 AMP; grad-clip 1.0
- Augmentation: horizontal flip (p=0.5), random erasing (p=0.25, grey rectangle)
- Preprocessing: letterbox 224x224 (grey-128 padding, bilinear) + CLIP normalization
- Checkpoint selection: best val Challenge_Avg (epoch 2 in the submitted run)
- Calibration: per-attribute logit bias, coordinate ascent (2 sweeps, 17-point grid in [-3,3]),
  maximizing val Challenge_Avg

## Docker (recommended by the organizers)

```bash
docker build -t upar2027-track1 .
# inside the container, the val split must be mounted at /data
docker run --gpus all -v /path/to/upar-data:/data upar2027-track1
```

The container builds on `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` — the same base
image the Codabench evaluation uses — and runs `validate_zip.py` on the submission zip
so the organizers can verify the Codabench result locally without adjustments.

## Notes

- The exact standing submission weights (`weights/par_model_ep2_calibrated.pt`) are
  byte-identical to `assets/par_model.pt` inside `submission_zip/task1_clip_vitb16_cal.zip`.
- `training/` is self-contained (no imports from the surrounding repo); paths resolve
  relative to the repo layout `data/` next to `training/`.
- The evaluation-time inference code (batching, letterbox, bias application) is the
  `run.py` at the release root — the same file shipped in the zip.

## Fact sheet

`fact_sheet/` contains the competition fact sheet (LaTeX source + compiled PDF) with the
team details and the leaderboard results reported above.
