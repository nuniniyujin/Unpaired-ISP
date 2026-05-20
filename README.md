# NTIRE 2026 Unpaired ISP Transfer Challenge
[CVPRW] Lightweight Unpaired Smartphone ISP Transfer with Semantic Pseudo-Pairing 

Our paper ranked 4th in the NTIRE 2026 Challenge on Learned Smartphone ISP with Unpaired Data.  


This branch provides a clean, generic workflow for:
- training (`training/train.py`),
- inference (`pipeline/infer.py`, `pipeline/predict.py`),
- pseudo-pair data preparation (`data_processing/*.py`).

## Repository Layout

- `training/train.py`: Main training script (former `train_cnn_new.py` workflow).
- `pipeline/infer.py`: Checkpoint inference on RAW `.npy` files.
- `pipeline/predict.py`: Challenge interface (`--input_dir`, `--output_dir`).
- `model/*.py`: Core models, losses, OT, RAW preprocessing.
- `model/third_party/ffdnet_pytorch/*`: bundled FFDNet dependency.
- `data_processing/raw_image_processing.py`: RAW `.npy` -> processed RGB patches.
- `data_processing/full_image_dino_ot_topk.py`: full-image DINO+OT top-k.
- `data_processing/patch_level_dino_ot_topk.py`: patch-level DINO+OT top-k.
- `data_processing/build_patch_ot_all_from_full.py`: sparse pairing `.npz` builder.
- `checkpoints/weight_url.txt`: pre-trained model checkpoint URL.

## 1) Environment Setup

<details>
<summary><b>Show command</b></summary>

```bash
cd /path/to/repo
pip install -r requirement.txt
```

</details>

## 2) Data Processing

### 2.1) RAW Image Processing

Convert RAW patches (`.npy`, `H x W x 4`) into processed RGB patches.

<details>
<summary><b>Show RAW processing command</b></summary>

```bash
python data_processing/raw_image_processing.py \
  --input_dir /path/to/raw_npy_patches \
  --output_dir /path/to/processed_rgb_patches \
  --raw4_order R,Gr,Gb,B \
  --black_levels 0,0,0,0 \
  --white_levels 1023,1023,1023,1023 \
  --demosaic_method demosaicnet \
  --denoise_method ffdnet \
  --ffdnet_noise_sigma 0.01 \
  --ffdnet_weights model/third_party/ffdnet_pytorch/models/net_rgb.pth \
  --ffdnet_device auto \
  --gamma 2.2 \
  --tone_mapping none \
  --output_ext png
```

</details>

### 2.2) Full-Image DINO+OT Top-k

<details>
<summary><b>Show full-image matching command</b></summary>

```bash
python data_processing/full_image_dino_ot_topk.py \
  --source_dir /path/to/full/source/images \
  --target_dir /path/to/full/target/images \
  --output_dir /path/to/output/full_topk \
  --topk 10 \
  --viz_count 0 \
  --device cuda
```

</details>

Output includes top-k files such as `matches_topk_long.csv` and JSON summaries.

### 2.3) Patch-Level DINO+OT Top-k

<details>
<summary><b>Show patch-level matching command</b></summary>

```bash
python data_processing/patch_level_dino_ot_topk.py \
  --source_dir /path/to/source/patches \
  --target_dir /path/to/target/patches \
  --output_dir /path/to/output/patch_topk \
  --topk 10 \
  --viz_count 0 \
  --device cuda
```

</details>

Output includes top-k files such as `matches_topk_long.csv` and JSON summaries.

### 2.4) Build Sparse Pairing NPZ

<details>
<summary><b>Show sparse pairing command</b></summary>

```bash
python data_processing/build_patch_ot_all_from_full.py \
  --full_topk_csv /path/to/full_topk/matches_topk_long.csv \
  --source_patch_dir /path/to/source/patches \
  --target_patch_dir /path/to/target/patches \
  --output_dir /path/to/pairing_dir \
  --topk_full 10
```

</details>

Use `/path/to/pairing_dir/patch_ot_all_weights_sparse.npz` in training (`--pairing_npz`).

## 3) Training

`train.py` supports stage-1 (no GAN) and stage-2 (GAN refinement) by CLI options.

### Stage 1 (no GAN)

<details>
<summary><b>Show Stage 1 command</b></summary>

```bash
python training/train.py \
  --train_raw_dir /path/to/processed/source/patches \
  --train_rgb_dir /path/to/target/patches \
  --pairing_npz /path/to/pairing_sparse.npz \
  --pairing_weight_key final_weight \
  --pairing_topk 8 \
  --pairing_sampling weighted \
  --target_domain nonlinear \
  --model_type demosaiccnn \
  --hidden 128 \
  --batch_size 24 \
  --lr 1e-4 \
  --epochs 10 \
  --disable_gan \
  --moment_weight 1.0 \
  --hist_luma_weight 1.0 \
  --hist_chroma_weight 1.5 \
  --gram_weight 1.0 \
  --tv_weight 0.05 \
  --resume none \
  --checkpoint_dir /path/to/checkpoints/stage1 \
  --tensorboard \
  --tb_log_dir /path/to/checkpoints/stage1/tensorboard \
  --tb_num_images 10
```

</details>

### Stage 2 (GAN refinement)

<details>
<summary><b>Show Stage 2 command</b></summary>

```bash
python training/train.py \
  --train_raw_dir /path/to/processed/source/patches \
  --train_rgb_dir /path/to/target/patches \
  --pairing_npz /path/to/pairing_sparse.npz \
  --pairing_weight_key final_weight \
  --pairing_topk 8 \
  --pairing_sampling weighted \
  --target_domain nonlinear \
  --model_type demosaiccnn \
  --hidden 128 \
  --batch_size 24 \
  --lr 1e-4 \
  --lr_d 2e-5 \
  --epochs 5 \
  --gan_weight 0.001 \
  --moment_weight 0.2 \
  --hist_luma_weight 1.0 \
  --hist_chroma_weight 1.5 \
  --gram_weight 1.0 \
  --tv_weight 0.01 \
  --init_checkpoint /path/to/checkpoints/stage1/last.pt \
  --resume none \
  --checkpoint_dir /path/to/checkpoints/stage2 \
  --tensorboard \
  --tb_log_dir /path/to/checkpoints/stage2/tensorboard \
  --tb_num_images 10
```

</details>

## 4) Inference

<details>
<summary><b>Show inference command</b></summary>

```bash
python pipeline/infer.py \
  --raw_path /path/to/raw_npy_or_dir \
  --checkpoint /path/to/checkpoint.pt \
  --out_path /path/to/output_dir \
  --output_ext jpg \
  --jpeg_quality 100 \
  --demosaic_method demosaicnet \
  --ffdnet_noise_sigma 0.01 \
  --ffdnet_device auto \
  --input_gamma 2.2 \
  --input_tone_mapping none \
  --output_postprocess none
```

</details>

### 4.1) Challenge Predict Interface

<details>
<summary><b>Show challenge predict command</b></summary>

```bash
python pipeline/predict.py \
  --input_dir /path/to/raw_npy_dir \
  --output_dir /path/to/output_png_dir
```




