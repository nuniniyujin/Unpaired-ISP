# Build All Patch OT Weights from Full-Image Top-K

Script: `data_processing/build_patch_ot_all_from_full.py`

This script expands full-image matching (`matches_topk_long.csv`) to patch-level OT:
- For each source patch `<prefix>_<idx>`, it collects all target patches from full-image top-K target prefixes.
- It computes patch-level OT weights over that candidate pool.
- It stores sparse outputs for training-time fast lookup.

## Example

```bash
python data_processing/build_patch_ot_all_from_full.py \
  --full_topk_csv /path/to/matches_topk_long.csv \
  --source_patch_dir /lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raws_png_dmnet_ffdnet \
  --target_patch_dir /lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/jpegs \
  --output_dir /path/to/output_pairing_dir \
  --topk_full 4 \
  --full_weight_mode rank_prior \
  --full_rank_prior 0.4,0.3,0.2,0.1 \
  --image_size 224 \
  --extract_batch_size 64 \
  --blend_mode multiply \
  --device cuda
```

## Outputs

- `patch_ot_all_weights_sparse.npz`:
  - `offsets`: row pointer for each source patch (`source_id`)
  - `target_ids`: candidate target patch ids
  - `full_rank`: full-image rank (1..K) for each edge
  - `full_score`: original full-image score for each edge
  - `full_weight`: normalized full-image top-K weight per edge
  - `patch_weight`: patch OT weight per edge
  - `final_weight`: final edge weight (`patch_weight * full_weight`, renormalized by source row in `multiply` mode)
  - `source_paths/source_prefix/source_idx`, `target_paths/target_prefix/target_idx`
- `source_index.csv`: per-source patch index with candidate count.
- `patch_ot_top1.csv`: top-1 target patch per source patch.
- `summary.json`: run metadata/statistics.

## JSON exports (enabled by default)

- `full_topk_by_source_prefix.json`: full-image top-K entries per source prefix (includes `rank`, `score`, `weight_norm`).
- `patch_ot_top1.json`: top-1 patch assignment per source patch (includes `full_rank`).
- `source_index.json`: per-source patch metadata and sparse range (`sparse_start`, `sparse_end`) into the `.npz` edge arrays.

Disable JSON outputs with:

```bash
--no_save_json
```

## Notes

- `--save_long_csv` writes `patch_ot_all_long.csv` (can be very large).
- `blend_mode`:
  - `multiply`: combines patch OT with full-image top-K prior.
  - `replace`: uses patch OT only.
- `full_weight_mode`:
  - `score_norm`: normalize clipped full-image scores (legacy behavior).
  - `rank_prior`: use rank prior weights for full top-K entries (e.g. top4 `0.4,0.3,0.2,0.1`).
