# Improving Video-Based Classification of Infant Fidgety Movements

MSc Artificial Intelligence project, University of Bedfordshire (2025/2026). Author: Aryan.

The project studies three linked papers and builds on them step by step:

1. **SMIL** (Hesse et al., MICCAI 2018): an infant body model. Reconstructed and used for registration.
2. **MINI-RGBD** (Hesse et al., ECCV Workshops 2018): an infant RGB-D benchmark. Reproduced from its SMIL parameters, with depth processing (point clouds, RANSAC table removal, 3D joint lifting).
3. **Kulvicius et al. (2025)**, *Communications Medicine* 5:16: classification of fidgety movements from video, inertial sensors and a pressure mat. The video-only classifier was re-implemented exactly and improved.

## Results

| Model (video only, 9 infant-wise folds) | Balanced accuracy |
|---|---|
| Kulvicius et al. (2025), reported | 90.7% |
| Faithful re-implementation (this repo) | 89.4% |
| + online mirror/rotation augmentation (single network) | 90.2% |
| **+ 16-network ensemble, confirmed on fresh seeds** | **92.0%** (95% CI 89.9-94.1) |

The final model is better than the original selection rule on 8 of 9 folds (Wilcoxon p = 0.012).

Body-model side: MINI-RGBD reconstructed to about 5 mm joint error on 10 of 12 sequences; registration recovers a perturbed pose from 82.2 mm to 3.75 mm; RANSAC table removal matches the dataset masks exactly (IoU 1.000).

## Repository structure

| Folder | Content |
|---|---|
| `reproduction/` | MINI-RGBD loading, SMIL forward reconstruction, depth/mask rendering, MPJPE / IoU / PCKh evaluation |
| `registration/` | Registration energy (scan term + pose/shape priors) and the controlled recovery test |
| `classification/` | `faithful.py` (exact re-implementation, augmentation, parallel multi-network training), `analyse.py` (selection rule, ensembles, Wilcoxon tests), `paper_numbers.py`, `teacher.py`, `smil3d.py` |
| `tools/` | RANSAC table removal and joint lifting demo, diagnostics, figure scripts |

## Data and models (not included)

The data and models below are covered by their own licences and must be downloaded from their owners:

- **SMIL model**: from the SMIL authors; place it in `smil/`.
- **MINI-RGBD**: from Fraunhofer IOSB; place it in `MINI-RGBD_web/`.
- **Kulvicius et al. (2025) keypoint data and code**: from the links in the paper; place them in `viddata/`.

## Setup and usage

```bash
pip install -r requirements.txt

# MINI-RGBD reconstruction and metrics
python reproduction/evaluate_all.py
python tools/demo_lift_and_table.py 01 0

# Video classifier: 8 plain + 8 augmented networks per fold, then the analysis
python classification/faithful.py --runs 8 --seed 1 --tag paper_r8_s1
python classification/faithful.py --runs 8 --seed 1 --aug 8 --tag aug8_r8_s1
python classification/analyse.py paper_r8_s1 aug8_r8_s1
```

GPU training needs the CUDA build of PyTorch. One experiment (9 folds x 8 networks) takes about 1.5 hours on an RTX 3050 laptop GPU.

## References

- Hesse, N. et al. (2018) Learning an infant body model from RGB-D data for accurate full body motion analysis. MICCAI 2018.
- Hesse, N. et al. (2018) Computer vision for medical infant motion analysis: state of the art and RGB-D data set. ECCV Workshops 2018.
- Kulvicius, T. et al. (2025) Deep learning empowered sensor fusion boosts infant movement classification. Communications Medicine 5, 16.
