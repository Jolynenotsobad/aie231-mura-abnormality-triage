# Data Directory

## MURA Dataset v1.1

This directory should contain the extracted MURA dataset. **Raw image data is NOT stored in git** (see `.gitignore`).

### Download Instructions

1. Download the MURA dataset from [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/) or [Kaggle](https://www.kaggle.com/datasets/cjinny/mura-v11).
2. Extract the archive so that the directory structure looks like:

```
data/
└── MURA-v1.1/
    ├── train/
    │   ├── XR_ELBOW/
    │   ├── XR_FINGER/
    │   ├── XR_FOREARM/
    │   ├── XR_HAND/
    │   ├── XR_HUMERUS/
    │   ├── XR_SHOULDER/
    │   └── XR_WRIST/
    ├── valid/
    │   └── (same 7 body parts)
    ├── train_image_paths.csv
    ├── train_labeled_studies.csv
    ├── valid_image_paths.csv
    └── valid_labeled_studies.csv
```

### Dataset Summary

| Split | Studies | Images | Positive (abnormal) | Negative (normal) |
|-------|---------|--------|--------------------|--------------------|
| Train | 13,457  | 36,808 | 5,177 (38.5%)      | 8,280 (61.5%)      |
| Valid | 1,199   | 3,197  | 538 (44.9%)        | 661 (55.1%)        |

### Citation

```bibtex
@article{rajpurkar2017mura,
  title={MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Bagul, Aarti and Ding, Daisy and Duan, Tony and Mehta, Hershel and Yang, Brandon and Zhu, Ke and Laird, Dillon and Ball, Robyn L and others},
  journal={arXiv preprint arXiv:1712.06957},
  year={2017}
}
```

### Licence

The MURA dataset is released under a [Stanford University School of Medicine research use agreement](https://stanfordmlgroup.github.io/competitions/mura/) and is permitted for academic use.
