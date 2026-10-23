
# Citrus Leaves and TrashNet: Baseline Evaluation

## Overview

This contribution adds two image classification datasets
to stable-datasets:

- Citrus Leaves: citrus leaf disease classification.
- TrashNet: household waste classification.

Both datasets were successfully downloaded, loaded,
and validated with automated tests.

## Experimental Setup

A simple supervised classification baseline was used
to verify that both datasets support model training.

- Model: SGDClassifier with logistic loss
- Library: scikit-learn
- Input: RGB images resized to 16 x 16
- Features: flattened RGB pixels normalized to [0, 1]
- Preprocessing: StandardScaler
- Random state: 42
- Metrics: Accuracy and Balanced Accuracy

## Results

| Dataset | Train | Test | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|---:|
| Citrus Leaves | 475 | 119 | 80.67% | 75.71% |
| TrashNet | 2019 | 508 | 41.93% | 39.25% |

## Dataset Details

### Citrus Leaves

- Total samples: 594
- Number of classes: 4
- Evaluation: stratified 80/20 train/test split
- Source: https://data.mendeley.com/datasets/3f83gxmv57/2

### TrashNet

- Total valid samples: 2527
- Number of classes: 6
- Training samples: 2019
- Test samples: 508
- Evaluation: deterministic class-wise holdout
- Source: https://github.com/garythung/trashnet

The TrashNet archive includes a macOS metadata file
that is not a valid image. The loader excludes this
file before generating the dataset cache.

## Reproduction

Install the package and required tools:

    python -m pip install -e .
    python -m pip install pytest

Run the automated tests:

    python -m pytest stable_datasets/tests/test_citrus_trashnet.py -v

Run the classification experiments:

    python examples/citrus_trashnet_baseline.py

## Limitations

These experiments use a basic supervised linear
classifier on resized raw pixels.

They serve as functional validation of the dataset
loading pipeline, rather than a state-of-the-art
classification benchmark.

They do not constitute a self-supervised learning
or world model evaluation.
