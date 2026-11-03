# Classification of Lung Respiratory Sounds

A deep learning project that classifies breathing cycles from lung auscultation recordings into **Normal**, **Crackle**, **Wheeze**, or **Both**, using log-mel spectrograms and a convolutional neural network (CNN).

## Overview

Respiratory diseases such as COPD, asthma and pneumonia are commonly screened by listening to lung sounds with a stethoscope. Abnormal sounds, mainly **crackles** and **wheezes**, are important indicators, but manual auscultation is subjective and depends on the clinician's experience.

This project builds an automated pipeline that:
1. Segments each recording into respiratory cycles using the dataset's annotations
2. Converts each cycle to a log-mel spectrogram
3. Classifies the cycle with a CNN into one of four classes

| Label | Class | Meaning |
|:-----:|-------|---------|
| 0 | Normal | No adventitious sounds |
| 1 | Crackle | Crackles only |
| 2 | Wheeze | Wheezes only |
| 3 | Both | Crackles and wheezes |

## Dataset

**Respiratory Sound Database (ICBHI 2017 Challenge)**
Source: [Kaggle - vbookshelf/respiratory-sound-database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)

- 920 annotated recordings from 126 patients
- 6,898 respiratory cycles, each annotated for the presence of crackles and wheezes
- Recordings were collected with different stethoscopes and sample rates, and include real-world noise
- Patient-level diagnoses (COPD, URTI, pneumonia, etc.) are **heavily imbalanced**, with COPD dominating

Because of this diagnosis-level imbalance, the project uses the **cycle-level crackle/wheeze labels**, which give a more balanced four-class problem. The cycle-level classes are still imbalanced (Normal is the largest, "Both" the smallest), and this is handled as described below.

> The dataset is not included in this repository. The notebook downloads it automatically via `kagglehub`.

## Approach

```
Audio (.wav) + annotations (.txt)
        │
        ▼
Cycle segmentation (start/end times)
        │
        ▼
Resample to 8 kHz → pad/crop to 5 s → peak normalisation
        │
        ▼
Log-mel spectrogram (64 mel bands, n_fft=512, hop=128)
        │
        ▼
CNN (4 conv blocks + GAP + dense) → softmax over 4 classes
```

**Handling class imbalance and avoiding data leakage**
- **Patient-wise split** (train / validation / test), so no patient appears in more than one split. A random split would leak data, since one patient has many similar recordings.
- **Class weights** in the loss function
- **SpecAugment** (frequency and time masking) applied to training data only
- **Evaluation with imbalance-aware metrics**: macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, and the official **ICBHI score** (average of specificity and sensitivity)

**Model**
- 4 convolutional blocks (32, 64, 128, 128 filters) with BatchNorm, ReLU, MaxPooling and Dropout
- Global average pooling, a dense layer (64) and a softmax output
- Adam optimiser, early stopping and learning-rate scheduling

## Results

Baseline CNN trained from scratch, evaluated on held-out test patients (1,698 cycles):

| Metric | Score |
|--------|:-----:|
| Accuracy | 0.427 |
| Balanced accuracy | 0.384 |
| Macro F1 | 0.374 |
| ICBHI score | 0.424 (Sp = 0.368, Se = 0.479) |

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| Normal | 0.565 | 0.368 | 0.446 | 798 |
| Crackle | 0.387 | 0.643 | 0.484 | 516 |
| Wheeze | 0.464 | 0.234 | 0.311 | 222 |
| Both | 0.225 | 0.290 | 0.253 | 162 |

**Observations**
- The model detects crackles best (highest recall), while wheeze and the combined class are hardest. They are the smallest classes, and a small CNN trained from scratch has little data to learn from.
- Results come from a strict patient-wise split, so they are lower, but more realistic, than random-split results reported in some papers.

**Requirements:** Python 3.9+, TensorFlow 2.x, Librosa, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Limitations

- Trained on a single dataset of limited size; results may not generalise to other devices or clinical settings
- Classification is on pre-segmented cycles, so it depends on the dataset's annotations
- Metrics come from one train/test split; with about 100 test patients, results can vary between splits
- This is an educational project and **not a medical diagnostic tool**

## Future Work

- Transfer learning with pretrained models (EfficientNet, YAMNet, PANNs, Audio Spectrogram Transformer)
- Waveform-level augmentation (noise, pitch shift, time stretch) and mixup
- Multi-seed or cross-validated evaluation reported as mean ± std
- Diagnosis-level classification (e.g. Healthy / COPD / Other) with appropriate rebalancing
- Model deployment as a simple web demo

## Acknowledgements

Dataset provided by the ICBHI 2017 Challenge organisers and shared on Kaggle by vbookshelf.

## Author

Anamika J
- GitHub: github.com/Anamika79
- LinkedIn: linkedin.com/in/anamika-j
