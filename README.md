# NHATS-Clock-Drawing-CNN
# NHATS Clock Drawing Classification using CNN (PyTorch)

This repository implements a **supervised deep learning pipeline** for classifying **NHATS Round 11 Clock Drawing Test (CDT) images** using **Convolutional Neural Networks (CNNs)** in PyTorch. The goal is to predict cognitive status using **clock drawing images combined with NHATS clinical labels**.

---

## 📌 Project Overview

- **Dataset**: NHATS Round 11 (CDT images + clinical/demographic data)
- **Task**: Multi-class classification of cognitive status
- **Input**: Clock Drawing Test (TIF images)
- **Labels**: Derived from NHATS health condition variables
- **Framework**: PyTorch
- **Environment**: Google Colab (GPU-supported)

---

## 🧠 Labels Definition

| Label | Meaning |
|------|--------|
| 0 | Pre-dementia |
| 1 | Post-dementia |
| 2 | Normal |

Labels are derived from `hc11disescn10` with temporal consistency across participants.

---

## 🧪 Data Processing Pipeline

1. Load NHATS Round 11 STATA files
2. Convert `.dta` → `.csv`
3. Clean invalid / missing responses
4. Align CDT images using participant IDs (`spid`)
5. Create balanced train / validation / test splits
6. Generate dictionary-based loaders for image streaming

---

## 🏗 Model Architecture

- Custom CNN built from scratch:
  - 2D Convolutions
  - Batch Normalization
  - ReLU activation
  - Max Pooling
  - Dropout (regularization)
  - Fully connected layers
- Optional support for **ResNet50 (transfer learning)**

---

## ⚙️ Training Configuration

- Optimizer: SGD / Adam
- Loss: CrossEntropyLoss
- Batch size: 8 (train), 4 (val), 1 (test)
- GPU acceleration supported
- Learning rate scheduling (ReduceLROnPlateau)

---

## 📊 Evaluation Metrics

- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix
- ROC-AUC (multi-class)
- Precision–Recall Curve

All outputs are saved under the `outputs/` directory.

---

## 📈 Visualization

- Training vs Validation Loss
- Training vs Validation Accuracy
- Confusion Matrix Heatmap
- ROC Curve (One-vs-Rest)
- Precision–Recall Curve
- TensorBoard support for model graph & predictions

---

## 🔍 Key Features

- End-to-end supervised CNN pipeline
- Balanced sampling strategy
- Robust NHATS-specific label engineering
- GPU-ready PyTorch implementation
- Reproducible Colab workflow

---

## 📚 References

- NHATS Public Use Dataset
- MADSmilestone2 (Ian Byrne)
- PyTorch Documentation

---

## 👤 Author

**Aqib**  
AI / ML Researcher  
Medical Imaging & Multimodal Learning
