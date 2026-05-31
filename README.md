# 🔥 Wildfire Detection Using Deep Learning Convolutional Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.26.0-orange.svg)](https://pytorch.org/vision/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project for automated wildfire detection from images using Convolutional Neural Networks (CNNs). Built with PyTorch, this system classifies images as either **fire** or **non-fire** to assist in early wildfire detection and monitoring.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Wildfires pose a significant threat to ecosystems, property, and human life. Early detection is critical for effective response. This project leverages deep learning to automatically detect the presence of fire in images, providing a scalable solution for monitoring systems.

**Key Features:**
- ✅ Binary classification: `fire_images` vs `non_fire_images`
- ✅ Custom CNN architecture with 3 convolutional blocks
- ✅ Data normalization and augmentation pipeline
- ✅ Class balancing via undersampling
- ✅ Training/validation monitoring with confusion matrix analysis
- ✅ Model checkpointing in PyTorch format

---

## 📊 Dataset

The dataset consists of labeled images organized into two classes:

| Class | Original Count | After Undersampling |
|-------|---------------|---------------------|
| 🔥 Fire Images | ~755 | 244 |
| 🌲 Non-Fire Images | ~244 | 244 |
| **Total** | **~999** | **488** |

### Data Preprocessing
- **Image Size:** Resized to `224 × 224` pixels
- **Color Mode:** Converted to RGB (3 channels)
- **Normalization:**
  - Mean: `[0.4201, 0.3118, 0.2111]`
  - Std: `[0.2819, 0.2403, 0.2240]`
- **Train/Validation Split:** 80% / 20%
- **Batch Size:** 32

### Class Balancing
The original dataset was imbalanced (significantly more fire images than non-fire images). We applied **random undersampling** to create a balanced dataset with equal representation of both classes, improving model generalization.

---

## 🧠 Model Architecture

The CNN architecture consists of 3 convolutional blocks followed by fully connected layers:

```
Sequential(
  (0): Conv2d(3 → 16, kernel=3×3, padding=1)
  (1): ReLU()
  (2): MaxPool2d(2×2, stride=2)

  (3): Conv2d(16 → 32, kernel=3×3, padding=1)
  (4): ReLU()
  (5): MaxPool2d(2×2, stride=2)

  (6): Conv2d(32 → 64, kernel=3×3, padding=1)
  (7): ReLU()
  (8): MaxPool2d(2×2, stride=2)

  (9): Flatten()
  (10): Dropout(p=0.5)
  (11): Linear(50176 → 500)
  (12): ReLU()
  (13): Dropout(p=0.5)
  (14): Linear(500 → 2)  [Output: Fire / Non-Fire]
)
```

### Model Summary
| Metric | Value |
|--------|-------|
| **Total Parameters** | 25,113,086 |
| **Trainable Parameters** | 25,113,086 |
| **Input Size** | (Batch, 3, 224, 224) |
| **Estimated Model Size** | ~100 MB |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.11.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/Ibnbuba-A360/Wildfire-Detection-Using-DeepLearning-Convolutional-Neural-Networks.git
cd Wildfire-Detection-Using-Convolutional-Neural-Networks

# Create virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install dependencies
pip install torch==2.11.0 torchvision==0.26.0
pip install numpy pandas matplotlib tqdm scikit-learn torchinfo Pillow
```

### Required Packages
```
torch>=2.11.0
torchvision>=0.26.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
scikit-learn>=1.3.0
torchinfo>=1.8.0
Pillow>=10.0.0
```

---

## 🚀 Usage

### 1. Dataset Exploration
Explore the dataset distribution and visualize sample images:

```python
from explore_dataset import sample_images, class_counts

# View random samples from each class
sample_images(data_dir, "fire_images")
sample_images(data_dir, "non_fire_images")

# Check class distribution
class_counts(dataset).plot(kind="bar")
```

### 2. Data Preprocessing & Undersampling
```python
from explore_dataset import undersample_dataset

# Balance the dataset
undersample_dataset(
    dataset_dir="data/train/dataset",
    output_dir="data/train/undersampled_dataset",
    target_count=None  # Uses minimum class count automatically
)
```

### 3. Model Training
```python
from training import train
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Load preprocessed dataset
dataset = datasets.ImageFolder(root="data/train/undersampled_dataset", transform=transform)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model, loss, and optimizer
model = build_model()  # Your CNN architecture
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
history = train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### 4. Evaluation & Prediction
```python
from training import predict, score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Evaluate on validation set
val_loss, val_accuracy = score(model, val_loader, loss_fn, device)
print(f"Validation Accuracy: {val_accuracy:.2%}")

# Generate predictions
probabilities = predict(model, val_loader, device)
predictions = torch.argmax(probabilities, dim=1)

# Confusion Matrix
cm = confusion_matrix(targets, predictions.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fire_images", "non_fire_images"])
disp.plot(cmap="Blues")
```

### 5. Save/Load Model
```python
# Save trained model
torch.save(model.state_dict(), "models/wildfire_cnn.pth")

# Load for inference
model.load_state_dict(torch.load("models/wildfire_cnn.pth"))
model.eval()
```

---

## 🏋️ Training

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross Entropy Loss |
| Epochs | 30 (initial) + 30 (fine-tuning) |
| Batch Size | 32 |
| Dropout Rate | 0.5 |
| Random Seed | 42 |

### Training Process
The model was trained in two phases:
1. **Phase 1 (Epochs 1-30):** Initial training from scratch
2. **Phase 2 (Epochs 31-60):** Continued training to refine weights

### Sample Training Log
```
Epoch: 1,  Training Loss: 0.84,  Validation Loss: 0.34,  Validation Accuracy: 0.86
Epoch: 5,  Training Loss: 0.11,  Validation Loss: 0.15,  Validation Accuracy: 0.93
Epoch: 10, Training Loss: 0.02,  Validation Loss: 0.25,  Validation Accuracy: 0.92
Epoch: 20, Training Loss: 0.02,  Validation Loss: 0.49,  Validation Accuracy: 0.93
Epoch: 30, Training Loss: 0.12,  Validation Loss: 0.21,  Validation Accuracy: 0.93
```

---

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~92-95% |
| **Training Loss (Final)** | ~0.00 |
| **Validation Loss (Final)** | ~0.21-0.47 |

### Confusion Matrix (Validation Set)

|  | Predicted Fire | Predicted Non-Fire |
|--|----------------|--------------------|
| **Actual Fire** | 49 (True Positives) | 6 (False Negatives) |
| **Actual Non-Fire** | 2 (False Positives) | 40 (True Negatives) |

**Key Metrics:**
- **Precision (Fire):** 49/51 = **96.1%**
- **Recall (Fire):** 49/55 = **89.1%**
- **Precision (Non-Fire):** 40/46 = **87.0%**
- **Recall (Non-Fire):** 40/42 = **95.2%**
- **Overall Accuracy:** 89/97 = **91.8%**

### Observations
- The model shows strong performance in distinguishing fire from non-fire scenes
- Very low false positive rate (2 non-fire images misclassified as fire)
- The balanced dataset approach effectively prevents bias toward the majority class
- Some overfitting is observed in later epochs (training loss ≈ 0 while validation loss remains ~0.3-0.5)

---

## 📁 Project Structure

```
Wildfire-Detection-Using-Convolutional-Neural-Networks/
│
├── 📂 data/
│   └── train/
│       ├── dataset/                    # Original unbalanced dataset
│       │   ├── fire_images/
│       │   └── non_fire_images/
│       └── undersampled_dataset/       # Balanced dataset (244 per class)
│           ├── fire_images/
│           └── non_fire_images/
│
├── 📂 models/
│   └── wildfire_cnn.pth                # Saved trained model weights
│
├── 📂 notebooks/
│   ├── explore_dataset.ipynb           # Data exploration & preprocessing
│   └── model_training.ipynb            # Model building & training
│
├── 📂 src/
│   ├── __init__.py
│   ├── training.py                     # Core training functions
│   ├── model.py                        # CNN architecture definition
│   └── utils.py                        # Helper functions (transforms, etc.)
│
├── 📄 README.md                        # Project documentation
├── 📄 requirements.txt                 # Python dependencies
└── 📄 LICENSE                          # MIT License
```

### Core Files Description

| File | Description |
|------|-------------|
| `training.py` | Core training loop, prediction, scoring, and evaluation functions |
| `explore_dataset.ipynb` | Dataset visualization, normalization computation, undersampling |
| `model_training.ipynb` | Model architecture definition, training execution, confusion matrix |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Potential Improvements
- [ ] Implement data augmentation (rotation, flip, color jitter)
- [ ] Add transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Deploy as web API using Flask/FastAPI
- [ ] Create real-time detection from video streams
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Optimize for mobile/edge deployment (TorchScript/ONNX)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Visionset AI** - Original project framework and methodology
- PyTorch team for the excellent deep learning framework
- Dataset contributors for wildfire and non-fire imagery

---

## 📧 Contact

**Author:** [Ibnbuba-A360](https://github.com/Ibnbuba-A360)

For questions or collaboration opportunities, please open an issue or reach out via GitHub.

---

<div align="center">
  <sub>Built with ❤️ and 🔥 detection using PyTorch</sub>
</div>
