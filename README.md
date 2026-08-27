# Cat vs Jet: CNN Image Classification 🐱✈️

A custom Convolutional Neural Network (CNN) built from scratch using **PyTorch** to classify images into two categories: **Cat** and **Jet (Airplane)**.

## 🚀 Key Features

* **Built from Scratch:** No pre-trained models or transfer learning were used. The convolution and pooling layers were custom-designed.
* **CUDA / GPU Optimization:** The training and testing processes automatically detect and utilize available hardware accelerators, such as NVIDIA RTX GPUs.
* **Modular Architecture:** Data preparation, model architecture, training, and testing are separated into independent modules following clean software engineering principles.
* **Image Classification:** The model is specifically designed to distinguish between cats and jet airplanes.

## 📁 Project Structure

The project consists of four core Python modules:

| File           | Description                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `data_prep.py` | Loads images from disk, resizes them to **64×64**, and creates PyTorch `DataLoader` batches with a batch size of **32**.                  |
| `model.py`     | Contains the CNN architecture through the `CatVsJetCNN` class, including the forward pass and feature extraction layers.                  |
| `train.py`     | Handles model training using the **Adam optimizer** and **Cross-Entropy Loss**, then saves the trained weights to `cat_vs_jet_model.pth`. |
| `test.py`      | Loads the trained model, evaluates it on unseen test data, and calculates the final classification accuracy.                              |

## 🛠️ Technologies Used

* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Libraries:** `torch`, `torchvision`
* **GPU Acceleration:** CUDA
* **Optimizer:** Adam
* **Loss Function:** Cross-Entropy Loss

## ⚙️ How to Run

### 1. Train the Model

Run the training script to train the CNN and generate the trained `.pth` model file:

```bash
python train.py
```

After training is completed, the model weights will be saved as:

```text
cat_vs_jet_model.pth
```

### 2. Test the Model

After training, evaluate the model using the unseen test dataset:

```bash
python test.py
```

The test script loads the trained weights and calculates the final classification accuracy.

## 🧠 Model Workflow

The overall workflow of the project is:

```text
Input Images
     ↓
Data Preparation
     ↓
Image Resizing (64×64)
     ↓
PyTorch DataLoader
     ↓
Custom CNN
     ↓
Feature Extraction
     ↓
Classification
     ↓
Cat / Jet
```

## 📊 Classification Categories

The model performs binary image classification:

* 🐱 **Cat**
* ✈️ **Jet (Airplane)**

## 💻 GPU Support

If a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available, the project automatically utilizes GPU acceleration to improve training and testing performance.

Otherwise, the model can run on the CPU.

## 👨‍💻 Developer

**Ebubekir DOĞAN**

---
