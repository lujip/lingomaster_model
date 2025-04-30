# 🧠 LingoMaster Model

[![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)

This repository contains the **training scripts, evaluation engine**, and **AI model weights** used in the LingoMaster app. The system evaluates user pronunciation and intonation using MFCC-based similarity scoring with deep learning.

---

## 📌 Description

The backend of LingoMaster is responsible for:

- Extracting MFCC features from speech input.
- Training a phrase-specific feedforward neural network (FFNN).
- Saving model weights (`.pth`) for later evaluation.
- Comparing user-input audio to reference data using cosine similarity and score thresholding.

The goal is to provide real-time, meaningful feedback on user pronunciation accuracy.

---

## 🗃️ Dataset

- **VCTK Corpus (Version 0.92)**  
A high-quality English speech dataset with recordings from 109 speakers with various accents.  
📎 [Dataset Link](https://datashare.ed.ac.uk/handle/10283/3443)

---

## 🛠️ Tech Stack

| Task           | Tool/Library     |
|----------------|------------------|
| Feature Extraction | `librosa`, `scipy`  |
| Model Training      | `PyTorch`            |
| Audio Handling     | `torchaudio`, `wave` |
| API Serving (if used) | `Render`|

---

## 🔍 Model Details

- **Architecture:** Feedforward Neural Network (FFNN)
- **Input:** MFCC vectors
- **Output:** Similarity score
- **Loss Function:** MSE or CosineSimilarity (customizable)
- **Training:** Each phrase is trained separately to maintain evaluation accuracy

---

## 🔁 How It Works

1. Input audio is trimmed, cleaned, and converted into MFCCs.
2. A pre-trained model corresponding to the input phrase is loaded.
3. The model evaluates similarity between the user's MFCC and the trained vector.
4. A score is returned with a comment (e.g., "Good", "Needs Improvement") based on thresholds.
