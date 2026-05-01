# 🌲 Deep Learning Covertype Classification

## 📌 Overview

This project implements a complete deep learning pipeline for large-scale **tabular multi-class classification** using the Forest Covertype dataset. The goal is to classify forest cover types based on 54 cartographic features.

The project focuses on **scalable experimentation, reproducibility, and deployment**, following best practices in deep learning.

---

## 🚀 Key Features

* Custom PyTorch Dataset for efficient data handling
* Configurable Multi-Layer Perceptron (MLP) model
* Manual batch-wise training loop
* 30 independent training experiments (different seeds)
* Statistical analysis (mean, std, best, worst accuracy)
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
* TensorBoard logging for training visualization
* Streamlit-based GUI for real-time inference

---

## 🧠 Model Architecture

* Input Layer: 54 features
* Hidden Layer 1: 256 neurons + ReLU + Dropout
* Hidden Layer 2: 128 neurons + ReLU + Dropout
* Output Layer: 7 classes

---

## 📊 Experiments

* Conducted **30 independent trials** using different random seeds
* Ensured robustness and reproducibility
* Aggregated results to evaluate model stability

---

## 🖥️ Deployment

A Streamlit web application allows users to:

* Input feature values manually
* Get predicted class probabilities in real time

Run the app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
deep-learning-covertype-classification/
│
├── notebook.ipynb        # Training & experiments
├── model.py              # Neural network architecture
├── app.py                # Streamlit GUI
├── best_model_seed_0.pt  # Saved model weights
├── requirements.txt      # Dependencies
└── README.md
```

---

## 📦 Dataset

Dataset is not included due to size.
Download from:
https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset

---

## ⚙️ Technologies Used

* Python
* PyTorch
* NumPy, Pandas
* Scikit-learn
* TensorBoard
* Streamlit

---

## 🎯 Learning Outcomes

* Built end-to-end deep learning pipeline
* Performed large-scale experimentation
* Applied statistical evaluation techniques
* Deployed model with user interface

---

## 👩‍💻 Author
Nimra Jabbar

Nimra Khan
# deep-learning-covertype-classification
