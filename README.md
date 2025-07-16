# 🧪 Skin Cancer Detection using CNN & Streamlit

A deep learning-based web application to classify **skin cancer** as **Benign** or **Malignant** using Convolutional Neural Networks (CNNs) and deployed using **Streamlit**.

---

## 📌 Project Overview

Skin cancer is one of the most common cancers worldwide. Early detection is crucial for effective treatment. This project uses image classification techniques with a CNN model trained on a dataset of skin lesion images to automate the detection of malignant and benign skin conditions.

---

## 🧠 Model Architecture

The model is built using TensorFlow and Keras and follows a custom CNN architecture with the following layers:

- **Conv2D + MaxPooling** (3 blocks: 32, 64, 128 filters)
- **Flatten Layer**
- **Dense Layer (512 neurons + Dropout 0.5)**
- **Final Output Layer (Sigmoid activation)** for binary classification

### 🔧 Model Compilation
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy  

---

## 📂 Dataset Structure

Organize your dataset as follows:

data\
├── train\
│ ├── benign\
│ └── malignant\
└── test\
├── benign\
└── malignant


Each class folder should contain respective images.

---

## 📊 Training Configuration

- **Image Size:** 224x224  
- **Batch Size:** 32  
- **Epochs:** 30  
- **EarlyStopping:** Enabled (patience=5)  
- **ReduceLROnPlateau:** Enabled (factor=0.3, patience=3)  
- **Data Augmentation:** Rotation, shift, shear, zoom, flip  

---

## 📈 Training History

After training, the model achieves good validation accuracy and balanced performance on both classes. A confusion matrix and classification report are plotted for evaluation.

---

## ✅ Predictions

The model takes an image as input and returns:
- **Prediction:** "Benign" or "Malignant"
- **Visualization:** Displays the image with the prediction label

---

## 💻 Streamlit App

You can run the web app with Streamlit for easy deployment and usage.

### 📦 Required Libraries

```bash
pip install streamlit tensorflow pillow
```

## 💾 Model Saving

```bash
skin_cancer_cnn_new.keras
```
