# 🧠 Early Brain Tumor Detection, Segmentation and Classification with Explainable AI

An advanced **deep learning–based medical imaging system** designed to detect and classify brain tumors from MRI scans using **ensemble CNN architectures and Explainable AI techniques**.

The system combines **DenseNet201, InceptionV3, and EfficientNetB3** with **Grad-CAM visualization** to provide accurate and interpretable predictions for clinical decision support.

---

# 🚀 Project Overview

Brain tumor diagnosis using MRI scans requires expert radiological analysis and can be time-consuming. This project proposes an **AI-powered diagnostic system** that automatically detects and classifies tumors while providing visual explanations of model decisions.

The proposed framework integrates multiple deep learning models into a **soft-voting ensemble**, improving diagnostic accuracy and robustness compared to individual models.

---

# 🎯 Objectives

* Develop a **deep learning pipeline for brain tumor detection**
* Classify MRI images into **four tumor categories**
* Improve accuracy using **ensemble deep learning**
* Integrate **Explainable AI (Grad-CAM)** for interpretability
* Provide an **AI-assisted diagnostic support tool**

---

# 🧬 Tumor Classes

The model classifies MRI images into the following categories:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

---

# 🏗 System Architecture

```
MRI Image
   │
   ▼
Image Preprocessing
(Resizing, Normalization, Augmentation)
   │
   ▼
Deep Learning Models
(DenseNet201, InceptionV3, EfficientNetB3)
   │
   ▼
Soft Voting Ensemble
   │
   ▼
Tumor Classification
   │
   ▼
Grad-CAM Visualization
```

---

# 🧠 Deep Learning Models

### DenseNet201

* Dense connectivity between layers
* Strong feature reuse
* Stable gradient propagation

### InceptionV3

* Multi-scale feature extraction
* Effective for irregular tumor shapes

### EfficientNetB3

* Compound scaling architecture
* High accuracy with optimized computation

The outputs of these models are combined using **ensemble learning** to improve performance.

---

# 🤖 Ensemble Learning

Instead of relying on a single model, predictions from multiple CNN architectures are combined.

```
Final Prediction =
0.5 × DenseNet201
+ 0.3 × InceptionV3
+ 0.2 × EfficientNetB3
```

This improves:

* Prediction stability
* Model robustness
* Overall accuracy

---

# 📊 Model Performance

| Model          | Accuracy   |
| -------------- | ---------- |
| DenseNet201    | ~97.4%     |
| InceptionV3    | ~97%       |
| EfficientNetB3 | ~99%       |
| Ensemble Model | **99.01%** |

The **ensemble model achieved the best performance**, demonstrating improved classification across all tumor classes.

---

# 🔍 Explainable AI (Grad-CAM)

Deep learning models often behave like **black boxes**.

To improve transparency, the system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM generates heatmaps that:

* Highlight tumor-relevant regions
* Explain model predictions
* Increase trust in AI-assisted diagnosis

These visual explanations help clinicians understand **why the model made a particular prediction**.

---

# 🛠 Tech Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn

### Computer Vision

* OpenCV

### Visualization

* Matplotlib
* Seaborn

### Development Tools

* Jupyter Notebook
* VS Code
* Git & GitHub

---

# ⚙️ System Requirements

### Hardware

* Intel i5/i7 processor
* Minimum **8 GB RAM**
* GPU recommended for model training

### Software

* Python 3.10+
* TensorFlow / Keras
* OpenCV
* Scikit-learn

---

# 📂 Project Workflow

```
Dataset Collection
      │
      ▼
Data Preprocessing
      │
      ▼
CNN Model Training
(DenseNet201, InceptionV3, EfficientNetB3)
      │
      ▼
Model Evaluation
      │
      ▼
Ensemble Learning
      │
      ▼
Explainable AI (Grad-CAM)
      │
      ▼
Tumor Classification & Visualization
```

---

# 🔮 Future Improvements

Potential future improvements include:

* **3D MRI analysis using 3D CNN models**
* Real-time hospital deployment
* Integration with medical imaging systems (PACS)
* Edge AI deployment for low-resource hospitals
* Mobile and clinical dashboard applications

---

# 👨‍💻 Author

**Yashwanth Hudumula**

MS in Computer Science
Blekinge Institute of Technology, Sweden

---

# 📜 License

This project is developed for **research and educational purposes**.

---

⭐ If you find this project useful, consider giving it a **star on GitHub**.
