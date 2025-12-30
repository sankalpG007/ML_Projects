# 🦴 Bone Tumor Detection Using Deep Learning (TensorFlow)

An end‑to‑end Machine Learning project that detects **benign vs malignant bone tumors** from X‑ray images using **Convolutional Neural Networks (CNN)** and **Transfer Learning (EfficientNetB0)**.

This project supports:

* Preprocessing of medical X‑ray images
* Training with separate **train / valid / test** dataset splits
* Model evaluation and classification metrics
* Single‑image inference
* Grad‑CAM heatmap visualization (model explainability)
* Google Colab ready notebook

---

# 📁 Project Structure

```
Bone Tumor Detection using ML/
│
├── requirements.txt
├── tumor/                      # Your dataset (train/valid/test)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── data/
│   └── processed/             # Preprocessed dataset (auto‑generated)
│
├── src/
│   ├── preprocess.py
│   ├── dataset_utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── notebooks/
│   └── 01-colab-train-and-gradcam.ipynb
│
└── models/
    └── checkpoints/           # Saved best/final model files
```

---

# 📦 Installation

## 1️⃣ Create a virtual environment (recommended)

**Windows:**

```
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
```

## 2️⃣ Install required dependencies

```
pip install -r requirements.txt
```

If TensorFlow fails to install, run:

```
pip install tensorflow==2.10
```

---

# 📂 Dataset Setup (IMPORTANT)

You downloaded the **Bone Cancer Detection Dataset (ziya07)** from Kaggle.
It already contains **train / valid / test** folders.

Structure example:

```
tumor/
├── train/
│   ├── benign/
│   └── malignant/
├── valid/
├── test/
```

This is exactly what our project needs.

---

# 🧼 Step 1 — Preprocess the dataset

Run these commands from your project root:

```
python src/preprocess.py --input_dir tumor/train --output_dir data/processed/train --img_size 224
python src/preprocess.py --input_dir tumor/valid --output_dir data/processed/valid --img_size 224
python src/preprocess.py --input_dir tumor/test  --output_dir data/processed/test  --img_size 224
```

This cleans, crops, resizes images to 224×224.

---

# 🤖 Step 2 — Train the Model

(Updated `train.py` already supports train/valid/test split)

```
python src/train.py --data_dir data/processed --epochs 10 --batch_size 16
```

During training, the script:

* Loads train/valid datasets
* Uses **EfficientNetB0** transfer learning
* Saves best model to `models/checkpoints/best.h5`
* Auto‑evaluates on the test set at the end

---

# 📊 Step 3 — Evaluate Model Performance

```
python src/evaluate.py --model models/checkpoints/best.h5 --data_dir data/processed/test
```

Outputs:

* Accuracy
* Precision / Recall / F1
* Confusion Matrix
* Classification Report

---

# 🔍 Step 4 — Run Inference on a Single Image

```
python src/inference.py --model models/checkpoints/best.h5 --image path/to/image.jpg
```

Output example:

```
Probability of tumor: 0.8731
Prediction: MALIGNANT / TUMOR
```

---

# 🔥 Step 5 — Grad‑CAM Visualization (Explainability)

Use the Colab notebook:

```
notebooks/01-colab-train-and-gradcam.ipynb
```

It generates:

* Heatmaps showing **which part of the X‑ray the model focused on**
* Correct/incorrect prediction visual explanations

---

# 🏁 Summary

| Step           | Command         | Purpose                    |
| -------------- | --------------- | -------------------------- |
| Preprocess     | `preprocess.py` | Clean & Resize images      |
| Train Model    | `train.py`      | Train EfficientNetB0 model |
| Evaluate       | `evaluate.py`   | Accuracy + Metrics         |
| Inference      | `inference.py`  | Predict a single image     |
| Explainability | Colab Notebook  | Grad‑CAM visual heatmaps   |

---

# ⭐ Future Improvements

* Try **MobileNetV2** for faster local training
* Add **data augmentation** (flip, rotate, contrast)
* Add **Flask Web App** for real‑time tumor prediction
* Train for more epochs (20–30) for higher accuracy
* Add **cross‑validation**

---

# 📌 Author

**Sankalp Singh** — MCA AI/ML Student
Focus areas: Machine Learning, Deep Learning, Computer Vision

If you want, I can also generate:
✅ A PDF project report
✅ A presentation (PPT)
✅ A GitHub‑ready description + badges

Just tell me *"Generate report"* or *"Generate GitHub README"*. 🚀
