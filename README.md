# Plant Disease Detection Using CNN 🌿

A Convolutional Neural Network (CNN) based image classification project built using **TensorFlow** and **Keras**
to detect plant leaf diseases from image data.

This project (code + example dataset) was created by **Fazin Firoz** as part of my AI/ML portfolio.

## 🔧 Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib (for training visualization)

## 📂 Dataset
Included in this repository:

- `data/healthy/`   → sample images of healthy leaves
- `data/diseased/`  → sample images of diseased leaves
- `dataset.csv`     → file mapping `image_path` to `label`

> Note: These are small synthetic sample images meant for demo and structure only.
> In a real project, you would replace them with actual labeled plant images.

## ▶️ How to Train
```bash
python train.py
```

This will:
- Load images from the `data/` directory
- Build a small CNN model
- Train it on the sample dataset
- Print training metrics
