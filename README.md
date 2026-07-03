# Explainable CNNMulti-Crop Leaf Disease Classification Using Custom  and Transfer Learning: A Tomato Leaf Disease Case Study

## Overview

This project addresses explainable multi-crop leaf disease classification using custom CNN and transfer learning; tomato serves as the primary case study due to public dataset availability, while the methodology generalizes to other crops. The system classifies tomato leaf images into 10 disease/healthy categories. The system will compare a custom CNN baseline against MobileNetV2 and EfficientNetB0 transfer-learning models, and will include Grad-CAM explainability, robustness testing, and a Streamlit web interface. Training will use a public Kaggle tomato leaf disease dataset. Implementation is in progress; features are being built incrementally.

## Disease Classes (10)

1. Tomato Healthy
2. Tomato Bacterial Spot
3. Tomato Early Blight
4. Tomato Late Blight
5. Tomato Leaf Mold
6. Tomato Septoria Leaf Spot
7. Tomato Spider Mites
8. Tomato Target Spot
9. Tomato Yellow Leaf Curl Virus
10. Tomato Mosaic Virus



## Tech Stack

Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, OpenCV/Pillow, Streamlit. Platform: Kaggle Notebook (GPU). Report: Overleaf/LaTeX. Version control: GitHub.

## Project Structure

```
tomato-guard-cnn/
├── README.md
├── HANDOVER.md
├── app.py
├── requirements.txt
├── label_map.json
├── .gitignore
├── notebook/
│   └── tomato_leaf_disease_cnn_kaggle.ipynb
├── results/
│   └── .gitkeep
├── report/
│   └── .gitkeep
└── presentation/
    └── presentation_video_link.txt
```



## How to Run



### Kaggle Notebook (training)

Before installing anything, check the runtime TensorFlow version:

```python
import tensorflow as tf; print(tf.__version__)
```

Kaggle's GPU image ships TensorFlow preinstalled. Do NOT reinstall TensorFlow unless the runtime version is incompatible with the code — reinstalling can cause long installs or CUDA mismatches. The pinned `requirements.txt` is primarily for the local Streamlit environment.

- Upload/attach the tomato leaf disease dataset under /kaggle/input
- Open notebook/tomato_leaf_disease_cnn_kaggle.ipynb
- Run all cells; outputs are saved to /kaggle/working/results and zipped



### Local Streamlit App (inference)

Place `best_model.keras` and `label_map.json` in the repo root.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```



## Status

Note: results (accuracy, confusion matrix, etc.) will be filled in ONLY from the executed notebook — no placeholder numbers. Grad-CAM explainability, robustness testing, and the Streamlit interface are planned features and not yet implemented.