Skin Disease Classifier — FastAPI Web Application
==================================================

Description
-----------
A clinical decision support tool for automated skin disease classification
using a hybrid ConvNeXt-Small + Swin-Small (AFF gate) deep learning model.
Trained on the Kaggle Skin Diseases Image Dataset (27,153 images, 10 classes).

Requirements
------------
Python 3.10 or higher
All dependencies listed in requirements.txt

Installation
------------
1. Install dependencies:
   pip install -r requirements.txt

2. Place best_model_hybrid.pth in the same folder as main.py

Running the App
---------------
From inside the SkinApp folder, run:
   uvicorn main:app --reload

Then open your browser and go to:
   http://127.0.0.1:8000

Endpoints
---------
GET  /          Upload interface (HTML)
POST /predict   Submit image for classification
POST /flag      Flag a prediction as incorrect
GET  /health    Check server and model status
GET  /docs      Auto-generated API documentation

Files
-----
main.py              FastAPI application and routing
inference.py         Model loading and prediction logic
templates/index.html Frontend interface
requirements.txt     Python dependencies
best_model_hybrid.pth Trained model weights (not included in submission)

Model
-----
Architecture : ConvNeXt-Small + Swin-Small with Attentional Feature Fusion gate
Pretrained   : ImageNet-22k fine-tuned on ImageNet-1k
Dataset      : Kaggle Skin Diseases Image Dataset
Classes      : 10 skin disease categories
Accuracy     : 90.27% ensemble + TTA (test set)

Disease Classes
---------------
0. Eczema
1. Warts / Molluscum
2. Melanoma
3. Atopic Dermatitis
4. Basal Cell Carcinoma
5. Melanocytic Nevi
6. Benign Keratosis
7. Psoriasis / Lichen Planus
8. Seborrheic Keratoses
9. Tinea / Candidiasis

Risk Classification
-------------------
HIGH   : Melanoma, Basal Cell Carcinoma
MEDIUM : Eczema, Atopic Dermatitis, Psoriasis / Lichen Planus
LOW    : All other classes

Disclaimer
----------
This tool is for clinical decision support only and does not constitute
a medical diagnosis. All predictions must be reviewed by a qualified
healthcare professional before any clinical action is taken.Skin Disease Classifier — FastAPI Web Application
==================================================