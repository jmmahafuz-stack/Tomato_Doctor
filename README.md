# Tomato Doctor

A Flask web app for classifying tomato leaf diseases with a ResNet18 PyTorch checkpoint.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py app.py
```

Open http://127.0.0.1:5000 and upload a leaf image.

Place `tomato_disease_resnet18.pth` in the project root. The checkpoint should contain a ResNet18 state dictionary with 10 output classes matching `CLASS_NAMES` in `app.py`.
