# Tomato Doctor

A Flask web app for classifying tomato leaf diseases with a ResNet18 PyTorch checkpoint.

## Checkpoint status

The bundled checkpoint currently has a 30-output ResNet18 head but only 10 entries in
`class_to_idx`, all for tomato classes at indices 20 through 29. Because 20 outputs
are unlabeled, the app deliberately refuses to load this checkpoint rather than
returning unreliable predictions. Use a checkpoint whose `class_to_idx` contains one
label for every output index.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py app.py
```

Open http://127.0.0.1:5000 and upload a leaf image.

Place `tomato_disease_resnet18.pth` in the project root. The checkpoint should contain
a ResNet18 state dictionary and a complete `class_to_idx` mapping. The number of
labels must match the final layer's output count.
