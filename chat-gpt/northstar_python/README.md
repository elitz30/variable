# Northstar Python companion

This is the GPU-first Python version of Northstar. On a suitable Python installation it uses:

- InsightFace + ONNX Runtime CUDA for face embeddings
- YOLO11n + PyTorch CUDA for object recognition
- SpeechBrain ECAPA + PyTorch CUDA for better speaker embeddings than the browser prototype

It stores only names and numerical embeddings in `profiles.json`; it does not retain face photos or audio recordings.

## Setup

Use **Python 3.11** for now. The Python available on this machine is 3.14, for which some of these native ML packages may not have a wheel yet.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run from this folder:

```powershell
python app.py enrol-face --name Maya path\to\maya-1.jpg path\to\maya-2.jpg
python app.py record-voice --name Maya
python app.py camera
python app.py listen
```

The first run downloads public model weights. With the RTX 5060 Laptop GPU available, the compatible CUDA builds in `requirements.txt` will be preferred automatically. Use `--high-quality` only when you want more detail than the fast defaults.
