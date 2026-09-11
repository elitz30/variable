# Task: Test and Run Northstar

## Status & Discovery
- Current working directory: `c:\Users\Hitesh.K\OneDrive\Documents\chat-gpt\northstar_python`
- When you typed `--high-quality`, PowerShell threw a syntax error (`Missing expression after unary operator '--'`) because `--` is a decrement operator in PowerShell unless passed as an argument to an executable (e.g. `python app.py --high-quality camera`).
- When running `python app.py camera`, it fails because the required Python ML libraries (`onnxruntime`, `insightface`, `ultralytics`, `torch`) are not yet installed in the current environment.
- Python 3.11 is available on this machine (`py -3.11`).

## Plan / Todo
- [ ] Clarify which Northstar version you want to test (Python companion or Web browser app)
- [ ] Set up Python 3.11 virtual environment (`py -3.11 -m venv .venv`) and install `requirements.txt` (if testing Python companion)
- [ ] Verify dependencies import correctly (torch, onnxruntime, ultralytics, cv2, sounddevice)
- [ ] Run a test command of `app.py` (e.g. `python app.py camera` or test face/voice enrol)
- [ ] Review security and code integrity

## Review
*(To be populated after testing)*
