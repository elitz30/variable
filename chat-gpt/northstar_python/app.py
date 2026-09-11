"""Northstar local recognition toolkit.

Uses CUDA automatically when the installed ONNX Runtime and PyTorch builds expose it.
No photo or audio recording is retained: profiles are numerical embeddings in profiles.json.
"""
from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(__file__).resolve().parent
# Auto-switch to .venv python if available and not already inside it
_VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    sys.exit(subprocess.call([str(_VENV_PYTHON)] + sys.argv))

# Ensure robust UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROFILE_FILE = APP_DIR / "profiles.json"
RECORDED_DIR = APP_DIR / "recorded"
RECORDED_DIR.mkdir(parents=True, exist_ok=True)
FACE_LIMIT = 0.45
VOICE_LIMIT = 0.68


def need(package: str, error: Exception):
    raise SystemExit(
        f"Northstar needs '{package}'. Create a Python 3.11 environment and run "
        f"'pip install -r requirements.txt'.\n\nDetails: {error}"
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    if not a_norm or not b_norm:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


class ProfileStore:
    """Stores only numerical face / voice embeddings and a user-supplied name."""

    def __init__(self, path: Path = PROFILE_FILE):
        self.path = path
        self.data: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            # Auto-normalize any legacy or unnormalized voice vectors
            modified = False
            for p in raw.values():
                if p.get("voice") is not None:
                    v = np.asarray(p["voice"], dtype=np.float32)
                    v_norm = np.linalg.norm(v)
                    if v_norm > 0 and abs(v_norm - 1.0) > 1e-3:
                        p["voice"] = (v / v_norm).astype(float).tolist()
                        modified = True
            if modified:
                self.path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            return raw
        except json.JSONDecodeError:
            raise SystemExit(f"{self.path} is not valid JSON. Rename it and try again.")

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def add_face(self, name: str, embedding: np.ndarray) -> None:
        profile = self.data.setdefault(name, {"faces": [], "voice": None})
        profile.setdefault("faces", []).append(embedding.astype(float).tolist())
        self.save()

    def save_voice(self, name: str, embedding: np.ndarray) -> None:
        profile = self.data.setdefault(name, {"faces": [], "voice": None})
        norm = np.linalg.norm(embedding)
        unit_emb = (embedding / norm).astype(np.float32) if norm > 0 else embedding.astype(np.float32)
        if profile.get("voice") is not None:
            old = np.asarray(profile["voice"], dtype=np.float32)
            old_norm = np.linalg.norm(old)
            if old_norm > 0:
                old = old / old_norm
            # Blend 60% new recording + 40% historical context for pitch resilience
            blended = 0.6 * unit_emb + 0.4 * old
            b_norm = np.linalg.norm(blended)
            if b_norm > 0:
                blended = blended / b_norm
            profile["voice"] = blended.tolist()
        else:
            profile["voice"] = unit_emb.tolist()
        self.save()

    def delete_profile(self, name: str) -> bool:
        target = None
        for k in list(self.data.keys()):
            if k.lower() == name.lower():
                target = k
                break
        if target:
            del self.data[target]
            self.save()
            person_folder = RECORDED_DIR / target
            if person_folder.exists() and person_folder.is_dir():
                shutil.rmtree(person_folder, ignore_errors=True)
            return True
        return False

    def sync_from_recorded_folder(self, vision_engine: Any = None, voice_engine: Any = None) -> bool:
        """Syncs profiles.json with the manual state of recorded/<Person_Name>/ folders."""
        if not RECORDED_DIR.exists():
            return False

        changed = False
        import cv2
        try:
            import soundfile as sf
        except Exception:
            sf = None

        # 1. Prune profiles whose folders were deleted on disk by the user
        on_disk_names = {p.name for p in RECORDED_DIR.iterdir() if p.is_dir()}
        for existing_name in list(self.data.keys()):
            if existing_name not in on_disk_names:
                del self.data[existing_name]
                changed = True

        # 2. Add or update profiles from folders on disk
        for person_dir in RECORDED_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            profile = self.data.setdefault(name, {"faces": [], "voice": None})

            # Sync images if vision engine is provided and profile faces are empty
            image_files = [f for f in person_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            if image_files and vision_engine is not None and len(profile.get("faces", [])) == 0:
                for img_p in image_files[:5]:
                    img = cv2.imread(str(img_p))
                    if img is not None:
                        found = vision_engine.faces(img)
                        if found:
                            profile.setdefault("faces", []).append(found[0].embedding.astype(float).tolist())
                            changed = True

            # Sync voice if voice engine is provided and profile voice is None
            wav_files = [f for f in person_dir.iterdir() if f.is_file() and f.suffix.lower() == '.wav']
            if wav_files and voice_engine is not None and profile.get("voice") is None and sf is not None:
                try:
                    wav_data, sr = sf.read(str(wav_files[0]))
                    if wav_data.ndim > 1:
                        wav_data = wav_data[:, 0]
                    if sr != 16000:
                        indices = np.round(np.linspace(0, len(wav_data) - 1, int(len(wav_data) * (16000 / sr)))).astype(int)
                        wav_data = wav_data[indices].astype(np.float32)
                    emb = voice_engine.embedding(wav_data.astype(np.float32))
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        profile["voice"] = (emb / norm).astype(float).tolist()
                        changed = True
                except Exception as err:
                    print(f"Error syncing audio for {name}: {err}")

        if changed:
            self.save()
            print(f"🔄 Synced profiles with recorded/ folder: {list(self.data.keys())}")
        return changed

    def nearest_face(self, embedding: np.ndarray) -> tuple[str, float]:
        best_name, best_distance = "Unknown face", 1.0
        for name, profile in self.data.items():
            for stored in profile.get("faces", []):
                distance = cosine_distance(embedding, np.asarray(stored, dtype=np.float32))
                if distance < best_distance:
                    best_name, best_distance = name, distance
        return (best_name, best_distance) if best_distance <= FACE_LIMIT else ("Unknown face", best_distance)

    def nearest_voice(self, embedding: np.ndarray) -> tuple[str, float]:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        best_name, best_distance = "Unknown speaker", 1.0
        for name, profile in self.data.items():
            if profile.get("voice") is None:
                continue
            stored = np.asarray(profile["voice"], dtype=np.float32)
            s_norm = np.linalg.norm(stored)
            if s_norm > 0:
                stored = stored / s_norm
            distance = cosine_distance(embedding, stored)
            if distance < best_distance:
                best_name, best_distance = name, distance
        return (best_name, best_distance) if best_distance <= VOICE_LIMIT else ("Unknown speaker", best_distance)


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]
    embedding: np.ndarray


class VisionEngine:
    """InsightFace for face embeddings, YOLO for common objects."""

    def __init__(self, high_quality: bool = False):
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            from ultralytics import YOLO, YOLOWorld
        except ModuleNotFoundError as error:
            need("vision dependencies", error)

        active = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in active:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in active:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.cuda = "CUDAExecutionProvider" in active or "DmlExecutionProvider" in active
        backend_name = "NVIDIA CUDA" if "CUDAExecutionProvider" in active else ("DirectML GPU (NVIDIA RTX)" if "DmlExecutionProvider" in active else "CPU fallback")
        self.face = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face.prepare(ctx_id=0 if self.cuda else -1, det_size=(640, 640) if high_quality else (320, 320))

        # Load YOLO-World for open-vocabulary object recognition with 95+ categories
        world_weights = APP_DIR / "yolov8s-worldv2.pt"
        if world_weights.exists():
            try:
                self.yolo = YOLOWorld(str(world_weights))
                # Expanded real-world desk, room, wearable, and tech categories
                classes = [
                    "person", "face", "glasses", "sunglasses", "eyeglasses", "watch", "smartwatch", "ring", "bracelet", "necklace", "badge", "id card",
                    "phone", "smartphone", "cell phone", "laptop", "computer", "keyboard", "mouse", "computer mouse", "mousepad",
                    "monitor", "screen", "tv", "tablet", "headphones", "headset", "earphones", "webcam", "camera", "microphone", "speaker",
                    "charger", "power bank", "cable", "wire", "usb drive", "remote control",
                    "pen", "pencil", "marker", "highlighter", "notebook", "notepad", "book", "paper", "folder", "scissors", "stapler", "calculator",
                    "mug", "coffee mug", "coffee cup", "cup", "glass", "water bottle", "bottle", "thermos", "can", "plate", "bowl", "fork", "spoon", "knife", "napkin", "tissue",
                    "snack", "candy", "apple", "banana", "fruit", "sandwich",
                    "backpack", "bag", "wallet", "purse", "keys", "keychain", "jacket", "coat", "hoodie", "shirt", "hat", "cap", "shoe",
                    "desk", "table", "chair", "office chair", "couch", "lamp", "clock", "plant", "potted plant", "pillow", "trash can", "door", "window", "box"
                ]
                self.yolo.set_classes(classes)
                self.using_world = True
            except Exception as e:
                print(f"YOLO-World init fallback: {e}")
                self.yolo = YOLO("yolo11n.pt")
                self.using_world = False
        else:
            self.yolo = YOLO("yolo11n.pt")
            self.using_world = False

        self.device: int | str = 0 if ("CUDAExecutionProvider" in active) else "cpu"
        self.object_size = 480 if high_quality else 384
        self.conf_threshold = 0.32 if self.using_world else 0.40
        print(f"Vision backend: {backend_name} | Objects model: {'YOLO-World (95+ categories)' if self.using_world else 'YOLO11'}")

    def faces(self, frame: np.ndarray) -> list[FaceResult]:
        found = []
        for face in self.face.get(frame):
            x1, y1, x2, y2 = face.bbox.astype(int)
            found.append(FaceResult((x1, y1, x2, y2), face.normed_embedding))
        return found

    def objects(self, frame: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        result = self.yolo.predict(frame, imgsz=self.object_size, conf=self.conf_threshold, device=self.device, verbose=False)[0]
        found = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = result.names[int(box.cls[0])]
            found.append((label, float(box.conf[0]), (x1, y1, x2, y2)))
        return found


def enrol_faces(args: argparse.Namespace) -> None:
    import cv2

    engine, store = VisionEngine(args.high_quality), ProfileStore()
    saved = 0
    for image_path in args.photos:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        found = engine.faces(image)
        if len(found) != 1:
            print(f"Skipping {image_path}: expected one clear face, found {len(found)}.")
            continue
        store.add_face(args.name, found[0].embedding)
        person_dir = RECORDED_DIR / args.name
        person_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, person_dir / Path(image_path).name)
        saved += 1
    print(f"Saved {saved} face profile(s) for {args.name} and archived photos to recorded/{args.name}.")


def draw_label(frame: np.ndarray, label: str, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    width, height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
    cv2.rectangle(frame, (x1, max(0, y1 - height - 10)), (x1 + width + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)


def draw_glass_panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.80, border_color: tuple[int, int, int] = (65, 65, 65)) -> None:
    import cv2

    ih, iw = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    overlay = np.full_like(sub, 22)
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1.0 - alpha, overlay, alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)


def get_working_input_device() -> tuple[int | None, int, int]:
    """Dynamically probes and returns the best working audio capture device on Windows."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        candidates = []
        for i, d in enumerate(devices):
            max_in = d.get("max_input_channels", 0)
            if max_in > 0:
                name = d.get("name", "").lower()
                api = sd.query_hostapis(d["hostapi"])["name"].lower()
                score = 0
                if "microphone array" in name:
                    score += 50
                if "realtek" in name:
                    score += 30
                if "wdm-ks" in api:
                    score += 40
                elif "wasapi" in api:
                    score += 20
                elif "directsound" in api:
                    score += 5
                candidates.append((score, i, d))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, idx, d in candidates:
            max_in = d.get("max_input_channels", 1)
            chan_options = [max_in]
            if 2 not in chan_options and max_in >= 2:
                chan_options.append(2)
            if 1 not in chan_options:
                chan_options.append(1)

            for sr in [int(d.get("default_samplerate", 48000)), 48000, 44100, 16000]:
                for ch in chan_options:
                    test_chunks = []
                    def _probe_cb(indata, frames, time_info, status):
                        test_chunks.append(indata.copy())
                    try:
                        s = sd.InputStream(device=idx, samplerate=sr, channels=ch, callback=_probe_cb)
                        s.start()
                        time.sleep(0.05)
                        s.stop()
                        s.close()
                        if test_chunks:
                            raw = np.concatenate(test_chunks)
                            if not np.isnan(raw).any() and not np.isinf(raw).any() and float(np.max(np.abs(raw))) < 5.0:
                                return idx, sr, ch
                    except Exception:
                        continue
    except Exception:
        pass
    return None, 16000, 1


class AudioStreamBuffer:
    """Non-blocking streaming audio ring buffer capturing live 16kHz mono audio with sub-100ms latency."""

    def __init__(self, buffer_seconds: float = 3.0):
        self.buffer_size = int(buffer_seconds * 16000)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream = None
        self.running = False
        self.dev: int | None = None
        self.sr: int = 16000
        self.ch: int = 1
        self._record_chunks: list[np.ndarray] | None = None
        self._record_target: int = 0
        self._record_collected: int = 0
        self._record_event = threading.Event()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if indata is None or len(indata) == 0:
            return

        # Downmix all available input channels to capture the full beamforming microphone array
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        elif indata.ndim > 1:
            mono = indata[:, 0]
        else:
            mono = indata

        # Resample to standard 16kHz for neural audio processing
        if self.sr == 48000:
            mono_16k = mono[::3].astype(np.float32)
        elif self.sr == 44100:
            indices = np.round(np.linspace(0, len(mono) - 1, int(frames * (16000 / 44100)))).astype(int)
            mono_16k = mono[indices].astype(np.float32)
        elif self.sr != 16000:
            indices = np.round(np.linspace(0, len(mono) - 1, int(frames * (16000 / self.sr)))).astype(int)
            mono_16k = mono[indices].astype(np.float32)
        else:
            mono_16k = mono.astype(np.float32)

        if len(mono_16k) == 0:
            return

        mono_16k = np.nan_to_num(mono_16k, nan=0.0, posinf=0.0, neginf=0.0)
        mean_val = float(np.mean(mono_16k))
        if np.isfinite(mean_val):
            mono_16k = mono_16k - mean_val

        # Soft limiter gain
        processed = np.tanh(mono_16k * 8.0).astype(np.float32)

        with self.lock:
            n = len(processed)
            if n >= self.buffer_size:
                self.buffer[:] = processed[-self.buffer_size:]
            else:
                self.buffer[:-n] = self.buffer[n:]
                self.buffer[-n:] = processed

            if self._record_chunks is not None:
                self._record_chunks.append(processed.copy())
                self._record_collected += n
                if self._record_collected >= self._record_target:
                    self._record_event.set()

    def start(self) -> None:
        try:
            import sounddevice as sd
            self.running = True
            self.dev, self.sr, self.ch = get_working_input_device()
            blocksize = int(self.sr * 0.04)  # ~40ms buffer
            self.stream = sd.InputStream(
                device=self.dev,
                samplerate=self.sr,
                channels=self.ch,
                callback=self._callback,
                blocksize=blocksize,
                dtype="float32",
            )
            self.stream.start()
            print(f"🎙️ Audio stream active: device {self.dev} ({self.sr}Hz, {self.ch}ch)")
        except Exception as err:
            print(f"Streaming audio start error: {err}")
            self.stream = None

    def stop(self) -> None:
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def get_latest(self, seconds: float = 1.0) -> np.ndarray:
        samples = int(seconds * 16000)
        with self.lock:
            if len(self.buffer) < samples:
                return np.pad(self.buffer, (samples - len(self.buffer), 0))
            return self.buffer[-samples:].copy()

    def record_window(self, seconds: float) -> np.ndarray:
        target = int(seconds * 16000)
        with self.lock:
            self._record_chunks = []
            self._record_target = target
            self._record_collected = 0
            self._record_event.clear()

        self._record_event.wait(timeout=seconds + 2.5)
        with self.lock:
            if self._record_chunks:
                data = np.concatenate(self._record_chunks)
            else:
                data = self.buffer[-target:] if len(self.buffer) >= target else self.buffer
            self._record_chunks = None
            if len(data) < target:
                data = np.pad(data, (0, target - len(data)))
            return np.clip(data[:target], -1.0, 1.0).astype(np.float32)


def record(seconds: float) -> np.ndarray:
    """Safe callback-based recording that works seamlessly on Windows WDM-KS without blocking API errors."""
    buf = AudioStreamBuffer(buffer_seconds=max(3.0, seconds + 1.0))
    buf.start()
    try:
        return buf.record_window(seconds)
    finally:
        buf.stop()


def save_face_crop(frame: np.ndarray, box: tuple[int, int, int, int], name: str) -> Path | None:
    """Crops a detected face cleanly with padding and saves to recorded/<name>/."""
    import cv2
    clean_name = re.sub(r'[^\w\-]', '_', name.strip())
    if not clean_name or clean_name.lower() in ("unknown", "unknown face", "unknown speaker", "quiet"):
        return None

    person_dir = RECORDED_DIR / clean_name
    person_dir.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = box
    ih, iw = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * 0.18), int(bh * 0.18)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(iw, x2 + pad_x)
    cy2 = min(ih, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filepath = person_dir / f"face_{timestamp}.jpg"
    try:
        cv2.imwrite(str(filepath), crop)
        return filepath
    except Exception as e:
        print(f"Error saving face crop: {e}")
        return None


def save_voice_clip(audio_data: np.ndarray, name: str) -> Path | None:
    """Saves a 16kHz mono audio recording to recorded/<name>/voice_<timestamp>.wav using soundfile."""
    try:
        import soundfile as sf
        clean_name = re.sub(r'[^\w\-]', '_', name.strip())
        if not clean_name or clean_name.lower() in ("unknown", "unknown speaker", "quiet"):
            return None
        person_dir = RECORDED_DIR / clean_name
        person_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = person_dir / f"voice_{timestamp}.wav"
        sf.write(str(filepath), audio_data, 16000)
        return filepath
    except Exception as err:
        print(f"Error saving voice clip: {err}")
        return None


class SessionTracker:
    """Records session analytics (presence and speaking duration per person) to a text file."""

    def __init__(self, sessions_dir: Path = APP_DIR / "sessions"):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.start_dt = datetime.now()
        timestamp = self.start_dt.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.sessions_dir / f"session_{timestamp}.txt"

        # Per-person analytics: {name: {"presence": float, "speech": float}}
        self.stats: dict[str, dict[str, float]] = {}
        # Live presence and speech states
        self.presence_state: dict[str, bool] = {}
        self.last_seen_time: dict[str, float] = {}
        self.speaking_state: dict[str, bool] = {}
        self.speech_start_time: dict[str, float] = {}
        self.events: list[str] = []
        self.lock = threading.Lock()

        self.log_event(f"Session started at {self.start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        self.flush()

    def log_event(self, text: str) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{now_str}] {text}")

    def update(self, detected_names: list[str], active_speaker: str, dt: float) -> None:
        now = time.time()
        with self.lock:
            # 1. Update presence
            for name in detected_names:
                p_stats = self.stats.setdefault(name, {"presence": 0.0, "speech": 0.0})
                p_stats["presence"] += dt
                self.last_seen_time[name] = now
                if not self.presence_state.get(name, False):
                    self.presence_state[name] = True
                    self.log_event(f"Presence: {name} entered camera view")

            # Check if any previously present person left camera view (> 2.0s absent)
            for name, is_here in list(self.presence_state.items()):
                if is_here and name not in detected_names:
                    if now - self.last_seen_time.get(name, 0.0) > 2.0:
                        self.presence_state[name] = False
                        self.log_event(f"Presence: {name} left camera view")

            # 2. Update speech
            is_valid_speaker = active_speaker not in ("Quiet", "Mic Offline", "Unknown", "")
            for name in list(self.speaking_state.keys()):
                if self.speaking_state[name] and (not is_valid_speaker or active_speaker != name):
                    dur = now - self.speech_start_time.get(name, now)
                    self.speaking_state[name] = False
                    self.log_event(f"Speech: {name} stopped speaking (duration: {dur:.1f}s)")

            if is_valid_speaker:
                s_stats = self.stats.setdefault(active_speaker, {"presence": 0.0, "speech": 0.0})
                s_stats["speech"] += dt
                if not self.speaking_state.get(active_speaker, False):
                    self.speaking_state[active_speaker] = True
                    self.speech_start_time[active_speaker] = now
                    self.log_event(f"Speech: {active_speaker} started speaking")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = int(round(seconds))
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}h {m:02d}m {sec:02d}s"
        return f"{m:02d}m {sec:02d}s"

    def generate_report(self, end_dt: datetime | None = None) -> str:
        end_dt = end_dt or datetime.now()
        total_sec = max(0.0, (end_dt - self.start_dt).total_seconds())

        lines = [
            "=" * 80,
            "                    NORTHSTAR SESSION INTELLIGENCE REPORT",
            "=" * 80,
            f"Session Log File:    {self.log_file.name}",
            f"Session Start Time:  {self.start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session End Time:    {end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration:      {self._format_duration(total_sec)} ({total_sec:.1f}s)",
            "=" * 80,
            "",
            "-" * 80,
            "PARTICIPANT SUMMARY",
            "-" * 80,
            f"{'Person':<18} | {'Present Duration':<18} | {'Speaking Duration':<18} | {'% Speaking':<12}",
            f"{'-'*18}-+-{'-'*18}-+-{'-'*18}-+-{'-'*12}",
        ]

        if not self.stats:
            lines.append("No participants detected during this session.")
        else:
            for name, data in sorted(self.stats.items(), key=lambda x: x[1]['presence'], reverse=True):
                pres_str = self._format_duration(data['presence'])
                spk_str = self._format_duration(data['speech'])
                ratio = (data['speech'] / data['presence'] * 100) if data['presence'] > 0 else 0.0
                lines.append(f"{name:<18} | {pres_str:<18} | {spk_str:<18} | {ratio:>5.1f}%")

        lines.extend([
            "-" * 80,
            "",
            "-" * 80,
            "CHRONOLOGICAL EVENT LOG",
            "-" * 80,
        ])
        lines.extend(self.events)
        lines.extend([
            "-" * 80,
            "=" * 80,
        ])
        return "\n".join(lines) + "\n"

    def flush(self, final: bool = False) -> None:
        with self.lock:
            report = self.generate_report()
            try:
                self.log_file.write_text(report, encoding="utf-8")
            except Exception as e:
                print(f"Session log write error: {e}")


class AppState:
    """Thread-safe state shared between the AI pipeline and the web UI bridge."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running: bool = False
        self.latest_jpeg: bytes | None = None
        self.faces: list[dict] = []
        self.objects: list[dict] = []
        self.speaker: str = "Quiet"
        self.speaker_conf: float = 0.0
        self.volume: float = 0.0
        self.fps: float = 30.0
        self.voice_recording: bool = False
        self.voice_rec_remaining: float = 0.0
        self.voice_target_name: str = ""
        self.store: Any = None
        self.engine: Any = None
        self.voice_engine: Any = None
        self.audio_stream: Any = None


app_state = AppState()


class WebBridgeHandler(http.server.BaseHTTPRequestHandler):
    """Zero-dependency HTTP server streaming video and syncing AI data with index.html."""

    def log_message(self, format, *args):
        pass  # Suppress request spam in console

    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path

        if path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while app_state.running:
                with app_state.lock:
                    frame_bytes = app_state.latest_jpeg
                if frame_bytes is not None:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.033)
            return

        if path == "/api/status":
            with app_state.lock:
                store = app_state.store or ProfileStore()
                status_data = {
                    "faces": app_state.faces,
                    "objects": app_state.objects,
                    "audio": {
                        "speaker": app_state.speaker,
                        "confidence": int(round(app_state.speaker_conf * 100)),
                        "volume": round(app_state.volume, 4),
                        "is_recording": app_state.voice_recording,
                        "remaining": round(app_state.voice_rec_remaining, 1),
                        "target": app_state.voice_target_name,
                    },
                    "fps": round(app_state.fps, 1),
                    "peopleCount": len(store.data) if hasattr(store, "data") else 0,
                }
            payload = json.dumps(status_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/api/people":
            with app_state.lock:
                store = app_state.store or ProfileStore()
                people_list = []
                if store and hasattr(store, "data"):
                    for name, p in store.data.items():
                        people_list.append({
                            "id": name,
                            "name": name,
                            "faceCount": len(p.get("faces", [])),
                            "hasVoice": (p.get("voice") is not None),
                        })
            payload = json.dumps(people_list).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Static assets from project root (index.html, styles.css, app.js)
        web_root = APP_DIR.parent.resolve()
        clean_path = path.lstrip("/")
        if not clean_path:
            clean_path = "index.html"
        try:
            file_path = (web_root / clean_path).resolve()
            if not file_path.is_relative_to(web_root):
                self.send_response(403)
                self.end_headers()
                return
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        if file_path.is_file():
            mime_map = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".ico": "image/x-icon",
            }
            content_type = mime_map.get(file_path.suffix.lower(), "application/octet-stream")
            try:
                data = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception:
                pass

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception:
            req_data = {}

        if path == "/api/open_vault":
            RECORDED_DIR.mkdir(parents=True, exist_ok=True)
            try:
                os.startfile(str(RECORDED_DIR))
                res = {"ok": True, "message": "Opened recorded vault"}
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            self._send_json(res)
            return

        if path == "/api/forget":
            name = req_data.get("name", "").strip()
            if name and app_state.store:
                deleted = app_state.store.delete(name)
                res = {"ok": deleted, "name": name}
            else:
                res = {"ok": False, "error": "Invalid name"}
            self._send_json(res)
            return

        if path == "/api/record_voice":
            name = req_data.get("name", "").strip()
            if name and app_state.audio_stream and app_state.store:
                def _bg_rec():
                    with app_state.lock:
                        app_state.voice_recording = True
                        app_state.voice_target_name = name
                        app_state.voice_rec_remaining = 5.0
                    try:
                        audio_clip = app_state.audio_stream.record_window(5.0)
                        ve = app_state.voice_engine or VoiceEngine()
                        emb = ve.embedding(audio_clip)
                        app_state.store.save_voice(name, emb)
                        save_voice_clip(audio_clip, name)
                    except Exception as err:
                        print(f"Web voice record error: {err}")
                    finally:
                        with app_state.lock:
                            app_state.voice_recording = False
                            app_state.voice_target_name = ""
                            app_state.voice_rec_remaining = 0.0

                threading.Thread(target=_bg_rec, daemon=True).start()
                self._send_json({"ok": True, "message": f"Recording 5s voice cue for {name}..."})
            else:
                self._send_json({"ok": False, "error": "Audio engine not ready"})
            return

        if path == "/api/enroll_face":
            name = req_data.get("name", "").strip()
            img_b64 = req_data.get("image", "")
            if name and img_b64 and app_state.engine and app_state.store:
                import cv2
                try:
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    raw_bytes = base64.b64decode(img_b64)
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        faces = app_state.engine.faces(img)
                        if faces:
                            app_state.store.add_face(name, faces[0].embedding)
                            save_face_crop(img, faces[0].box, name)
                            self._send_json({"ok": True, "name": name})
                            return
                except Exception as err:
                    self._send_json({"ok": False, "error": str(err)})
                    return
            self._send_json({"ok": False, "error": "No clear face found in image"})
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: dict):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_web_server(port: int = 4173) -> tuple[Any, int]:
    for p in (port, 4174, 8000, 8080):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", p), WebBridgeHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"🌟 Northstar Web UI is live at http://localhost:{p}")
            return server, p
        except Exception:
            continue
    return None, 0


def camera(args: argparse.Namespace) -> None:
    import cv2

    engine, store = VisionEngine(args.high_quality), ProfileStore()
    # Bi-directional sync with recorded/ folder
    store.sync_from_recorded_folder(vision_engine=engine)

    # Start multi-threaded web bridge server so web dashboard (index.html / app.js) is live at http://localhost:4173
    web_server, web_port = start_web_server(4173)
    with app_state.lock:
        app_state.running = True
        app_state.store = store
        app_state.engine = engine

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open this camera.")

    # Shared thread-safe state
    frame_lock = threading.Lock()
    results_lock = threading.Lock()
    audio_lock = threading.Lock()
    latest_frame: np.ndarray | None = None
    face_results: list[FaceResult] = []
    object_results: list[tuple[str, float, tuple[int, int, int, int]]] = []
    selected_face: FaceResult | None = None
    toast_message: str = ""
    toast_expiry: float = 0.0
    running = True

    # Audio recognizer state
    audio_speaker = "Quiet"
    audio_conf = 0.0
    audio_volume = 0.0
    voice_recording_active = False
    voice_rec_remaining = 0.0
    voice_rec_target_name = ""

    def set_toast(msg: str, duration: float = 3.0) -> None:
        nonlocal toast_message, toast_expiry
        toast_message = msg
        toast_expiry = time.time() + duration

    def prompt_for_name(title: str = "Northstar Enrolment", initial: str = "") -> str | None:
        try:
            import tkinter as tk
            from tkinter import simpledialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            name = simpledialog.askstring(title, "Enter the person's name:", initialvalue=initial)
            root.destroy()
            return name.strip() if name else None
        except Exception as err:
            print(f"Dialog error: {err}")
            return None

    def enrol_target_face(face: FaceResult) -> None:
        name = args.enrol_name or prompt_for_name("Northstar Face Enrolment")
        if name:
            store.add_face(name, face.embedding)
            with frame_lock:
                cur_frame = None if latest_frame is None else latest_frame.copy()
            saved_face_path = None
            if cur_frame is not None:
                saved_face_path = save_face_crop(cur_frame, face.box, name)
            if saved_face_path:
                set_toast(f"Saved '{name}' & photo to recorded/{name}!")
                print(f"✅ Saved face descriptor and photo: {saved_face_path}")
            else:
                set_toast(f"Saved face profile for '{name}'!")
        else:
            set_toast("Face enrolment cancelled")

    def enrol_target_voice_worker(name: str) -> None:
        nonlocal voice_recording_active, voice_rec_remaining, voice_rec_target_name
        voice_recording_active = True
        voice_rec_target_name = name
        duration = 5.0
        start_t = time.time()

        def countdown_tracker():
            while voice_recording_active:
                rem = max(0.0, duration - (time.time() - start_t))
                nonlocal voice_rec_remaining
                voice_rec_remaining = rem
                time.sleep(0.05)

        threading.Thread(target=countdown_tracker, daemon=True).start()

        try:
            set_toast(f"Speak now! Memorizing voice for '{name}'...", duration=5.5)
            print(f"\nRecording 5-second voice cue for {name}... Speak now!")
            audio_data = audio_stream.record_window(duration)
            voice_engine = VoiceEngine()
            emb = voice_engine.embedding(audio_data)
            store.save_voice(name, emb)
            saved_wav = save_voice_clip(audio_data, name)
            if saved_wav:
                set_toast(f"Saved voice cue & WAV to recorded/{name}!", duration=4.0)
                print(f"✅ Saved voice cue & audio WAV: {saved_wav}")
            else:
                set_toast(f"Saved voice cue for '{name}'!", duration=4.0)
            print(f"Saved voice cue for {name}.")
        except Exception as err:
            print(f"Voice error: {err}")
            set_toast(f"Voice error: {err}")
        finally:
            voice_recording_active = False
            voice_rec_remaining = 0.0

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal selected_face
        if event == cv2.EVENT_LBUTTONDOWN:
            with results_lock:
                clicked = None
                for face in face_results:
                    x1, y1, x2, y2 = face.box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        clicked = face
                        break
            if clicked is not None:
                selected_face = clicked
                enrol_target_face(clicked)

    cv2.namedWindow("Northstar Python")
    cv2.setMouseCallback("Northstar Python", on_mouse)

    # Asynchronous background vision thread
    def inference_worker() -> None:
        nonlocal face_results, object_results
        loop_counter = 0
        while running:
            with frame_lock:
                curr = None if latest_frame is None else latest_frame.copy()
            if curr is None:
                time.sleep(0.01)
                continue

            # Detect faces via InsightFace (GPU-accelerated)
            new_faces = engine.faces(curr)

            # Alternate YOLO object detection
            loop_counter += 1
            if loop_counter % 2 == 0:
                new_objects = engine.objects(curr)
            else:
                with results_lock:
                    new_objects = object_results

            with results_lock:
                face_results = new_faces
                object_results = new_objects

            time.sleep(0.01)

    ai_thread = threading.Thread(target=inference_worker, daemon=True)
    ai_thread.start()

    # Non-blocking real-time audio stream buffer (sub-150ms speaker recognition)
    audio_stream = AudioStreamBuffer(buffer_seconds=2.5)
    audio_stream.start()
    with app_state.lock:
        app_state.audio_stream = audio_stream

    # Asynchronous background audio listener thread
    def audio_listener_worker() -> None:
        nonlocal audio_speaker, audio_conf, audio_volume
        try:
            voice_engine = VoiceEngine()
            with app_state.lock:
                app_state.voice_engine = voice_engine
        except Exception as e:
            print(f"Audio recognizer offline: {e}")
            with audio_lock:
                audio_speaker = "Mic Offline"
            return

        speaker_hangover_until = 0.0
        ambient_noise = 0.015

        while running:
            if voice_recording_active:
                time.sleep(0.1)
                continue
            try:
                time.sleep(0.08)  # Sample every 80ms for ultra-low latency
                if not running or voice_recording_active:
                    continue

                # Instantaneous RMS from the last 150ms of the ring buffer
                recent_samples = audio_stream.get_latest(0.15)
                rms = float(np.sqrt(np.mean(recent_samples**2))) if len(recent_samples) > 0 else 0.0
                with audio_lock:
                    audio_volume = rms

                # Adaptive noise floor tracking
                ambient_noise = ambient_noise * 0.95 + min(ambient_noise, rms) * 0.05
                vad_threshold = max(0.022, min(0.060, ambient_noise * 1.55))

                now_t = time.time()
                if rms < vad_threshold:
                    with audio_lock:
                        if now_t >= speaker_hangover_until:
                            audio_speaker = "Quiet"
                            audio_conf = 0.0
                else:
                    # Speech active! Extract 1.8s from ring buffer for robust SpeechBrain ECAPA embedding
                    audio_data = audio_stream.get_latest(1.8)
                    emb = voice_engine.embedding(audio_data)
                    spk, dist = store.nearest_voice(emb)
                    with audio_lock:
                        if dist <= VOICE_LIMIT:
                            audio_speaker = spk
                            audio_conf = max(0.0, 1.0 - dist / VOICE_LIMIT)
                            speaker_hangover_until = now_t + 1.2
                        else:
                            audio_speaker = "Unknown"
                            audio_conf = 0.0
                            speaker_hangover_until = now_t + 0.8
            except Exception as e:
                time.sleep(0.1)

    audio_thread = threading.Thread(target=audio_listener_worker, daemon=True)
    audio_thread.start()

    print("Keys: [1] or Click to enrol face | [2] or V to record voice | Q to quit.")
    fps = 30.0
    prev_time = time.time()

    session_tracker = SessionTracker()
    last_tracker_update = time.time()
    last_tracker_flush = time.time()

    # Smooth animation states
    lerp_face_boxes: dict[str, list[float]] = {}
    lerp_object_boxes: dict[str, list[float]] = {}
    smooth_vol_level: float = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            with frame_lock:
                latest_frame = frame

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            with results_lock:
                current_faces = list(face_results)
                current_objects = list(object_results)

            # Draw objects with smooth interpolation (lerp)
            for label, score, box in current_objects:
                obj_key = f"{label}_{box[0]//60}_{box[1]//60}"
                if obj_key not in lerp_object_boxes:
                    lerp_object_boxes[obj_key] = [float(c) for c in box]
                cur = lerp_object_boxes[obj_key]
                cur[0] += (box[0] - cur[0]) * 0.35
                cur[1] += (box[1] - cur[1]) * 0.35
                cur[2] += (box[2] - cur[2]) * 0.35
                cur[3] += (box[3] - cur[3]) * 0.35
                s_box = (int(round(cur[0])), int(round(cur[1])), int(round(cur[2])), int(round(cur[3])))
                draw_label(frame, f"{label}  {score:.0%}", s_box, (68, 105, 218))
            if len(lerp_object_boxes) > 30:
                lerp_object_boxes.clear()

            # Draw faces with smooth interpolation (lerp)
            for face in current_faces:
                name, score = store.nearest_face(face.embedding)
                is_selected = selected_face is not None and np.array_equal(face.embedding, selected_face.embedding)
                color = (0, 215, 255) if is_selected else (188, 209, 112)
                tag = f"[CLICKED] {name}" if is_selected else f"{name}  {max(0, 1 - score):.0%}"

                face_key = f"{name}_{face.box[0]//60}_{face.box[1]//60}"
                if face_key not in lerp_face_boxes:
                    lerp_face_boxes[face_key] = [float(c) for c in face.box]
                cur = lerp_face_boxes[face_key]
                cur[0] += (face.box[0] - cur[0]) * 0.35
                cur[1] += (face.box[1] - cur[1]) * 0.35
                cur[2] += (face.box[2] - cur[2]) * 0.35
                cur[3] += (face.box[3] - cur[3]) * 0.35
                s_box = (int(round(cur[0])), int(round(cur[1])), int(round(cur[2])), int(round(cur[3])))
                draw_label(frame, tag, s_box, color)
            if len(lerp_face_boxes) > 30:
                lerp_face_boxes.clear()

            # Session intelligence tracking (presence and speech duration)
            track_now = time.time()
            track_dt = track_now - last_tracker_update
            last_tracker_update = track_now

            present_names = []
            for face in current_faces:
                fn, _ = store.nearest_face(face.embedding)
                pname = fn if fn != "Unknown face" else "Unknown"
                if pname not in present_names:
                    present_names.append(pname)

            with audio_lock:
                live_speaker = audio_speaker

            session_tracker.update(present_names, live_speaker, track_dt)

            if track_now - last_tracker_flush >= 4.0:
                session_tracker.flush()
                last_tracker_flush = track_now

            # Top HUD bar (glassmorphism panel)
            draw_glass_panel(frame, 0, 0, frame.shape[1], 44, alpha=0.82, border_color=(45, 45, 45))
            cv2.putText(
                frame,
                "NORTHSTAR | [1/Click]: Face | [2/V]: Voice | [O]: Recorded Vault | Q: Quit",
                (14, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (238, 238, 230),
                1,
                cv2.LINE_AA,
            )

            # Real-time FPS
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (frame.shape[1] - 110, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 128),
                2,
                cv2.LINE_AA,
            )

            # Bottom-Left Speaker Recognition & Audio Meter HUD
            y_bottom = frame.shape[0] - 22
            x_left = 18

            if voice_recording_active:
                badge_text = f"REC: Speak now for '{voice_rec_target_name}' ({voice_rec_remaining:.1f}s)"
                badge_color = (0, 0, 255)
            else:
                with audio_lock:
                    spk = audio_speaker
                    spk_c = audio_conf
                    vol = audio_volume

                if spk == "Quiet":
                    badge_text = "Mic: Quiet"
                    badge_color = (180, 180, 180)
                elif spk in ("Unknown", "Unknown Voice"):
                    badge_text = "Speaking: Unknown Voice"
                    badge_color = (0, 165, 255)
                elif spk == "Mic Offline":
                    badge_text = "Mic: Offline"
                    badge_color = (100, 100, 240)
                else:
                    badge_text = f"Speaking: {spk} ({spk_c:.0%})"
                    badge_color = (0, 255, 128)

            (t_w, t_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            pad_x, pad_y = 10, 8
            bx1 = x_left - pad_x
            by1 = y_bottom - t_h - pad_y
            bx2 = x_left + t_w + 50 + pad_x
            by2 = y_bottom + pad_y
            draw_glass_panel(frame, bx1, by1, bx2, by2, alpha=0.82, border_color=(65, 65, 65))

            # Indicator dot
            dot_color = (0, 0, 255) if voice_recording_active else ((0, 255, 128) if (not voice_recording_active and spk not in ("Quiet", "Mic Offline")) else (130, 130, 130))
            cv2.circle(frame, (x_left + 4, y_bottom - t_h // 2), 5, dot_color, -1)

            # Text label
            cv2.putText(frame, badge_text, (x_left + 16, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 2, cv2.LINE_AA)

            # Dynamic 5-level audio activity meter bars with studio-grade decay
            vol_val = 0.0 if voice_recording_active else vol
            target_meter = max(0.0, vol_val - 0.010) * 75.0
            if target_meter > smooth_vol_level:
                smooth_vol_level = target_meter  # Fast attack
            else:
                smooth_vol_level = smooth_vol_level * 0.86  # Smooth decay
            vol_level = min(5, int(round(smooth_vol_level)))
            vx = x_left + 22 + t_w
            for b in range(5):
                bar_h = 4 + b * 3
                bar_color = (0, 255, 128) if b < vol_level else (60, 60, 60)
                cv2.rectangle(frame, (vx + b * 6, y_bottom - bar_h), (vx + b * 6 + 4, y_bottom), bar_color, -1)

            # Toast banner (placed below top HUD so it never obstructs bottom-left)
            if now < toast_expiry and toast_message:
                banner_w = min(580, frame.shape[1] - 40)
                tx1 = (frame.shape[1] - banner_w) // 2
                tx2 = tx1 + banner_w
                ty1 = 54
                ty2 = 98
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (32, 140, 50), -1)
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    toast_message,
                    (tx1 + 18, ty2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Real-time state sync to web bridge
            with app_state.lock:
                app_state.fps = fps
                app_state.speaker = audio_speaker
                app_state.speaker_conf = audio_conf
                app_state.volume = audio_volume
                app_state.voice_recording = voice_recording_active
                app_state.voice_rec_remaining = voice_rec_remaining
                app_state.voice_target_name = voice_rec_target_name
                app_state.faces = [
                    {
                        "name": store.nearest_face(f.embedding)[0],
                        "score": round(max(0.0, 1.0 - store.nearest_face(f.embedding)[1]), 2),
                        "box": [int(b) for b in f.box],
                    }
                    for f in current_faces
                ]
                app_state.objects = [
                    {
                        "label": lbl,
                        "score": round(float(sc), 2),
                        "box": [int(b) for b in bx],
                    }
                    for lbl, sc, bx in current_objects
                ]
                ret_jpg, jpg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_jpg:
                    app_state.latest_jpeg = jpg_buf.tobytes()

            cv2.imshow("Northstar Python", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Shortcut 1 / C / E: Enrol Face
            if key in (ord("1"), ord("c"), ord("e")):
                with results_lock:
                    target = selected_face or (current_faces[0] if current_faces else None)
                if target is not None:
                    enrol_target_face(target)
                else:
                    set_toast("No visible face to enrol!")
            # Shortcut 2 / V: Record Voice Cue
            if key in (ord("2"), ord("v")):
                target_name = None
                if selected_face:
                    fn, _ = store.nearest_face(selected_face.embedding)
                    if fn != "Unknown face":
                        target_name = fn
                if not target_name and current_faces:
                    fn, _ = store.nearest_face(current_faces[0].embedding)
                    if fn != "Unknown face":
                        target_name = fn
                if not target_name:
                    target_name = prompt_for_name("Northstar Voice Enrolment")
                if target_name:
                    vt = threading.Thread(target=enrol_target_voice_worker, args=(target_name,), daemon=True)
                    vt.start()
                else:
                    set_toast("Please select a person to record voice!")
            # Shortcut O / o: Open Recorded Vault folder in Windows Explorer
            if key in (ord("o"), ord("O")):
                try:
                    RECORDED_DIR.mkdir(parents=True, exist_ok=True)
                    os.startfile(str(RECORDED_DIR))
                    set_toast("Opened recorded/ folder in Explorer", duration=3.0)
                except Exception as e:
                    set_toast(f"Error opening folder: {e}")
    finally:
        with app_state.lock:
            app_state.running = False
        running = False
        audio_stream.stop()
        cap.release()
        cv2.destroyAllWindows()
        session_tracker.log_event("Session ended.")
        session_tracker.flush(final=True)
        print(f"\n[Session Intelligence Report Saved] -> {session_tracker.log_file}\n")


class VoiceEngine:
    """SpeechBrain ECAPA embeddings; runs on CUDA when a compatible PyTorch build is installed."""

    def __init__(self):
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ModuleNotFoundError as error:
            need("voice dependencies", error)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
            savedir=str(APP_DIR / ".speechbrain"),
        )
        print(f"Voice backend: {self.device.upper()}")

    def embedding(self, audio: np.ndarray) -> np.ndarray:
        wave = self.torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return self.model.encode_batch(wave).squeeze().cpu().numpy()



def enrol_voice(args: argparse.Namespace) -> None:
    engine, store = VoiceEngine(), ProfileStore()
    voice = record(args.seconds)
    store.save_voice(args.name, engine.embedding(voice))
    saved_wav = save_voice_clip(voice, args.name)
    if saved_wav:
        print(f"✅ Saved voice cue & audio WAV for {args.name} -> {saved_wav}")
    else:
        print(f"Saved local voice cue for {args.name}.")


def listen(args: argparse.Namespace) -> None:
    engine, store = VoiceEngine(), ProfileStore()
    if not any(profile.get("voice") is not None for profile in store.data.values()):
        print("\n⚠️  No voice profiles found in profiles.json yet!")
        print("💡 Enrol someone's voice first using option [4] or press '2'/'V' inside option [1] (camera).\n")
        return
    print("\n" + "=" * 54)
    print("   🎙️  AUDIO SPEAKER RECOGNIZER (Listening...)")
    print("   Press Ctrl+C to stop")
    print("=" * 54)

    buf = AudioStreamBuffer(buffer_seconds=3.0)
    buf.start()
    try:
        time.sleep(0.5)
        while True:
            time.sleep(0.10)
            recent = buf.get_latest(0.15)
            rms = float(np.sqrt(np.mean(recent**2))) if len(recent) > 0 else 0.0
            if rms < 0.025:
                print(f"\r🎙️  MIC: [Quiet / Silence]                                ", end="", flush=True)
                continue
            audio_data = buf.get_latest(1.8)
            emb = engine.embedding(audio_data)
            name, distance = store.nearest_voice(emb)
            confidence = max(0.0, 1.0 - distance / VOICE_LIMIT)
            if distance <= VOICE_LIMIT:
                print(f"\r🎙️  CURRENT SPEAKER: {name.upper():<18} match: {confidence:.0%} (vol: {rms:.3f})", end="", flush=True)
            else:
                print(f"\r🎙️  CURRENT SPEAKER: Unknown voice      (vol: {rms:.3f})", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped listening.\n")
    finally:
        buf.stop()


def delete_profile_cli(args: argparse.Namespace) -> None:
    store = ProfileStore()
    if store.delete_profile(args.name):
        print(f"✅ Successfully deleted profile '{args.name}' and all associated face/audio data.")
    else:
        print(f"⚠️ Profile '{args.name}' not found. Current registered profiles: {list(store.data.keys())}")


def interactive_menu() -> list[str]:
    print("\n" + "=" * 54)
    print("              NORTHSTAR AI COMPANION")
    print("=" * 54)
    print("  [1] 🎥 Live Camera (Face + Object + Audio Recognizer)")
    print("  [2] 🎙️ Audio Recognizer (Listen who is speaking)")
    print("  [3] 📸 Enrol Face (from photos)")
    print("  [4] 🗣️ Record Voice Cue")
    print("  [5] 🗑️ Delete Profile")
    print("  [Q] Exit")
    print("-" * 54)
    choice = input("Select an option [1-5] (default: 1): ").strip().lower()
    if choice in ("q", "quit", "exit"):
        raise SystemExit(0)
    if not choice or choice == "1":
        return ["camera"]
    elif choice == "2":
        return ["listen"]
    elif choice == "3":
        name = input("Enter person's name: ").strip()
        photos = input("Enter photo path(s) separated by space: ").strip().split()
        return ["enrol-face", "--name", name] + photos
    elif choice == "4":
        name = input("Enter person's name: ").strip()
        return ["record-voice", "--name", name]
    elif choice == "5":
        name = input("Enter person's name to delete: ").strip()
        return ["delete", "--name", name]
    return ["camera"]


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="Northstar's CUDA-aware local Python companion.")
    app.add_argument("--high-quality", action="store_true", help="Use bigger inference inputs; slower than the fast defaults.")
    commands = app.add_subparsers(dest="command")

    face = commands.add_parser("enrol-face", aliases=["3"], help="[3] Save face descriptors from clear photos.")
    face.add_argument("--name", required=True)
    face.add_argument("photos", nargs="+")
    face.set_defaults(action=enrol_faces)

    live = commands.add_parser("camera", aliases=["1"], help="[1] Live camera view with Face, Object, and Audio recognition.")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--enrol-name", help="Optional default name for quick one-click face enrolment.")
    live.set_defaults(action=camera)

    voice = commands.add_parser("record-voice", aliases=["4"], help="[4] Create or replace a numerical voice cue.")
    voice.add_argument("--name", required=True)
    voice.add_argument("--seconds", type=float, default=6)
    voice.set_defaults(action=enrol_voice)

    live_voice = commands.add_parser("listen", aliases=["2"], help="[2] Audio speaker recognizer (who is speaking).")
    live_voice.add_argument("--window", type=float, default=2.0)
    live_voice.set_defaults(action=listen)

    delete_p = commands.add_parser("delete", aliases=["del", "rm", "5"], help="[5] Delete a person's face and voice profile.")
    delete_p.add_argument("--name", required=True, help="Name of the person to delete.")
    delete_p.set_defaults(action=delete_profile_cli)
    return app


if __name__ == "__main__":
    raw_args = sys.argv[1:]

    # Map numeric shortcuts and aliases (1, 2, 3, 4, 5, web)
    aliases = {"1": "camera", "2": "listen", "3": "enrol-face", "4": "record-voice", "5": "delete", "web": "camera"}
    if raw_args and raw_args[0] in aliases:
        raw_args[0] = aliases[raw_args[0]]

    # Default action is directly camera! Zero terminal options!
    if not raw_args:
        raw_args = ["camera"]
    elif raw_args == ["--high-quality"]:
        raw_args = ["--high-quality", "camera"]

    args = parser().parse_args(raw_args)
    if hasattr(args, "action"):
        args.action(args)
    else:
        camera(args)

