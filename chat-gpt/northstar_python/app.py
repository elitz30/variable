"""Northstar local recognition toolkit.

Uses CUDA automatically when the installed ONNX Runtime and PyTorch builds expose it.
No photo or audio recording is retained: profiles are numerical embeddings in profiles.json.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(__file__).resolve().parent
PROFILE_FILE = APP_DIR / "profiles.json"
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
            return json.loads(self.path.read_text(encoding="utf-8"))
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
        profile["voice"] = embedding.astype(float).tolist()
        self.save()

    def nearest_face(self, embedding: np.ndarray) -> tuple[str, float]:
        best_name, best_distance = "Unknown face", 1.0
        for name, profile in self.data.items():
            for stored in profile.get("faces", []):
                distance = cosine_distance(embedding, np.asarray(stored, dtype=np.float32))
                if distance < best_distance:
                    best_name, best_distance = name, distance
        return (best_name, best_distance) if best_distance <= FACE_LIMIT else ("Unknown face", best_distance)

    def nearest_voice(self, embedding: np.ndarray) -> tuple[str, float]:
        best_name, best_distance = "Unknown speaker", 1.0
        for name, profile in self.data.items():
            if profile.get("voice") is None:
                continue
            distance = cosine_distance(embedding, np.asarray(profile["voice"], dtype=np.float32))
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
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            need("vision dependencies", error)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        active = ort.get_available_providers()
        self.cuda = "CUDAExecutionProvider" in active
        self.face = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face.prepare(ctx_id=0 if self.cuda else -1, det_size=(640, 640) if high_quality else (320, 320))
        self.yolo = YOLO("yolo11n.pt")
        self.device: int | str = 0 if self.cuda else "cpu"
        self.object_size = 640 if high_quality else 416
        print(f"Vision backend: {'NVIDIA CUDA' if self.cuda else 'CPU fallback'}")

    def faces(self, frame: np.ndarray) -> list[FaceResult]:
        found = []
        for face in self.face.get(frame):
            x1, y1, x2, y2 = face.bbox.astype(int)
            found.append(FaceResult((x1, y1, x2, y2), face.normed_embedding))
        return found

    def objects(self, frame: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        result = self.yolo.predict(frame, imgsz=self.object_size, conf=0.55, device=self.device, verbose=False)[0]
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
        saved += 1
    print(f"Saved {saved} face profile(s) for {args.name}. Original images were not copied.")


def draw_label(frame: np.ndarray, label: str, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    width, height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    cv2.rectangle(frame, (x1, max(0, y1 - height - 10)), (x1 + width + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)


def camera(args: argparse.Namespace) -> None:
    import cv2

    engine, store = VisionEngine(args.high_quality), ProfileStore()
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open this camera.")

    face_results: list[FaceResult] = []
    object_results: list[tuple[str, float, tuple[int, int, int, int]]] = []
    frame_number = 0
    print("Keys: Q quits. E saves the first visible face to --enrol-name (if set).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        if frame_number % 2 == 0:
            face_results = engine.faces(frame)
        if frame_number % 6 == 0:
            object_results = engine.objects(frame)
        for face in face_results:
            name, score = store.nearest_face(face.embedding)
            draw_label(frame, f"{name}  {max(0, 1 - score):.0%}", face.box, (188, 209, 112))
        for label, score, box in object_results:
            draw_label(frame, f"{label}  {score:.0%}", box, (68, 105, 218))
        cv2.putText(frame, "NORTHSTAR PY  |  Q: quit  E: enrol visible face", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 238, 230), 2, cv2.LINE_AA)
        cv2.imshow("Northstar Python", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("e") and args.enrol_name and face_results:
            store.add_face(args.enrol_name, face_results[0].embedding)
            print(f"Saved a face descriptor for {args.enrol_name}; no image was retained.")
    cap.release()
    cv2.destroyAllWindows()


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


def record(seconds: float) -> np.ndarray:
    try:
        import sounddevice as sd
    except ModuleNotFoundError as error:
        need("sounddevice", error)
    sample_rate = 16000
    print(f"Recording for {seconds:g} seconds…")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def enrol_voice(args: argparse.Namespace) -> None:
    engine, store = VoiceEngine(), ProfileStore()
    voice = record(args.seconds)
    store.save_voice(args.name, engine.embedding(voice))
    print(f"Saved local voice cue for {args.name}. Audio was discarded.")


def listen(args: argparse.Namespace) -> None:
    engine, store = VoiceEngine(), ProfileStore()
    if not any(profile.get("voice") is not None for profile in store.data.values()):
        raise SystemExit("No voice cues saved. Run 'record-voice' first.")
    print("Listening in rolling windows. Press Ctrl+C to stop.")
    try:
        while True:
            audio = record(args.window)
            name, distance = store.nearest_voice(engine.embedding(audio))
            confidence = max(0, 1 - distance / VOICE_LIMIT)
            print(f"\rCURRENT SPEAKER: {name.upper():<24} similarity {confidence:.0%}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped listening.")


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="Northstar's CUDA-aware local Python companion.")
    app.add_argument("--high-quality", action="store_true", help="Use bigger inference inputs; slower than the fast defaults.")
    commands = app.add_subparsers(dest="command", required=True)
    face = commands.add_parser("enrol-face", help="Save face descriptors from clear photos.")
    face.add_argument("--name", required=True)
    face.add_argument("photos", nargs="+")
    face.set_defaults(action=enrol_faces)
    live = commands.add_parser("camera", help="Live face + object view. Press E to enrol one displayed face.")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--enrol-name", help="Name used when pressing E to save a visible face.")
    live.set_defaults(action=camera)
    voice = commands.add_parser("record-voice", help="Create or replace a numerical voice cue.")
    voice.add_argument("--name", required=True)
    voice.add_argument("--seconds", type=float, default=6)
    voice.set_defaults(action=enrol_voice)
    live_voice = commands.add_parser("listen", help="Print the likely current speaker continuously.")
    live_voice.add_argument("--window", type=float, default=2.0)
    live_voice.set_defaults(action=listen)
    return app


if __name__ == "__main__":
    args = parser().parse_args()
    args.action(args)
