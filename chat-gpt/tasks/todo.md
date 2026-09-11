# Task: Redo Audio Recognizer & Microphone Stream Architecture

## 1. Problem & Root Cause Diagnostics
- **User Request**:
  > "it isnt working bro redo the whole code block"

- **Diagnostic Investigation & Root Cause Discovery**:
  1. **Corrupted Voice Profile in `profiles.json` (The Zero-Embedding Poisoning)**:
     - Inspection of `profiles.json` revealed that Hitesh's stored voice embedding has a cosine distance of **0.0258 (97.4% similarity) to pure digital silence (all zeros)**!
     - In `recorded/Hitesh/`, both `voice_20260911_152523.wav` and `voice_20260911_152602.wav` were 5.0-second files containing **pure zeros (`max abs: 0.0, rms: 0.0`)**.
     - Because Hitesh's stored voice profile was silence, whenever Hitesh actually spoke, his real voice produced a cosine distance of ~0.95–1.0 against the zero embedding.
     - Since 0.95 > `VOICE_LIMIT` (0.68), the recognition model classified his speech as `"Unknown speaker"`.
  2. **Session Intelligence Filter Dropping All Unknown Speech**:
     - In `SessionTracker.update()`:
       `is_valid_speaker = active_speaker not in ("Quiet", "Mic Offline", "Unknown", "")`
     - Because `"Unknown"` was excluded, all speech classified as `"Unknown"` was discarded.
     - As a result, `session_20260911_152427.txt` logged `Speaking Duration: 00m 00s (0.0%)` for every single participant across the entire 6-minute meeting.
  3. **WDM-KS Exclusivity & Stream Lifecycle on Windows**:
     - When OpenCV initializes `cv2.VideoCapture(0, cv2.CAP_DSHOW)`, Windows re-enumerates DirectShow devices.
     - MME and DirectSound APIs throw `PaErrorCode -9999 (host error)` or `Invalid device` when opened alongside DirectShow.
     - Only Realtek WDM-KS (`Microphone Array WDM-KS`, device 16) successfully streams audio alongside DirectShow.
     - However, rapid open/close probing on WDM-KS or PortAudio buffer underruns can cause the stream to stall or deliver zeros if not backed by a persistent, self-healing stream buffer.
  4. **Microphone Level & VAD Floating Floor**:
     - Realtek laptop mic array raw RMS is ~0.003–0.005 in quiet environments.
     - The previous soft gain (`* 8.0`) pushed conversational speech to ~0.035–0.045, while the noise floor tracking threshold (`ambient_noise * 1.55`) reached ~0.038, causing speech to intermittently drop below the VAD threshold.

- **Architecture Solution ("What would Mark Zuckerberg do")**:
  - **Philosophy**: Ruthless simplicity. Zero crashes. Self-healing stream. Never save silent garbage.
  - **1. Bulletproof Audio Ring Buffer (`AudioStreamBuffer`)**:
    - Increase ring buffer to 6.0s (`buffer_seconds=6.0`) to comfortably accommodate 5.0s recordings and 1.8s neural recognition windows.
    - Software gain with soft limiter: `np.tanh(mono_16k * 14.0)` so ambient noise is ~0.035 and speech sits at 0.12–0.30 RMS.
    - Self-healing watchdog: if `self.stream is None` or inactive, re-probe and restart automatically without freezing the app.
    - Silence Guard in `record_window`: calculate RMS before saving. If `rms < 0.005`, reject the recording and notify the user to speak louder rather than poisoning `profiles.json`.
  - **2. Redo `audio_listener_worker` in `camera()`**:
    - Decouple fast volume polling (50ms) for responsive HUD/Web UI meters from neural speaker embedding.
    - Rate-limit SpeechBrain ECAPA inference to once every 350ms to keep CPU light (<5%) and eliminate PortAudio buffer underruns.
    - Robust 1.4s hangover timer to bridge natural pauses between words and sentences.
  - **3. Fix Session Intelligence Tracking**:
    - Update `session_tracker.update()` to track `"Unknown"` speaker speech duration (`active_speaker not in ("Quiet", "Mic Offline", "")`).
  - **4. Reset Corrupted Voice Profile**:
    - Reset Hitesh's poisoned zero-vector voice profile in `profiles.json` (`"voice": None`) so it can be re-recorded cleanly with live audio.

---

## 2. Todo Items
- [ ] 1. Enhance `AudioStreamBuffer` in `northstar_python/app.py`:
  - Set default buffer size to 6.0s
  - Apply clean `np.tanh(mono_16k * 14.0)` gain
  - Add auto-reconnect watchdog if stream stalls or has no frames
  - Add Silence Guard in `record_window()` to reject silent clips (`rms < 0.005`)
- [ ] 2. Redo the entire audio block in `camera()` (`northstar_python/app.py` lines ~1120–1185):
  - Streamlined `audio_stream` initialization with `buffer_seconds=6.0`
  - High-efficiency `audio_listener_worker` with 50ms volume polling and 350ms rate-limited ECAPA embedding
  - Adaptive VAD with smooth 1.4s hangover timer
- [ ] 3. Fix `session_tracker.update()` in `northstar_python/app.py` (line ~628):
  - Change `active_speaker not in ("Quiet", "Mic Offline", "Unknown", "")` to `active_speaker not in ("Quiet", "Mic Offline", "")`
- [ ] 4. Clean up corrupted voice profile in `profiles.json`:
  - Reset Hitesh's voice embedding to `null` so it can be cleanly re-taught
- [ ] 5. Syntax validation & end-to-end verification:
  - Run `py_compile` on `app.py`
  - Test live audio stream + camera co-existence + recording validation
- [ ] 6. Update Review section in `tasks/todo.md` and report to user

---

## 3. Review
*(Pending execution - to be updated after verification)*
