/**
 * Northstar Vision — Web Frontend Client
 * Connects directly to the Python AI Companion (InsightFace, YOLO-World, SpeechBrain ECAPA).
 */

// Node.js Execution Guard: provide friendly terminal guidance if executed with `node app.js`
if (typeof window === "undefined") {
  console.log("\n========================================================");
  console.log("       🌟 NORTHSTAR VISION & AUDIO RECOGNITION 🌟        ");
  console.log("========================================================");
  console.log("  app.js is a browser client interface.");
  console.log("");
  console.log("  👉 To start the AI Engine & Web Dashboard:");
  console.log("     npm start");
  console.log("     (or: northstar_python\\.venv\\Scripts\\python.exe northstar_python\\app.py)");
  console.log("");
  console.log("  👉 To launch the web server independently:");
  console.log("     npm run web");
  console.log("     (or: node server.mjs)");
  console.log("");
  console.log("  🌐 Web Dashboard: http://localhost:4173");
  console.log("========================================================\n");
  process.exit(0);
}

// DOM Selector Helper
const $ = (selector) => document.querySelector(selector);

const el = {
  stage: $("#stage"),
  liveFeed: $("#liveFeed"),
  photo: $("#photoPreview"),
  overlay: $("#overlay"),
  modelStatus: $("#modelStatus"),
  sourceTitle: $("#sourceTitle"),
  scanHint: $("#scanHint"),
  fileStatus: $("#fileStatus"),
  cameraButton: $("#cameraButton"),
  imageInput: $("#imageInput"),
  vaultButton: $("#vaultButton"),
  stopButton: $("#stopButton"),
  fullscreenButton: $("#fullscreenButton"),
  resultList: $("#resultList"),
  recognitionSummary: $("#recognitionSummary"),
  clearLogButton: $("#clearLogButton"),
  peopleCount: $("#peopleCount"),
  personGrid: $("#personGrid"),
  personTemplate: $("#personTemplate"),
  enrollButton: $("#enrollButton"),
  emptyEnrollButton: $("#emptyEnrollButton"),
  enrollDialog: $("#enrollDialog"),
  enrollForm: $("#enrollForm"),
  personName: $("#personName"),
  enrollmentImages: $("#enrollmentImages"),
  dropZone: $("#dropZone"),
  fileCount: $("#fileCount"),
  enrollFeedback: $("#enrollFeedback"),
  savePersonButton: $("#savePersonButton"),
  closeEnrollButton: $("#closeEnrollButton"),
  privacyButton: $("#privacyButton"),
  privacyDialog: $("#privacyDialog"),
  closePrivacyButton: $("#closePrivacyButton"),
  speakerStrip: $("#speakerStrip"),
  speakerName: $("#speakerName"),
  speakerDetail: $("#speakerDetail"),
  listenButton: $("#listenButton"),
  soundBars: $("#soundBars"),
  voiceDialog: $("#voiceDialog"),
  closeVoiceButton: $("#closeVoiceButton"),
  cancelVoiceButton: $("#cancelVoiceButton"),
  voicePersonName: $("#voicePersonName"),
  recordVoiceButton: $("#recordVoiceButton"),
  recordMeter: $("#recordMeter"),
  recordSeconds: $("#recordSeconds"),
  voiceFeedback: $("#voiceFeedback"),
};

// Application State
let isConnected = false;
let isStreaming = false;
let isListening = true;
let peopleList = [];
let statusPollTimer = null;
let currentVoiceTarget = "";
let voiceCountdownTimer = null;

// Sanitize text for HTML injection safety
const escapeHtml = (str) =>
  String(str).replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]));

// Generate up to 2 uppercase initials
const getInitials = (name) =>
  name.split(/\s+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("").toUpperCase() || "?";

// Update the top right status badge
function setModelStatus(text, isReady = false) {
  if (!el.modelStatus) return;
  el.modelStatus.textContent = text;
  el.modelStatus.classList.toggle("ready", isReady);
}

// Generic API call wrapper
async function apiCall(endpoint, method = "GET", data = null) {
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (data && method !== "GET") {
    options.body = JSON.stringify(data);
  }
  const response = await fetch(endpoint, options);
  if (!response.ok) {
    throw new Error(`API ${endpoint} failed with HTTP ${response.status}`);
  }
  return await response.json();
}

// Convert file to base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = (err) => reject(err);
    reader.readAsDataURL(file);
  });
}

// Synchronize people profiles from Python backend
async function loadPeople() {
  try {
    const people = await apiCall("/api/people");
    peopleList = people || [];
    renderPeople();
  } catch (err) {
    console.warn("Could not fetch enrolled people:", err);
  }
}

// Render people grid cards
function renderPeople() {
  if (!el.personGrid || !el.personTemplate) return;
  el.peopleCount.textContent = peopleList.length;

  // Clear existing cards, keeping the emptyEnrollButton
  [...el.personGrid.querySelectorAll(".person-card")].forEach((c) => c.remove());

  peopleList.forEach((person) => {
    const card = el.personTemplate.content.firstElementChild.cloneNode(true);
    card.querySelector(".initials").textContent = getInitials(person.name);
    card.querySelector("h3").textContent = person.name;
    card.querySelector(".face-status").textContent = `${person.faceCount || 1} face descriptor${person.faceCount === 1 ? "" : "s"}`;
    card.querySelector(".voice-status").textContent = person.hasVoice ? "Voice cue active" : "No voice cue";

    const teachBtn = card.querySelector(".teach-voice-button");
    teachBtn.textContent = person.hasVoice ? "RETEACH VOICE" : "TEACH VOICE";
    teachBtn.onclick = () => openVoiceModal(person.name);

    const forgetBtn = card.querySelector(".forget-button");
    forgetBtn.onclick = async () => {
      if (confirm(`Remove ${person.name}? Face and voice data will be deleted.`)) {
        try {
          await apiCall("/api/forget", "POST", { name: person.name });
          await loadPeople();
        } catch (err) {
          alert(`Error deleting profile: ${err.message}`);
        }
      }
    };

    el.personGrid.insertBefore(card, el.personGrid.firstChild);
  });
}

// Render the real-time Observation Ledger
function renderLedger(faces = [], objects = []) {
  if (!el.resultList || !el.recognitionSummary) return;

  const total = faces.length + objects.length;
  el.recognitionSummary.innerHTML = `<strong>${total || "—"}</strong><span>${
    total ? `${faces.length} FACE${faces.length === 1 ? "" : "S"} · ${objects.length} OBJECT${objects.length === 1 ? "" : "S"}` : "WAITING FOR A VIEW"
  }</span>`;

  if (total === 0) {
    el.resultList.innerHTML = '<div class="empty-results"><p>Faces and objects detected in the frame are listed here.</p></div>';
    return;
  }

  const rows = [];

  // 1. Faces
  faces.forEach((f) => {
    const isKnown = f.name && f.name !== "Unknown face" && f.name !== "Unknown";
    const conf = Math.round((f.score || 0.9) * 100);
    const sub = isKnown ? "Enrolled Identity Match" : "Not in local vault";
    rows.push(`
      <div class="result-row">
        <i class="result-mark face"></i>
        <div class="result-label">
          <strong>${escapeHtml(f.name.toUpperCase())}</strong>
          <span>${sub}</span>
        </div>
        <span class="confidence">${conf}%</span>
      </div>
    `);
  });

  // 2. Objects
  objects.forEach((o) => {
    const conf = Math.round((o.score || 0.8) * 100);
    rows.push(`
      <div class="result-row">
        <i class="result-mark object"></i>
        <div class="result-label">
          <strong>${escapeHtml(o.label.toUpperCase())}</strong>
          <span>Object Detected (YOLO)</span>
        </div>
        <span class="confidence">${conf}%</span>
      </div>
    `);
  });

  el.resultList.innerHTML = rows.slice(0, 16).join("");
}

// Dynamically scale sound bar heights to live mic volume
function animateSoundBars(vol = 0) {
  if (!el.soundBars) return;
  const bars = el.soundBars.querySelectorAll("i");
  if (!bars.length) return;
  const v = Math.min(1.0, Math.max(0.0, vol * 4.0));
  bars.forEach((bar, idx) => {
    if (v < 0.02) {
      bar.style.height = `${[4, 10, 6, 4][idx]}px`;
    } else {
      const mult = [0.7, 1.25, 0.9, 0.65][idx];
      const h = Math.max(4, Math.min(15, Math.round(v * 15 * mult)));
      bar.style.height = `${h}px`;
    }
  });
}

// Render Voice Check strip
function renderVoice(audio) {
  if (!el.speakerName || !el.speakerDetail || !el.speakerStrip) return;

  if (!isListening) {
    el.speakerName.textContent = "Microphone paused";
    el.speakerDetail.textContent = "Click START LISTENING to resume audio recognition.";
    el.speakerStrip.classList.remove("listening");
    if (el.listenButton) {
      el.listenButton.textContent = "START LISTENING";
      el.listenButton.classList.remove("active");
    }
    animateSoundBars(0);
    return;
  }

  if (el.listenButton) {
    el.listenButton.textContent = "MUTE LISTENING";
    el.listenButton.classList.add("active");
  }

  if (!audio) {
    el.speakerName.textContent = "Microphone is off";
    el.speakerDetail.textContent = "Start Python companion to listen.";
    el.speakerStrip.classList.remove("listening");
    animateSoundBars(0);
    return;
  }

  const { speaker, confidence, volume, is_recording, remaining, target } = audio;

  // Active voice recording countdown
  if (is_recording) {
    el.speakerName.textContent = `REC: SPEAK FOR ${escapeHtml(target.toUpperCase())}`;
    el.speakerDetail.textContent = `Listening... ${remaining.toFixed(1)}s remaining`;
    el.speakerStrip.classList.add("listening");
    animateSoundBars(Math.max(0.35, volume));
    return;
  }

  animateSoundBars(volume);

  if (speaker === "Quiet" || !speaker) {
    el.speakerName.textContent = "Mic: Quiet";
    el.speakerDetail.textContent = `Ambient noise: ${(volume * 100).toFixed(0)}% · Listening for speech...`;
    el.speakerStrip.classList.remove("listening");
  } else if (speaker === "Unknown" || speaker === "Unknown voice" || speaker === "Unknown Voice") {
    el.speakerName.textContent = "Speaking: Unknown Voice";
    el.speakerDetail.textContent = `Vol: ${(volume * 100).toFixed(0)}% · Not enrolled in voice index`;
    el.speakerStrip.classList.add("listening");
  } else if (speaker === "Mic Offline") {
    el.speakerName.textContent = "Mic: Offline";
    el.speakerDetail.textContent = "Microphone stream unavailable";
    el.speakerStrip.classList.remove("listening");
    animateSoundBars(0);
  } else {
    el.speakerName.textContent = `${escapeHtml(speaker.toUpperCase())} IS SPEAKING`;
    el.speakerDetail.textContent = `Match: ${confidence}% · SpeechBrain ECAPA`;
    el.speakerStrip.classList.add("listening");
  }
}

// Poll real-time status from Python backend
async function pollStatus() {
  try {
    const status = await apiCall("/api/status");
    if (!isConnected) {
      isConnected = true;
      setModelStatus("ONLINE · PYTHON AI", true);
      el.scanHint.textContent = "Connected to Python AI engine (InsightFace + YOLO + SpeechBrain).";
      loadPeople();
    }

    // Render live updates
    renderLedger(status.faces, status.objects);
    renderVoice(status.audio);

    if (isStreaming && status.fps) {
      el.sourceTitle.textContent = `LIVE CAMERA · ${status.fps.toFixed(0)} FPS`;
    }
  } catch (err) {
    if (isConnected) {
      isConnected = false;
      setModelStatus("DISCONNECTED", false);
      el.scanHint.textContent = "Backend offline. Run: npm start to launch Python companion.";
      renderVoice(null);
    }
  }
}

// Start camera feed
function startCamera() {
  el.stage.classList.add("active");
  el.stage.dataset.source = "live";
  el.liveFeed.src = `/video_feed?t=${Date.now()}`;
  el.sourceTitle.textContent = "LIVE CAMERA";
  el.cameraButton.innerHTML = "<span>●</span> CAMERA ACTIVE";
  el.stopButton.disabled = false;
  isStreaming = true;
}

// Stop camera feed
function stopCamera() {
  el.stage.classList.remove("active");
  delete el.stage.dataset.source;
  el.liveFeed.removeAttribute("src");
  el.sourceTitle.textContent = "NO SOURCE";
  el.cameraButton.innerHTML = "<span>●</span> OPEN CAMERA";
  el.stopButton.disabled = true;
  isStreaming = false;
  renderLedger([], []);
}

// Open Enrolment Modal
function openEnrollModal() {
  el.enrollForm.reset();
  el.enrollFeedback.textContent = "";
  el.dropZone.classList.remove("has-files");
  el.fileCount.textContent = "1–6 images · saved to recorded/ vault";
  el.enrollDialog.showModal();
  setTimeout(() => el.personName.focus(), 100);
}

// Handle Face Enrolment Form Submit
async function handleEnrollSubmit(e) {
  e.preventDefault();
  const name = el.personName.value.trim();
  const files = el.enrollmentImages.files;
  if (!name || !files.length) return;

  el.savePersonButton.disabled = true;
  el.savePersonButton.textContent = "SAVING FACE...";
  el.enrollFeedback.textContent = "Processing image via InsightFace...";

  try {
    const base64Img = await fileToBase64(files[0]);
    const res = await apiCall("/api/enroll_face", "POST", { name, image: base64Img });

    if (res.ok) {
      el.enrollFeedback.textContent = `Saved face profile for ${name}!`;
      await loadPeople();
      setTimeout(() => el.enrollDialog.close(), 600);
    } else {
      throw new Error(res.error || "No clear face found in photo");
    }
  } catch (err) {
    el.enrollFeedback.textContent = `Error: ${err.message}`;
  } finally {
    el.savePersonButton.disabled = false;
    el.savePersonButton.textContent = "SAVE FACE MEMORY";
  }
}

// Open Voice Teaching Modal
function openVoiceModal(personName) {
  currentVoiceTarget = personName;
  el.voicePersonName.textContent = personName;
  el.voiceFeedback.textContent = "Ready to record 5-second voice cue.";
  el.recordSeconds.textContent = "05";
  el.recordMeter.classList.remove("recording");
  el.recordVoiceButton.disabled = false;
  el.recordVoiceButton.textContent = "RECORD 5-SECOND SAMPLE";
  el.voiceDialog.showModal();
}

// Record Voice Cue via Python Backend
async function handleRecordVoice() {
  if (!currentVoiceTarget) return;

  el.recordVoiceButton.disabled = true;
  el.recordVoiceButton.textContent = "RECORDING NOW...";
  el.recordMeter.classList.add("recording");
  el.voiceFeedback.textContent = "Speak naturally near the microphone now...";

  try {
    await apiCall("/api/record_voice", "POST", { name: currentVoiceTarget });

    let remaining = 5.0;
    clearInterval(voiceCountdownTimer);
    voiceCountdownTimer = setInterval(() => {
      remaining -= 0.1;
      if (remaining <= 0) {
        clearInterval(voiceCountdownTimer);
        el.recordSeconds.textContent = "00";
        el.recordMeter.classList.remove("recording");
        el.voiceFeedback.textContent = `Voice cue memorized for ${currentVoiceTarget}!`;
        el.recordVoiceButton.textContent = "COMPLETED";
        loadPeople();
        setTimeout(() => el.voiceDialog.close(), 1200);
      } else {
        el.recordSeconds.textContent = String(Math.ceil(remaining)).padStart(2, "0");
      }
    }, 100);
  } catch (err) {
    el.voiceFeedback.textContent = `Recording error: ${err.message}`;
    el.recordMeter.classList.remove("recording");
    el.recordVoiceButton.disabled = false;
    el.recordVoiceButton.textContent = "TRY AGAIN";
  }
}

// Open Recorded Vault Folder in Windows File Explorer
async function handleOpenVault() {
  try {
    await apiCall("/api/open_vault", "POST", {});
  } catch (err) {
    console.error("Vault open error:", err);
  }
}

// Attach Event Listeners
function setupEvents() {
  el.cameraButton.onclick = () => (isStreaming ? stopCamera() : startCamera());
  el.stopButton.onclick = stopCamera;
  el.vaultButton.onclick = handleOpenVault;

  el.imageInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      openEnrollModal();
      const dt = new DataTransfer();
      dt.items.add(file);
      el.enrollmentImages.files = dt.files;
      el.dropZone.classList.add("has-files");
      el.fileCount.textContent = `1 photo selected: ${file.name}`;
    }
  };

  el.enrollButton.onclick = openEnrollModal;
  el.emptyEnrollButton.onclick = openEnrollModal;
  el.closeEnrollButton.onclick = () => el.enrollDialog.close();
  el.enrollForm.onsubmit = handleEnrollSubmit;

  el.enrollmentImages.onchange = () => {
    const count = el.enrollmentImages.files.length;
    el.dropZone.classList.toggle("has-files", count > 0);
    el.fileCount.textContent = count ? `${count} photo${count === 1 ? "" : "s"} selected` : "1–6 images · saved to recorded/ vault";
  };

  el.recordVoiceButton.onclick = handleRecordVoice;
  el.closeVoiceButton.onclick = () => {
    clearInterval(voiceCountdownTimer);
    el.voiceDialog.close();
  };
  el.cancelVoiceButton.onclick = () => {
    clearInterval(voiceCountdownTimer);
    el.voiceDialog.close();
  };

  el.clearLogButton.onclick = () => renderLedger([], []);
  if (el.listenButton) {
    el.listenButton.onclick = () => {
      isListening = !isListening;
      renderVoice(null);
    };
  }
  el.fullscreenButton.onclick = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      el.stage.requestFullscreen?.();
    }
  };

  el.privacyButton.onclick = () => el.privacyDialog.showModal();
  el.closePrivacyButton.onclick = () => el.privacyDialog.close();
}

// Initialize Application
function init() {
  setupEvents();
  setModelStatus("CONNECTING...", false);
  loadPeople();

  // Begin real-time state polling (100ms interval for high-responsiveness)
  pollStatus();
  statusPollTimer = setInterval(pollStatus, 120);

  // Automatically start camera stream if connected
  setTimeout(() => {
    if (isConnected && !isStreaming) {
      startCamera();
    }
  }, 600);
}

// Start when document is loaded
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
