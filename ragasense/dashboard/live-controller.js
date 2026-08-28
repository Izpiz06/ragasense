import { LiveAudioManager, PitchPreprocessor, PitchTracker, PredictionSmoother, RagaInferenceClient, RollingSequence } from "./live-audio.js";

const $ = id => document.getElementById(id);
const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

class LiveAnalysisController {
  constructor() {
    this.audio = new LiveAudioManager(); this.tracker = new PitchTracker();
    this.preprocessor = new PitchPreprocessor({ tonic: 110 }); this.sequence = new RollingSequence(5000);
    this.smoother = new PredictionSmoother(); this.mock = new URLSearchParams(location.search).get("liveMock") === "1";
    this.client = new RagaInferenceClient(undefined, this.mock); this.contour = []; this.confidenceHistory = []; this.running = false;
    this.inferencePending = false; this.abortController = null; this.lastInferenceAt = 0; this.startedAt = 0;
  }
  async initialize() {
    $("stop-live").addEventListener("click", () => this.stop());
    window.addEventListener("ragasense:start-live", () => this.start());
    window.addEventListener("ragasense:mode-change", () => this.resetPrediction());
    window.addEventListener("ragasense:tonic-change", event => { this.preprocessor.setTonic(event.detail.tonic); this.resetSequence(); });
    await this.listDevices(); navigator.mediaDevices?.addEventListener?.("devicechange", () => this.listDevices());
  }
  async listDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const select = $("input-device"), value = select.value;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
    select.innerHTML = `<option value="">Default microphone</option>${devices.map((device, index) => `<option value="${device.deviceId}">${device.label || `Microphone ${index + 1}`}</option>`).join("")}`;
    select.value = [...select.options].some(option => option.value === value) ? value : "";
  }
  setState(state, detail = "") {
    $("live-state").textContent = state; $("live-state").className = `live-status ${state === "Listening" ? "is-listening" : state === "Error" ? "is-error" : ""}`;
    if (detail) $("input-status").textContent = detail;
  }
  async start() {
    if (this.running) return;
    this.resetSequence(); this.setState("Permission required", "Requesting microphone access…");
    try {
      await this.audio.start($("input-device").value, (samples, sampleRate) => this.onAudioFrame(samples, sampleRate));
      this.running = true; this.startedAt = performance.now(); $("stop-live").disabled = false;
      this.setState("Listening", "Listening for voiced violin pitch. Collecting melodic information…"); await this.listDevices();
    } catch (error) {
      const message = error.name === "NotAllowedError" ? "Microphone permission was denied. Allow access and try again." : error.name === "NotFoundError" ? "No microphone is available." : error.message;
      this.setState("Error", message); this.stop(false);
    }
  }
  async stop(showStatus = true) {
    const finalSequence = this.sequence.ready ? this.sequence.snapshot() : null;
    this.running = false; this.audio.stop(); this.abortController?.abort(); this.abortController = null; this.inferencePending = false; $("stop-live").disabled = true;
    if (!showStatus) return;
    if (!finalSequence) {
      const missing = this.sequence.length - this.sequence.values.length;
      this.setState("Stopped", `Stopped. A final DeepSRGM result requires ${missing.toLocaleString()} more voiced frames (5,000 total).`);
      return;
    }
    this.setState("Processing", "Finalizing the recorded melodic sequence…");
    try {
      const result = await this.client.analyze(finalSequence, this.preprocessor.tonic, window.ragaSenseLive?.getMode?.() || "all");
      const smooth = this.smoother.add(result); this.renderPrediction(result, { ...smooth, stable: true });
      this.setState("Stopped", "Final raga prediction is shown above. Microphone access has been released.");
    } catch (error) { this.setState("Error", this.mock ? error.message : "The final sequence was captured, but the inference service is unavailable."); }
  }
  resetSequence() { this.sequence.clear(); this.smoother.clear(); this.contour = []; this.confidenceHistory = []; this.drawConfidenceHistory(); this.resetPrediction(); }
  resetPrediction() { $("prediction-name").textContent = "Collecting melody"; $("prediction-tradition").textContent = "Tradition —"; $("confidence-value").textContent = "—"; document.querySelector(".confidence-value").style.strokeDashoffset = 320; $("prediction-note").textContent = "Collecting melodic information…"; $("candidates-list").innerHTML = `<p class="empty-state">Waiting for a stable melodic sequence.</p>`; }
  onAudioFrame(samples, sampleRate) {
    if (!this.running && !this.audio.active) return;
    const pitch = this.tracker.detect(samples, sampleRate); this.drawWaveform(samples); this.renderFrame(pitch);
    if (!pitch.frequency || pitch.confidence < .45) { $("input-status").textContent = "Listening — waiting for a clear voiced violin pitch…"; return; }
    this.sequence.push(this.preprocessor.quantize(pitch.frequency));
    if (!this.sequence.ready) { $("input-status").textContent = `Collecting melodic information… ${this.sequence.values.length.toLocaleString()} / 5,000 voiced frames`; return; }
    const now = performance.now(); if (!this.inferencePending && now - this.lastInferenceAt > 2500) this.infer();
  }
  renderFrame({ frequency, confidence, level }) {
    $("audio-level").style.width = `${Math.min(100, level * 700)}%`; $("live-pitch-confidence").textContent = `${Math.round(confidence * 100)}%`;
    if (frequency) { $("live-frequency").textContent = `${frequency.toFixed(1)} Hz`; $("live-note").textContent = this.noteFor(frequency); this.contour.push(this.preprocessor.toRelativeCents(frequency)); if (this.contour.length > 100) this.contour.shift(); this.drawContour(); }
    if (this.startedAt) $("live-duration").textContent = this.formatDuration((performance.now() - this.startedAt) / 1000);
  }
  noteFor(frequency) { const midi = Math.round(69 + 12 * Math.log2(frequency / 440)); return `${NOTE_NAMES[(midi + 120) % 12]}${Math.floor(midi / 12) - 1}`; }
  drawContour() {
    const path = document.querySelector(".contour-chart path"); if (!path || this.contour.length < 2) return;
    const points = this.contour.map((cents, index) => `${index ? "L" : "M"}${(index / 99) * 800} ${Math.max(10, Math.min(250, 130 - cents / 10))}`); path.setAttribute("d", points.join(" "));
  }
  drawWaveform(samples) { const canvas = $("live-waveform"), context = canvas.getContext("2d"), { width, height } = canvas; context.clearRect(0, 0, width, height); context.strokeStyle = "#f6a65d"; context.lineWidth = 1.5; context.beginPath(); for (let x = 0; x < width; x++) { const sample = samples[Math.floor(x / width * samples.length)] || 0; const y = height / 2 + sample * height * .42; x ? context.lineTo(x, y) : context.moveTo(x, y); } context.stroke(); }
  drawConfidenceHistory() { const canvas = $("confidence-history"), context = canvas.getContext("2d"), { width, height } = canvas; context.clearRect(0, 0, width, height); if (this.confidenceHistory.length < 2) return; context.strokeStyle = "#f6a65d"; context.lineWidth = 2; context.beginPath(); this.confidenceHistory.forEach((value, index) => { const x = index / 29 * width, y = height - value * height; index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke(); }
  async infer() {
    this.inferencePending = true; this.lastInferenceAt = performance.now(); this.abortController = new AbortController(); this.setState("Processing", "Analyzing the current rolling melodic sequence…");
    try {
      const result = await this.client.analyze(this.sequence.snapshot(), this.preprocessor.tonic, window.ragaSenseLive?.getMode?.() || "all", this.abortController.signal);
      if (!this.running) return; const smooth = this.smoother.add(result); this.renderPrediction(result, smooth); this.setState("Listening", smooth.stable ? "Stable prediction updated from recent rolling windows." : "Prediction stabilizing…");
    } catch (error) { if (error.name !== "AbortError") this.setState("Error", this.mock ? error.message : "Live inference service is unavailable. Start the local RagaSense API, then try again."); }
    finally { this.inferencePending = false; }
  }
  renderPrediction(result, smooth) {
    this.confidenceHistory.push(result.confidence); if (this.confidenceHistory.length > 30) this.confidenceHistory.shift(); this.drawConfidenceHistory();
    if (smooth.stable) { $("prediction-name").textContent = smooth.raga; $("confidence-value").textContent = `${(smooth.confidence * 100).toFixed(1)}%`; document.querySelector(".confidence-value").style.strokeDashoffset = 320 - 320 * smooth.confidence; }
    $("prediction-tradition").textContent = `Tradition · ${result.tradition}`;
    $("mode-indicator").textContent = `${smooth.stable ? "stable" : "stabilizing"} prediction`.toUpperCase();
    $("prediction-note").textContent = smooth.stable ? `Raw model: ${result.raga} ${(result.confidence * 100).toFixed(1)}%` : `Raw model: ${result.raga} · Prediction stabilizing…`;
    $("candidates-list").innerHTML = result.top_predictions.map((item, index) => `<div class="candidate"><div class="candidate-label"><span>${index + 1}. ${item.raga}</span><span>${(item.probability * 100).toFixed(1)}%</span></div><div class="bar"><i style="width:${item.probability * 100}%"></i></div></div>`).join("");
  }
  formatDuration(seconds) { const minutes = Math.floor(seconds / 60); return `${String(minutes).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`; }
}

const controller = new LiveAnalysisController(); controller.initialize();
window.ragaSenseLiveController = controller;
