export class PitchPreprocessor {
  constructor({ tonic = 110, k = 5, vocabSize = 256 } = {}) { this.tonic = tonic; this.k = k; this.vocabSize = vocabSize; }
  setTonic(tonic) { if (!(tonic > 0)) throw new Error("Tonic must be positive."); this.tonic = tonic; }
  toRelativeCents(frequency) { return 1200 * Math.log2(frequency / this.tonic); }
  quantize(frequency) { if (!(frequency > 0)) return 0; return Math.min(this.vocabSize - 1, Math.max(0, Math.round(this.toRelativeCents(frequency) * this.k / 100))); }
}

export class RollingSequence {
  constructor(length = 5000) { this.length = length; this.values = []; }
  push(value) { this.values.push(value); if (this.values.length > this.length) this.values.shift(); }
  get ready() { return this.values.length === this.length; }
  snapshot() { return this.values.slice(); }
  clear() { this.values = []; }
}

export class PredictionSmoother {
  constructor(size = 5, required = 3) { this.size = size; this.required = required; this.history = []; }
  add(result) { this.history.push(result); if (this.history.length > this.size) this.history.shift(); const scores = new Map(); for (const item of this.history) scores.set(item.raga, (scores.get(item.raga) || 0) + item.confidence); const [raga, score] = [...scores.entries()].sort((a,b) => b[1] - a[1])[0]; const matches = this.history.filter(item => item.raga === raga).length; return { stable: matches >= this.required, raga, confidence: score / matches, latest: result }; }
  clear() { this.history = []; }
}

export class RagaInferenceClient {
  constructor(endpoint = "http://127.0.0.1:8001/api/analyze/live", mock = false) { this.endpoint = endpoint; this.mock = mock; }
  async analyze(sequence, tonic, tradition, signal) {
    if (this.mock) return this.mockResult(tradition);
    const response = await fetch(this.endpoint, { method: "POST", signal, headers: { "content-type": "application/json" }, body: JSON.stringify({ pitch_sequence: sequence, tonic, sample_rate: 100, tradition }) });
    if (!response.ok) throw new Error(`Inference service returned ${response.status}.`);
    const body = await response.json();
    if (!body.raga || !Number.isFinite(body.confidence) || !Array.isArray(body.top_predictions)) throw new Error("Inference service returned an invalid response.");
    return body;
  }
  mockResult(tradition) { const carnatic = { raga:"Mōhanaṁ", tradition:"Carnatic", confidence:.923, top_predictions:[{raga:"Mōhanaṁ",probability:.923},{raga:"Kāpi",probability:.034},{raga:"Kāṁbhōji",probability:.018},{raga:"Kalyāṇi",probability:.011},{raga:"Tōḍi",probability:.006}] }; const hindustani = { raga:"Yaman kalyāṇ", tradition:"Hindustani", confidence:.948, top_predictions:[{raga:"Yaman kalyāṇ",probability:.948},{raga:"Bihāg",probability:.021},{raga:"Khamāj",probability:.015},{raga:"Dēś",probability:.008},{raga:"Jōg",probability:.004}] }; return Promise.resolve(tradition === "Carnatic" ? carnatic : hindustani); }
}

export class LiveAudioManager {
  async start(deviceId, onFrame) { if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) throw new Error("This browser does not support microphone analysis."); this.stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: deviceId || undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); this.context = new AudioContext(); this.source = this.context.createMediaStreamSource(this.stream); this.analyser = this.context.createAnalyser(); this.analyser.fftSize = 2048; this.source.connect(this.analyser); this.data = new Float32Array(this.analyser.fftSize); let last = 0; const tick = time => { if (!this.active) return; this.analyser.getFloatTimeDomainData(this.data); if (time - last > 90) { last = time; onFrame(this.data.slice(), this.context.sampleRate); } this.raf = requestAnimationFrame(tick); }; this.active = true; this.raf = requestAnimationFrame(tick); }
  stop() { this.active = false; cancelAnimationFrame(this.raf); this.source?.disconnect(); this.stream?.getTracks().forEach(track => track.stop()); this.context?.close(); this.stream = this.source = this.context = null; }
}

export class PitchTracker {
  detect(buffer, sampleRate) { let rms = 0; for (const value of buffer) rms += value * value; rms = Math.sqrt(rms / buffer.length); if (rms < .008) return { frequency: 0, confidence: 0, level: rms }; const minLag = Math.floor(sampleRate / 1100), maxLag = Math.min(Math.floor(sampleRate / 55), Math.floor(buffer.length / 2)); let bestLag = 0, best = -Infinity; for (let lag = minLag; lag <= maxLag; lag++) { let sum = 0; for (let i = 0; i < buffer.length - lag; i++) sum += buffer[i] * buffer[i + lag]; if (sum > best) { best = sum; bestLag = lag; } } const confidence = Math.min(1, Math.max(0, best / (buffer.length * rms * rms))); return { frequency: confidence > .55 ? sampleRate / bestLag : 0, confidence, level: rms }; }
}
