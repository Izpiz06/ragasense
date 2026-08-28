import assert from "node:assert/strict";
import { PitchPreprocessor, PitchTracker, PredictionSmoother, RagaInferenceClient, RollingSequence } from "./live-audio.js";

const preprocessor = new PitchPreprocessor({ tonic: 110 });
assert.equal(preprocessor.quantize(110), 0);
assert.equal(preprocessor.quantize(220), 60);
assert.equal(Math.round(preprocessor.toRelativeCents(220)), 1200);
const sequence = new RollingSequence(3); sequence.push(1); sequence.push(2); sequence.push(3); sequence.push(4);
assert.deepEqual(sequence.snapshot(), [2, 3, 4]); assert.equal(sequence.ready, true);
const smoother = new PredictionSmoother(3, 2); smoother.add({ raga: "A", confidence: .8 }); const output = smoother.add({ raga: "A", confidence: .9 });
assert.equal(output.stable, true); assert.equal(output.raga, "A");
const tracker = new PitchTracker(); const sine = Float32Array.from({ length: 2048 }, (_, i) => .4 * Math.sin(2 * Math.PI * 220 * i / 44100));
const pitch = tracker.detect(sine, 44100); assert.ok(Math.abs(pitch.frequency - 220) < 8); assert.ok(pitch.confidence > .55);
const response = await new RagaInferenceClient(undefined, true).analyze([], 110, "Carnatic");
assert.equal(response.tradition, "Carnatic"); assert.equal(response.top_predictions.length, 5);
console.log("live-audio tests passed");
