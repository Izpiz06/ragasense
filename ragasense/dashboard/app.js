const ragas = [
  ["Aṭāna","Carnatic",9,3,.8],["Bhairavi","Carnatic",9,3,1],["Kalyāṇi","Carnatic",9,3,.8],["Kāpi","Carnatic",9,3,1],["Kāṁbhōji","Carnatic",9,3,1],["Kēdāragauḷa","Carnatic",9,3,.857],["Mōhanaṁ","Carnatic",9,3,1],["Rītigauḷa","Carnatic",9,3,1],["Tōḍi","Carnatic",9,3,1],["Śankarābharaṇaṁ","Carnatic",9,3,1],
  ["Bhairav","Hindustani",9,1,1],["Jōg","Hindustani",9,1,1],["Bihāg","Hindustani",9,1,1],["Bilāsakhānī tōḍī","Hindustani",9,1,1],["Darbāri","Hindustani",9,1,1],["Khamāj","Hindustani",9,1,1],["Dēś","Hindustani",9,1,1],["Miyān malhār","Hindustani",9,1,1],["Yaman kalyāṇ","Hindustani",9,1,.667],["Śrī","Hindustani",9,1,1]
].map(([name, tradition, recordings, heldOut, f1]) => ({name, tradition, recordings, heldOut, f1}));

const fmt = value => `${(value * 100).toFixed(value === 1 ? 0 : 1)}%`;
const title = id => ({analysis:"WORKSPACE",corpus:"CORPUS",architecture:"ARCHITECTURE",evaluation:"EVALUATION"}[id]);
let activeMode = "all";

function navigate(route) {
  const valid = ["analysis", "corpus", "architecture", "evaluation"];
  const page = valid.includes(route) ? route : "analysis";
  document.querySelectorAll("[data-page]").forEach(section => section.hidden = section.id !== page);
  document.querySelectorAll("[data-route]").forEach(link => link.classList.toggle("is-active", link.dataset.route === page));
  document.getElementById("page-crumb").textContent = title(page);
  document.querySelector(".sidebar").classList.remove("is-open");
}

function stat(label, value, detail) { return `<article class="stat-card"><span class="eyebrow">${label}</span><strong>${value}</strong><small>${detail}</small></article>`; }
function renderCorpus(filter = "", tradition = "all") {
  const items = ragas.filter(r => (tradition === "all" || r.tradition === tradition) && `${r.name} ${r.tradition}`.toLocaleLowerCase().includes(filter.toLocaleLowerCase()));
  document.getElementById("corpus-table").innerHTML = items.map(r => `<tr><td>${r.tradition}</td><td><strong>${r.name}</strong></td><td class="number">${r.recordings}</td><td class="number">${r.heldOut}</td><td><span class="coverage"><i style="width:100%"></i><span class="mono">pitch + tonic</span></span></td></tr>`).join("") || `<tr><td colspan="5" class="empty-state">No ragas match this filter.</td></tr>`;
}
function renderMatrix() {
  const labels = ragas.map(r => r.name.replaceAll("ā", "a").replaceAll("ō", "o").replaceAll("ṁ", "m").slice(0, 5));
  let cells = `<span></span>${labels.map(label => `<span class="matrix-column-label">${label}</span>`).join("")}`;
  ragas.forEach((r, i) => { cells += `<span class="matrix-label">${labels[i]}</span>`; ragas.forEach((_, j) => { const value = i === j ? r.f1 : (i === 18 && j === 12 ? .33 : (i === 0 && j === 5 ? .33 : (i === 2 && j === 5 ? .33 : 0))); cells += `<span class="matrix-cell" style="--level:${value ? (.08 + value * .82).toFixed(2) : .02}" data-tip="${r.name} → ${ragas[j].name}: ${fmt(value)}"></span>`; }); });
  document.getElementById("matrix").innerHTML = `<div class="matrix-grid">${cells}</div>`;
}
function setMode(mode) {
  activeMode = mode;
  const label = mode === "all" ? "All traditions" : `${mode} mode`;
  document.querySelectorAll("#analysis-mode [data-mode]").forEach(button => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", selected);
  });
  document.getElementById("mode-indicator").textContent = label.toUpperCase();
  document.getElementById("analysis-mode-description").textContent = mode === "all"
    ? "Identify raga from tonic-normalized melodic pitch, never absolute tuning."
    : `Identify ${mode} ragas from tonic-normalized melodic pitch within the selected tradition.`;
  document.getElementById("tradition-filter").value = mode;
  renderCorpus(document.getElementById("raga-search").value, mode);
  document.getElementById("prediction-name").textContent = "Awaiting input";
  document.getElementById("confidence-value").textContent = "—";
  document.querySelector(".confidence-value").style.strokeDashoffset = 320;
  document.getElementById("candidates-list").innerHTML = `<p class="empty-state">${label} candidates will appear after input.</p>`;
  window.dispatchEvent(new CustomEvent("ragasense:mode-change", { detail: { mode } }));
}
function simulatePrediction() {
  const options = activeMode === "Carnatic"
    ? [["Mōhanaṁ", .923, [["Kāpi",.034],["Kāṁbhōji",.018],["Kalyāṇi",.011]]]]
    : activeMode === "Hindustani"
      ? [["Yaman kalyāṇ", .948, [["Bihāg",.021],["Khamāj",.015],["Dēś",.008]]]]
      : [["Yaman kalyāṇ", .948, [["Bihāg",.021],["Khamāj",.015],["Dēś",.008]]],["Mōhanaṁ", .923, [["Kāpi",.034],["Kāṁbhōji",.018],["Kalyāṇi",.011]]]];
  const [name, confidence, candidates] = options[Math.floor(Math.random() * options.length)];
  document.getElementById("prediction-name").textContent = name;
  document.getElementById("confidence-value").textContent = fmt(confidence);
  document.querySelector(".confidence-value").style.strokeDashoffset = 320 - 320 * confidence;
  document.getElementById("prediction-note").textContent = "Rolling windows are being stabilized into one live prediction.";
  document.getElementById("candidates-list").innerHTML = candidates.map(([candidate, score]) => `<div class="candidate"><div class="candidate-label"><span>${candidate}</span><span>${fmt(score)}</span></div><div class="bar"><i style="width:${score * 100}%"></i></div></div>`).join("");
}

document.querySelectorAll("[data-route]").forEach(link => link.addEventListener("click", event => { event.preventDefault(); const route = link.dataset.route; history.replaceState(null, "", `#${route}`); navigate(route); }));
window.addEventListener("hashchange", () => navigate(location.hash.slice(1)));
document.querySelector(".menu-button").addEventListener("click", () => document.querySelector(".sidebar").classList.toggle("is-open"));
document.querySelectorAll("#analysis-mode [data-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.querySelectorAll(".live-action").forEach(button => button.addEventListener("click", () => window.dispatchEvent(new Event("ragasense:start-live"))));
document.getElementById("audio-file").addEventListener("change", event => { const file = event.target.files[0]; if (!file) return; document.getElementById("input-status").textContent = `${file.name} selected. File analysis is not connected to the live inference endpoint.`; });
document.getElementById("auto-tonic").addEventListener("click", () => { document.getElementById("input-status").textContent = "Automatic tonic detection is not available in the current live backend; using fixed A reference."; document.getElementById("tonic-note").textContent = "A"; document.getElementById("tonic-hz").textContent = "110.00 Hz · fixed reference"; window.dispatchEvent(new CustomEvent("ragasense:tonic-change", { detail: { tonic: 110 } })); });
document.getElementById("manual-tonic").addEventListener("click", () => { const hz = Number(prompt("Enter tonic / Sa frequency in Hz (for example: 110 for A)", "110")); if (Number.isFinite(hz) && hz > 0) { document.getElementById("tonic-note").textContent = "Manual"; document.getElementById("tonic-hz").textContent = `${hz.toFixed(2)} Hz`; window.dispatchEvent(new CustomEvent("ragasense:tonic-change", { detail: { tonic: hz } })); } });
document.getElementById("raga-search").addEventListener("input", event => renderCorpus(event.target.value, document.getElementById("tradition-filter").value));
document.getElementById("tradition-filter").addEventListener("change", event => {
  renderCorpus(document.getElementById("raga-search").value, event.target.value);
  setMode(event.target.value);
});

document.getElementById("corpus-stats").innerHTML = [stat("Selected ragas", "20", "10 Carnatic · 10 Hindustani"),stat("Source recordings", "180", "9 recordings per raga"),stat("Held-out recordings", "40", "recording-level split"),stat("Feature basis", "2", "pitch and tonic tracks")].join("");
document.getElementById("evaluation-stats").innerHTML = [stat("Recording accuracy", "95.0%", "38 of 40 held-out recordings"),stat("Macro precision", "0.963", "class-balanced precision"),stat("Macro recall", "0.967", "class-balanced recall"),stat("Macro F1", "0.956", "20-raga proof of concept")].join("");
document.getElementById("config-list").innerHTML = [["Sequence length", "5,000 frames"],["Vocabulary", "256 pitch tokens"],["Embedding", "128 dimensions"],["LSTM hidden size", "768"],["Dropout", "0.3"],["Learning rate", "1e−4"],["Batch size", "40"]].map(([key,value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
document.getElementById("raga-metrics").innerHTML = ragas.sort((a,b) => a.f1-b.f1).map(r => `<div class="raga-row"><div class="raga-row-top"><span>${r.name}</span><span>F1 ${r.f1.toFixed(3)}</span></div><div class="f1-bar"><i style="width:${r.f1 * 100}%"></i></div></div>`).join("");
renderCorpus(); renderMatrix(); navigate(location.hash.slice(1));
window.ragaSenseLive = { getMode: () => activeMode, setMode };
