// pipeline.js
// Fase 4 — Rakit pipeline penuh di browser:
//   File audio -> audio-decode.js (Tahap 1, PCM) -> hybrid_cqt.js (Fase 2)
//   -> onnxruntime-web ensemble 5 model (Fase 1) -> viterbi-worker.js (Fase 3)
//   -> {chords, bpm}
//
// Semua parameter (SR, hop, fmin, n_bins, bins_per_octave, SPEC_DIM, OFFSET)
// disamakan persis dengan fase2b_full_pipeline_test.js yang sudah tervalidasi
// (99.04% match ke baseline Python, sisa mismatch cuma boundary timing).
//
// TIDAK mengubah bobot model apa pun — murni orkestrasi runtime.

(function (global) {
  'use strict';

  const DEFAULT_SR = 22050;
  const DEFAULT_HOP = 512;
  const BINS_PER_OCTAVE = 36;
  const N_BINS = 288;
  const FMIN_FSHARP0 = 23.12465141947715; // librosa.note_to_hz('F#0')
  const SPEC_DIM = 252;
  const OFFSET = 18; // SHIFT_HIGH(6) * SHIFT_STEP(3), lihat chordnet_ismir_naive.py

  const MODEL_PATHS = [
    'models/ChordNet_s0.onnx',
    'models/ChordNet_s1.onnx',
    'models/ChordNet_s2.onnx',
    'models/ChordNet_s3.onnx',
    'models/ChordNet_s4.onnx',
  ];
  const OUT_NAMES = [
    'triad_logits', 'bass_logits', 'seventh_logits',
    'ninth_logits', 'eleventh_logits', 'thirteenth_logits',
  ];

  // ---------------- Session & worker (di-cache, di-load sekali) ----------------

  let sessionsPromise = null;
  function getSessions(onProgress) {
    if (!sessionsPromise) {
      sessionsPromise = (async () => {
        const sessions = [];
        for (let i = 0; i < MODEL_PATHS.length; i++) {
          if (onProgress) onProgress({ stage: 'load_model', modelIndex: i + 1, modelCount: MODEL_PATHS.length });
          const s = await ort.InferenceSession.create(MODEL_PATHS[i], { executionProviders: ['wasm'] });
          sessions.push(s);
        }
        return sessions;
      })();
    }
    return sessionsPromise;
  }

  let worker = null;
  function getWorker() {
    if (!worker) worker = new Worker('viterbi-worker.js');
    return worker;
  }

  // ---------------- Helper: softmax per baris ----------------

  function softmaxRows(data, nRow, nCol) {
    const out = new Float32Array(data.length);
    for (let t = 0; t < nRow; t++) {
      let max = -Infinity;
      const rowOff = t * nCol;
      for (let c = 0; c < nCol; c++) max = Math.max(max, data[rowOff + c]);
      let sum = 0;
      for (let c = 0; c < nCol; c++) {
        const ex = Math.exp(data[rowOff + c] - max);
        out[rowOff + c] = ex;
        sum += ex;
      }
      for (let c = 0; c < nCol; c++) out[rowOff + c] /= sum;
    }
    return out;
  }

  // Dispose tensor kalau library sediakan method-nya (jaga WASM heap tetap
  // bersih antar-inferensi, lihat catatan Fase 4 di planning kamu).
  function disposeTensor(t) {
    if (t && typeof t.dispose === 'function') {
      try { t.dispose(); } catch (e) { /* no-op */ }
    }
  }

  // ---------------- Inference ensemble 5 model ----------------

  async function runEnsembleInference(cqtCols, nFrames, onProgress) {
    const sessions = await getSessions(onProgress);

    // cqtCols: array[nBins] of Float64Array[nFrames] (bin-major, output
    // hybridCqt). Susun ulang -> (time, bin) row-major Float32Array, slice
    // window [OFFSET:OFFSET+SPEC_DIM] persis kayak model.inference() di
    // chordnet_ismir_naive.py.
    const inputFlat = new Float32Array(nFrames * SPEC_DIM);
    for (let t = 0; t < nFrames; t++) {
      const rowOff = t * SPEC_DIM;
      for (let b = 0; b < SPEC_DIM; b++) {
        inputFlat[rowOff + b] = cqtCols[OFFSET + b][t];
      }
    }

    const allProbs = [];
    for (let s = 0; s < sessions.length; s++) {
      const inputTensor = new ort.Tensor('float32', inputFlat, [1, nFrames, SPEC_DIM]);
      const results = await sessions[s].run({ input_cqt: inputTensor });

      const probs = OUT_NAMES.map((name) => {
        const t = results[name];
        const dims = t.dims;
        const nRow = dims.length === 3 ? dims[1] : dims[0];
        const nCol = dims[dims.length - 1];
        const sm = softmaxRows(t.data, nRow, nCol);
        disposeTensor(t);
        return { data: sm, dims: [nRow, nCol] };
      });

      disposeTensor(inputTensor);
      allProbs.push(probs);
      if (onProgress) onProgress({ stage: 'inference', modelIndex: s + 1, modelCount: sessions.length });
    }

    // Ensemble average per komponen (triad/bass/seventh/ninth/eleventh/thirteenth)
    const avgProbs = OUT_NAMES.map((_, i) => {
      const dims = allProbs[0][i].dims;
      const nRow = dims[0], nCol = dims[1];
      const avg = new Float32Array(nRow * nCol);
      for (const modelProbs of allProbs) {
        const d = modelProbs[i].data;
        for (let k = 0; k < avg.length; k++) avg[k] += d[k] / allProbs.length;
      }
      return { data: avg, dims: [nRow, nCol] };
    });

    return avgProbs;
  }

  // ---------------- Viterbi decode (Web Worker) ----------------

  function decodeViterbi(avgProbs, templateLines, hopLength, sr) {
    return new Promise((resolve, reject) => {
      const w = getWorker();
      const onMsg = (e) => {
        const msg = e.data;
        if (msg.type === 'DONE') {
          w.removeEventListener('message', onMsg);
          resolve(msg.chordlab);
        } else if (msg.type === 'ERROR') {
          w.removeEventListener('message', onMsg);
          reject(new Error(msg.message));
        }
      };
      w.addEventListener('message', onMsg);

      // Transferable Objects: buffer dipindah (transfer), bukan di-copy,
      // sesuai catatan Fase 3 di planning.
      const transferList = avgProbs.map((p) => p.data.buffer);
      w.postMessage({ type: 'DECODE', probs: avgProbs, templateLines, hopLength, sr }, transferList);
    });
  }

  // ---------------- BPM (estimasi, BUKAN port librosa.beat.beat_track) ----------------
  //
  // CATATAN PENTING: ini estimasi tempo pakai onset envelope (spectral-flux
  // sederhana dari energi RMS per frame) + autocorrelation, BUKAN hasil
  // port 1:1 dari librosa.beat.beat_track() (algoritma beda total, dan
  // belum divalidasi numerik terhadap baseline Python). Dipakai HANYA untuk
  // badge BPM di UI dan tempo default file MIDI (fallback 120 BPM kalau
  // gagal) — TIDAK memengaruhi akurasi deteksi chord sama sekali, karena
  // XHMMDecoder di pipeline ini pakai use_beats=False (lihat komentar di
  // xhmm_decoder.js). Kalau butuh BPM yang presisi/tervalidasi, ini bagian
  // yang perlu divalidasi terpisah dulu sebelum dipercaya penuh.
  function estimateBpm(pcm, sr) {
    const hop = 512;
    const winSize = 2048;
    const nFrames = Math.floor((pcm.length - winSize) / hop);
    if (nFrames < 8) return null;

    const energy = new Float32Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      let sum = 0;
      const start = t * hop;
      for (let i = 0; i < winSize; i++) {
        const v = pcm[start + i] || 0;
        sum += v * v;
      }
      energy[t] = Math.sqrt(sum / winSize);
    }
    const flux = new Float32Array(nFrames);
    for (let t = 1; t < nFrames; t++) {
      flux[t] = Math.max(0, energy[t] - energy[t - 1]);
    }

    const frameRate = sr / hop;
    const minBpm = 60, maxBpm = 200;
    const minLag = Math.round(frameRate * 60 / maxBpm);
    const maxLag = Math.round(frameRate * 60 / minBpm);

    let bestLag = -1, bestScore = -Infinity;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let t = lag; t < nFrames; t++) sum += flux[t] * flux[t - lag];
      if (sum > bestScore) { bestScore = sum; bestLag = lag; }
    }
    if (bestLag <= 0) return null;
    const bpm = 60 * frameRate / bestLag;
    return Math.round(bpm * 10) / 10;
  }

  // ---------------- Entry point ----------------

  let templateLinesCache = null;
  async function loadTemplateLines() {
    if (templateLinesCache) return templateLinesCache;
    const res = await fetch('submission_chord_list.txt');
    if (!res.ok) throw new Error('Gagal load submission_chord_list.txt (' + res.status + ')');
    const text = await res.text();
    templateLinesCache = text.split('\n');
    return templateLinesCache;
  }

  /**
   * @param {File} file
   * @param {object} [opts]
   * @param {(info:object)=>void} [opts.onProgress]
   * @returns {Promise<{chords: Array<{start:number,end:number,chord:string}>, bpm: number|null}>}
   */
  async function analyze(file, opts) {
    opts = opts || {};
    const onProgress = opts.onProgress || function () {};

    // --- Instrumentasi waktu per-tahap (buat diagnosa bottleneck) ---
    const _t0 = performance.now();
    let _tPrev = _t0;
    const _lap = (label) => {
      const now = performance.now();
      console.log(`[ChordPipeline] ${label}: ${((now - _tPrev) / 1000).toFixed(2)}s (total ${((now - _t0) / 1000).toFixed(2)}s)`);
      _tPrev = now;
    };

    onProgress({ stage: 'decode' });
    const decoded = await AudioDecode.decodeAudioFileToPCM(file, DEFAULT_SR);
    _lap('decode (audio -> PCM)');

    onProgress({ stage: 'cqt' });
    const { data: cqtCols, nFrames } = HybridCQT.hybridCqt(
      decoded.data, DEFAULT_SR, DEFAULT_HOP, FMIN_FSHARP0, N_BINS, BINS_PER_OCTAVE, 1
    );
    _lap(`cqt (${nFrames} frame)`);

    const templateLines = await loadTemplateLines();

    onProgress({ stage: 'inference', modelIndex: 0, modelCount: MODEL_PATHS.length });
    const avgProbs = await runEnsembleInference(cqtCols, nFrames, onProgress);
    _lap('inference (ensemble 5 model)');

    onProgress({ stage: 'decode_viterbi' });
    const chordlabRaw = await decodeViterbi(avgProbs, templateLines, DEFAULT_HOP, DEFAULT_SR);
    const chords = chordlabRaw.map(([start, end, chord]) => ({ start, end, chord }));
    _lap('viterbi decode');

    onProgress({ stage: 'bpm' });
    const bpm = estimateBpm(decoded.data, DEFAULT_SR);
    _lap('bpm estimate');

    console.log(`[ChordPipeline] TOTAL: ${((performance.now() - _t0) / 1000).toFixed(2)}s untuk ${nFrames} frame (~${(nFrames * DEFAULT_HOP / DEFAULT_SR).toFixed(1)}s audio)`);

    onProgress({ stage: 'done' });
    return { chords, bpm };
  }

  global.ChordPipeline = { analyze };

})(typeof window !== 'undefined' ? window : this);
