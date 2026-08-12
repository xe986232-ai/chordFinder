// viterbi-worker.js
// Fase 3/4 — jalanin XHMMDecoder (Viterbi smoothing) di luar main thread
// biar UI nggak nge-freeze pas nge-decode lagu yang panjang.
// Sesuai catatan planning-porting-chord-engine.md Fase 3.

importScripts('complex_chord.js', 'xhmm_decoder.js');

self.onmessage = function (e) {
  const msg = e.data;
  if (msg.type !== 'DECODE') return;

  try {
    const { probs, templateLines, hopLength, sr } = msg;
    // probs: array 6 {data: Float32Array, dims: [nFrame, nCol]}, urutan:
    // [triad, bass, seventh, ninth, eleventh, thirteenth] — HARUS sesuai
    // urutan OUT_NAMES di pipeline.js / chordnet_ismir_naive.py.
    const matrices = probs.map((p) => new XHMM.Matrix2D(p.data, p.dims));

    const decoder = new XHMM.XHMMDecoder({ templateLines, hopLength, sr });
    const chordlab = decoder.decodeToChordlab(matrices, false);
    // chordlab: array of [startSec, endSec, chordName]

    self.postMessage({ type: 'DONE', chordlab });
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: err.message, stack: err.stack });
  }
};
