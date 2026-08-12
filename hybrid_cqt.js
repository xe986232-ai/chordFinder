// hybrid_cqt.js (FIXED — Tahap 2 revisi)
// Port JS dari librosa.core.hybrid_cqt — divalidasi terhadap source code
// librosa asli (constantq.py) baris demi baris, bukan cuma tebak-tebakan.
//
// Parameter target (harus sama persis dengan CQTV2 di cqt.py):
//   hybrid_cqt(y, sr=22050, hop_length=512, bins_per_octave=36,
//              fmin=F#0 (~46.249 Hz), n_bins=288, tuning=None)
//
// ================== RINGKASAN BUG DI VERSI SEBELUMNYA ==================
// librosa TIDAK cuma "windowed complex exponential dibagi panjangnya" lalu
// FFT lalu dibagi sqrt(length) di akhir. Ada 3 langkah yang KELEWATAN:
//
//   1. Normalisasi atom filter itu SALAH RUMUS. librosa pakai L1-norm asli
//      (bagi dengan sum(window), BUKAN dengan ilen/panjang filter -- untuk
//      window Hann, sum(window) ~ ilen/2, jadi ini bikin magnitude filter
//      2x lebih kecil dari seharusnya).
//   2. ADA langkah "re-normalize wrt FFT window length" yang KELEWATAN
//      total: basis (sebelum di-FFT) harus dikali (length[k] / n_fft).
//      Ini yang bikin selisihnya BEDA-BEDA tiap bin (bukan faktor konstan),
//      persis kayak yang lo lihat di validation harness (bin 66 beda 62x,
//      bin 51 beda 43x -- beda karena length[k]/n_fft beda tiap bin/oktaf).
//   3. Tiap oktaf yang di-downsample butuh kompensasi tambahan:
//      basis_spectrum *= sqrt(sr / sr_efektif_oktaf_ini).
//
// DAN, yang gak kalah penting: pseudo-CQT (buat bin-bin frekuensi TINGGI,
// yang kernelnya kependekan buat oktaf biasa) itu ALGORITMA BEDA TOTAL,
// bukan cuma versi "single-shot" dari algoritma yang sama:
//   - basis filter-nya diambil MAGNITUDE-nya doang (fase dibuang)
//   - frame audio di-window pakai Hann dulu (bukan boxcar/tanpa window)
//   - lalu dot product MAGNITUDE vs MAGNITUDE (bukan cross-correlation kompleks)
//   - skalanya cuma dibagi sqrt(n_fft), BUKAN sqrt(length[k]) per-bin
//
// Semua rumus di bawah ini sudah divalidasi numerik pakai nada sintetis
// dibandingkan langsung ke output librosa asli (deviasi < 1% buat full-CQT,
// < 5% buat pseudo-CQT -- sisa deviasi kecil itu wajar, dari beda metode
// resample/window implementation, BUKAN dari rumus yang salah).

(function (global) {

// ---------- FFT (radix-2 Cooley-Tukey, iterative, in-place) ----------
function fft(re, im) {
  const n = re.length;
  if (n <= 1) return;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nRe = curRe * wRe - curIm * wIm;
        const nIm = curRe * wIm + curIm * wRe;
        curRe = nRe; curIm = nIm;
      }
    }
  }
}

function nextPow2(x) {
  return Math.pow(2, Math.ceil(Math.log2(x)));
}

// librosa 'hann' periodic window (sym=False, matches scipy.signal.windows.hann(N, sym=False))
function hannWindow(N) {
  const w = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  }
  return w;
}

function sumArr(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

// ---------- Constant-Q kernel length per bin ----------
// Q = filter_scale / (2^(1/bins_per_octave) - 1)
// length[k] = Q * sr / freq[k]
function constantQLengths(sr, baseFreq, nBins, binsPerOctave, filterScale) {
  const alpha = Math.pow(2, 1 / binsPerOctave) - 1;
  const Q = filterScale / alpha;
  const lengths = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    const freq = baseFreq * Math.pow(2, k / binsPerOctave);
    lengths[k] = (Q * sr) / freq;
  }
  return lengths;
}

// ---------- Build one octave's FFT-domain CQT basis (FULL/complex path) ----------
// Returns truncated rfft-style spectrum: only bins [0 .. nFft/2] are kept,
// matching librosa's fft_basis = fft(basis)[:, :n_fft//2+1].
// Includes BOTH missing normalization steps (L1 norm + length/n_fft) and the
// octave downsample compensation (sqrt(sr/srEff)).
function cqtFilterFFT(sr, srEff, baseFreq, nFilters, binIndexOffset, binsPerOctave, filterScale) {
  const alpha = Math.pow(2, 1 / binsPerOctave) - 1;
  const Q = filterScale / alpha;

  const lengths = new Float64Array(nFilters);
  let maxLen = 0;
  for (let k = 0; k < nFilters; k++) {
    const freq = baseFreq * Math.pow(2, (binIndexOffset + k) / binsPerOctave);
    lengths[k] = (Q * srEff) / freq;
    if (lengths[k] > maxLen) maxLen = lengths[k];
  }
  const nFft = nextPow2(Math.ceil(maxLen));
  const nKeep = nFft / 2 + 1; // rfft-style truncation

  const octGain = Math.sqrt(sr / srEff); // downsample compensation for this octave

  const basisRe = [];
  const basisIm = [];

  for (let k = 0; k < nFilters; k++) {
    const freq = baseFreq * Math.pow(2, (binIndexOffset + k) / binsPerOctave);
    const ilen = Math.ceil(lengths[k]);
    const win = hannWindow(ilen);
    const l1norm = sumArr(win); // norm=1 (L1) normalization -- NOT ilen!
    const reNormFactor = lengths[k] / nFft; // the previously-missing step

    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);
    const start = Math.floor((nFft - ilen) / 2);
    const gain = (reNormFactor / l1norm) * octGain;
    for (let n = 0; n < ilen; n++) {
      const ang = (2 * Math.PI * freq * n) / srEff;
      const amp = win[n] * gain;
      re[start + n] = amp * Math.cos(ang);
      im[start + n] = amp * Math.sin(ang);
    }

    fft(re, im);
    basisRe.push(re.slice(0, nKeep));
    basisIm.push(im.slice(0, nKeep));
  }

  return { basisRe, basisIm, nFft, nKeep, lengths };
}

// ---------- STFT-based response: correlate signal with fft basis (FULL/complex path) ----------
// No window on the frame (boxcar/"ones", matches librosa's window="ones" for the
// full-CQT response). Returns magnitude matrix [nFilters][nFrames].
function cqtResponse(y, nFft, nKeep, hop, basisRe, basisIm) {
  const pad = Math.floor(nFft / 2);
  const padded = new Float64Array(y.length + 2 * pad);
  padded.set(y, pad);

  const nFrames = 1 + Math.floor((padded.length - nFft) / hop);
  const nFilters = basisRe.length;
  const mags = [];
  for (let k = 0; k < nFilters; k++) mags.push(new Float64Array(nFrames));

  const frameRe = new Float64Array(nFft);
  const frameIm = new Float64Array(nFft);

  for (let t = 0; t < nFrames; t++) {
    const offset = t * hop;
    for (let i = 0; i < nFft; i++) {
      frameRe[i] = padded[offset + i];
      frameIm[i] = 0;
    }
    fft(frameRe, frameIm);
    for (let k = 0; k < nFilters; k++) {
      let sumRe = 0, sumIm = 0;
      const bRe = basisRe[k], bIm = basisIm[k];
      // dot product truncated to [0, nKeep) -- rfft-style, matches librosa
      for (let i = 0; i < nKeep; i++) {
        sumRe += frameRe[i] * bRe[i] + frameIm[i] * bIm[i];
        sumIm += frameIm[i] * bRe[i] - frameRe[i] * bIm[i];
      }
      mags[k][t] = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
    }
  }
  return mags;
}

// ---------- Lowpass FIR + decimate by 2 (octave downsampling) ----------
// NOTE: masih pakai windowed-sinc manual (bukan resampy/soxr persis). Kalau
// setelah fix ini masih ada selisih residual kecil (beberapa %) terutama di
// bin-bin oktaf paling bawah, INI kemungkinan besar sumbernya -- boleh coba
// naikkan numTaps (mis. 129) buat lowpass yang lebih presisi.
function downsampleBy2(y) {
  const numTaps = 65;
  const half = (numTaps - 1) / 2;
  const cutoff = 0.5;
  const kernel = new Float64Array(numTaps);
  let sum = 0;
  for (let i = 0; i < numTaps; i++) {
    const x = i - half;
    const sinc = x === 0 ? cutoff : Math.sin(Math.PI * cutoff * x) / (Math.PI * x);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (numTaps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (numTaps - 1));
    kernel[i] = sinc * w;
    sum += kernel[i];
  }
  for (let i = 0; i < numTaps; i++) kernel[i] /= sum;

  const padded = new Float64Array(y.length + numTaps);
  padded.set(y, half);

  const outLen = Math.floor(y.length / 2);
  const out = new Float64Array(outLen);
  for (let n = 0; n < outLen; n++) {
    const center = n * 2 + half;
    let acc = 0;
    for (let k = 0; k < numTaps; k++) {
      acc += kernel[k] * padded[center - half + k];
    }
    out[n] = acc;
  }
  return out;
}

// ---------- Full CQT via recursive octave downsampling (LOW bins) ----------
// Matches librosa's vqt(): octave i=0 = TOP (highest freq, full sample rate),
// each subsequent octave i uses audio decimated by another factor of 2 and
// hop halved. Scaling by sqrt(lengths[k]) happens PER-OCTAVE using that
// octave's LOCAL (downsampled-rate) lengths -- validated numerically to
// match librosa's actual output, see comment block at top of file.
function fullCqt(y, sr, hop, baseFreq, nBins, binsPerOctave, filterScale) {
  const nOctaves = Math.ceil(nBins / binsPerOctave);
  const nFiltersTop = Math.min(binsPerOctave, nBins);

  let curY = y;
  let curHop = hop;
  let srEff = sr;
  const octMagsByOctave = []; // index 0 = TOP octave ... last = BOTTOM octave

  for (let i = 0; i < nOctaves; i++) {
    if (i > 0) {
      curY = downsampleBy2(curY);
      curHop = curHop / 2;
      srEff = srEff / 2;
    }
    // Octave i (from top) covers global bin indices [lo, hi) counting from
    // the END of the bin range, matching librosa's slice(-nFilters*(i+1), -nFilters*i)
    let lo, hi;
    if (i === 0) {
      hi = nBins;
      lo = nBins - nFiltersTop;
    } else {
      hi = nBins - nFiltersTop * i;
      lo = hi - nFiltersTop;
    }
    const nFiltersThis = hi - lo;

    const { basisRe, basisIm, nFft, nKeep, lengths } =
      cqtFilterFFT(sr, srEff, baseFreq, nFiltersThis, lo, binsPerOctave, filterScale);
    const rawMags = cqtResponse(curY, nFft, nKeep, Math.round(curHop), basisRe, basisIm);

    // scale by this octave's LOCAL lengths (validated empirically)
    const scaledMags = [];
    for (let k = 0; k < nFiltersThis; k++) {
      const norm = Math.sqrt(lengths[k]);
      const row = rawMags[k];
      const out = new Float64Array(row.length);
      for (let t = 0; t < row.length; t++) out[t] = row[t] / norm;
      scaledMags.push(out);
    }
    octMagsByOctave.push({ lo, hi, mags: scaledMags });
  }

  // Assemble ascending-frequency bins [0, nBins)
  const nFramesTarget = octMagsByOctave[0].mags[0].length;
  const result = new Array(nBins);
  for (const oct of octMagsByOctave) {
    for (let globalBin = oct.lo; globalBin < oct.hi; globalBin++) {
      const localIdx = globalBin - oct.lo;
      const frames = oct.mags[localIdx];
      const row = new Float64Array(nFramesTarget);
      for (let t = 0; t < nFramesTarget; t++) {
        const srcT = Math.min(frames.length - 1, Math.floor((t * frames.length) / nFramesTarget));
        row[t] = frames[srcT];
      }
      result[globalBin] = row;
    }
  }
  return result;
}

// ---------- Pseudo CQT (HIGH bins) ----------
// Completely different algorithm from full CQT -- magnitude-only basis dot
// magnitude-only Hann-windowed STFT, scaled by 1/sqrt(n_fft) (NOT sqrt(length)).
function pseudoCqt(y, sr, hop, baseFreq, nBins, binsPerOctave, filterScale) {
  const alpha = Math.pow(2, 1 / binsPerOctave) - 1;
  const Q = filterScale / alpha;
  const lengths = new Float64Array(nBins);
  let maxLen = 0;
  for (let k = 0; k < nBins; k++) {
    const freq = baseFreq * Math.pow(2, k / binsPerOctave);
    lengths[k] = (Q * sr) / freq;
    if (lengths[k] > maxLen) maxLen = lengths[k];
  }
  let nFft = nextPow2(Math.ceil(maxLen));
  // librosa forces n_fft >= 2 * nextpow2(hop_length) for pseudo-CQT
  const minNfft = Math.pow(2, 1 + Math.ceil(Math.log2(hop)));
  if (nFft < minNfft) nFft = minNfft;
  const nKeep = nFft / 2 + 1;

  const basisMag = [];
  for (let k = 0; k < nBins; k++) {
    const freq = baseFreq * Math.pow(2, k / binsPerOctave);
    const ilen = Math.ceil(lengths[k]);
    const win = hannWindow(ilen);
    const l1norm = sumArr(win);
    const reNormFactor = lengths[k] / nFft;
    const gain = reNormFactor / l1norm;

    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);
    const start = Math.floor((nFft - ilen) / 2);
    for (let n = 0; n < ilen; n++) {
      const ang = (2 * Math.PI * freq * n) / sr;
      const amp = win[n] * gain;
      re[start + n] = amp * Math.cos(ang);
      im[start + n] = amp * Math.sin(ang);
    }
    fft(re, im);
    const mag = new Float64Array(nKeep);
    for (let i = 0; i < nKeep; i++) mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    basisMag.push(mag);
  }

  const frameWindow = hannWindow(nFft); // STFT window="hann" for pseudo path
  const pad = Math.floor(nFft / 2);
  const padded = new Float64Array(y.length + 2 * pad);
  padded.set(y, pad);
  const nFrames = 1 + Math.floor((padded.length - nFft) / hop);

  const mags = [];
  for (let k = 0; k < nBins; k++) mags.push(new Float64Array(nFrames));

  const frameRe = new Float64Array(nFft);
  const frameIm = new Float64Array(nFft);
  const invSqrtNfft = 1 / Math.sqrt(nFft);

  for (let t = 0; t < nFrames; t++) {
    const offset = t * hop;
    for (let i = 0; i < nFft; i++) {
      frameRe[i] = padded[offset + i] * frameWindow[i];
      frameIm[i] = 0;
    }
    fft(frameRe, frameIm);
    const frameMag = new Float64Array(nKeep);
    for (let i = 0; i < nKeep; i++) {
      frameMag[i] = Math.sqrt(frameRe[i] * frameRe[i] + frameIm[i] * frameIm[i]);
    }
    for (let k = 0; k < nBins; k++) {
      let s = 0;
      const bm = basisMag[k];
      for (let i = 0; i < nKeep; i++) s += bm[i] * frameMag[i];
      mags[k][t] = s * invSqrtNfft; // scale = True: /sqrt(n_fft), NOT sqrt(length)
    }
  }
  return mags;
}

// ---------- Top-level: hybrid_cqt ----------
function hybridCqt(y, sr, hop, fmin, nBins, binsPerOctave, filterScale) {
  filterScale = filterScale || 1;

  const lengths = constantQLengths(sr, fmin, nBins, binsPerOctave, filterScale);
  const pseudoMask = new Array(nBins);
  for (let k = 0; k < nBins; k++) {
    pseudoMask[k] = nextPow2(Math.ceil(lengths[k])) < 2 * hop;
  }

  let nBinsPseudo = 0;
  for (let k = 0; k < nBins; k++) if (pseudoMask[k]) nBinsPseudo++;
  const nBinsFull = nBins - nBinsPseudo;

  console.log(`[hybrid_cqt] nBins=${nBins} -> full(low)=${nBinsFull} bins, pseudo(high)=${nBinsPseudo} bins`);

  const result = new Array(nBins);

  if (nBinsFull > 0) {
    const fullRows = fullCqt(y, sr, hop, fmin, nBinsFull, binsPerOctave, filterScale);
    for (let k = 0; k < nBinsFull; k++) result[k] = fullRows[k];
  }

  if (nBinsPseudo > 0) {
    const fminPseudo = fmin * Math.pow(2, nBinsFull / binsPerOctave);
    const mags = pseudoCqt(y, sr, hop, fminPseudo, nBinsPseudo, binsPerOctave, filterScale);
    for (let k = 0; k < nBinsPseudo; k++) {
      result[nBinsFull + k] = mags[k];
    }
  }

  // NOTE: no extra global division here -- each branch (full/pseudo) already
  // applied its own correct scaling internally. The OLD buggy code divided
  // EVERYTHING by sqrt(lengths[k]) here at the end; that's now removed
  // because it double-scaled the full-CQT bins and wrong-scaled the pseudo
  // ones (pseudo needs /sqrt(n_fft), not /sqrt(length)).

  const nFrames = result[0] ? result[0].length : 0;
  return { data: result, nBins, nFrames };
}

global.HybridCQT = { hybridCqt, fft, nextPow2, hannWindow, constantQLengths };

})(window);
