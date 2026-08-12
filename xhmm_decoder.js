// xhmm_decoder.js
// Port JS dari XHMMDecoder (xhmm_ismir.py). Scope: cuma jalur yang benar-benar
// dipakai pipeline asli, yaitu chord_recognition.py:
//   hmm.decode_to_chordlab(entry, probs, False)
// -> use_layer_decode=False, use_beats=False (default), use_downbeats=False (default)
// Artinya cabang triad_decode()/layer_decode()/get_triad_bass_obs() dan
// beat-tracking di __get_beat_arr() TIDAK dipakai di jalur ini, jadi belum
// diport (biar scope kecil & fokus). Kalau nanti butuh beat-synced decode,
// bagian itu perlu ditambah menyusul — jangan diasumsikan sudah ada.
//
// Catatan model: output ChordNet/ChordNetCNN (chordnet_ismir_naive.py) semua
// 2D (n_frame, dim) — bukan 3D per-root — jadi cabang
// `if(len(suffix_probs[i].shape)==3)` di Python tidak pernah kepakai di
// model ini dan sengaja tidak diport.

(function (global) {
  'use strict';

  const ComplexChord =
    typeof module !== 'undefined' && module.exports
      ? require('./complex_chord.js')
      : global.ComplexChord;

  const { NUM_TO_ABS_SCALE, Chord, shiftComplexChordArray } = ComplexChord;

  // Wrapper sederhana buat array probabilitas 2D (n_frame, dim) flat row-major.
  // data: Float32Array, dims: [nFrame, dim]
  class Matrix2D {
    constructor(data, dims) {
      this.data = data;
      this.shape = dims;
    }
    get(t, d) {
      return this.data[t * this.shape[1] + d];
    }
  }

  function argmaxRow(arr, offset, len) {
    let bestIdx = 0;
    let bestVal = arr[offset];
    for (let i = 1; i < len; i++) {
      const v = arr[offset + i];
      if (v > bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  class XHMMDecoder {
    /**
     * @param {object} opts
     * @param {number} [opts.diffTransPenalty=30.0]
     * @param {number[]} [opts.beatTransPenalty=[15.0,45.0,100.0]] - belum dipakai (use_beats=False)
     * @param {string[]} opts.templateLines - isi file *_chord_list.txt, split per baris
     * @param {boolean} [opts.useBass=true]
     * @param {boolean} [opts.use7=true]
     * @param {boolean} [opts.useExtended=true]
     * @param {number} [opts.hopLength=512]
     * @param {number} [opts.sr=22050]
     */
    constructor(opts) {
      this.diffTransPenalty = opts.diffTransPenalty != null ? opts.diffTransPenalty : 30.0;
      this.beatTransPenalty = opts.beatTransPenalty || [15.0, 45.0, 100.0];
      this.useBass = opts.useBass !== false;
      this.use7 = opts.use7 !== false;
      this.useExtended = opts.useExtended !== false;
      this.hopLength = opts.hopLength || 512;
      this.sr = opts.sr || 22050;
      this._initKnownChordNames(opts.templateLines);
    }

    // setara __init_known_chord_names()
    _initKnownChordNames(templateLines) {
      const knownChordArrayPool = new Map(); // key: "n0,n1,...,n5" -> chord name
      const knownTriadBassSet = new Map(); // key: "n0,n1" -> [n0,n1]

      for (let rawLine of templateLines) {
        const chordName = rawLine.trim();
        if (!chordName) continue;
        if (chordName.indexOf('/') >= 0 && !this.useBass) continue;
        if (chordName.indexOf(':') >= 0) {
          const tokens = chordName.split(':');
          if (tokens[0] !== 'C') throw new Error('assert tokens[0]==C failed for ' + chordName);
          const c = new Chord(chordName);
          const array = c.toArray();
          if (array.indexOf(-2) >= 0) continue;
          for (let shift = 0; shift < 12; shift++) {
            const shiftName = NUM_TO_ABS_SCALE[shift] + ':' + tokens[1];
            const shiftArray = shiftComplexChordArray(array, shift);
            const key = shiftArray.join(',');
            if (knownChordArrayPool.has(key)) continue;
            knownChordArrayPool.set(key, shiftName);
            const triadBassKey = shiftArray.slice(0, 2).join(',');
            if (!knownTriadBassSet.has(triadBassKey)) {
              knownTriadBassSet.set(triadBassKey, shiftArray.slice(0, 2));
            }
          }
        }
      }

      this.knownChordArray = [[[0, -1, -1, -1, -1, -1], 'N']];
      for (const [key, name] of knownChordArrayPool) {
        this.knownChordArray.push([key.split(',').map(Number), name]);
      }

      this.knownTriadBass = [[0, -1]];
      for (const [, arr] of knownTriadBassSet) {
        this.knownTriadBass.push(arr);
      }
    }

    // setara get_chord_tag_obs()
    // probList: [probTriad, probBass, prob7, prob9, prob11, prob13] semua Matrix2D
    getChordTagObs(probList, triadRestriction) {
      const suffixProbs = [probList[2], probList[3], probList[4], probList[5]];
      const resultNames = [];
      const resultArrays = [];

      for (const [array, name] of this.knownChordArray) {
        let inRange = true;
        for (let i = 0; i < 6; i++) {
          const p = probList[i];
          if (p != null && array[i] >= p.shape[1]) {
            inRange = false;
            break;
          }
        }
        if (inRange) {
          if (array[0] < 0) throw new Error('assert array[0]>=0 failed');
          resultNames.push(name);
          resultArrays.push(array.slice());
        }
      }

      const nChord = resultArrays.length;
      const nFrame = probList[0].shape[0];

      // bass adjust: array[1] += 1 (index 0 = "no bass")
      for (const arr of resultArrays) arr[1] += 1;

      const resultLogProb = new Float32Array(nFrame * nChord);
      const probTriad = probList[0];
      for (let t = 0; t < nFrame; t++) {
        const rowOff = t * nChord;
        for (let c = 0; c < nChord; c++) {
          resultLogProb[rowOff + c] = Math.log(probTriad.get(t, resultArrays[c][0]));
        }
      }

      if (this.useBass) {
        const probBass = probList[1];
        for (let c = 0; c < nChord; c++) {
          const bassId = resultArrays[c][1];
          if (bassId >= 0) {
            for (let t = 0; t < nFrame; t++) {
              resultLogProb[t * nChord + c] += Math.log(probBass.get(t, bassId));
            }
          }
        }
      }

      for (let i = 0; i < 4; i++) {
        if ((i === 0 && this.use7) || (i > 0 && this.useExtended)) {
          const sp = suffixProbs[i];
          for (let c = 0; c < nChord; c++) {
            const val = resultArrays[c][i + 2];
            if (val >= 0) {
              for (let t = 0; t < nFrame; t++) {
                resultLogProb[t * nChord + c] += Math.log(sp.get(t, val));
              }
            }
          }
        }
      }

      if (triadRestriction) {
        for (let t = 0; t < nFrame; t++) {
          const rowOff = t * nChord;
          const restr = triadRestriction[t]; // [triadId, bassId(sudah +1)]
          for (let c = 0; c < nChord; c++) {
            if (resultArrays[c][0] !== restr[0] || resultArrays[c][1] !== restr[1]) {
              resultLogProb[rowOff + c] = -Infinity;
            }
          }
        }
      }

      return { names: resultNames, logProb: resultLogProb, nFrame, nChord };
    }

    // setara decode() — Viterbi DP murni
    decode(probList, beatArr, triadRestriction) {
      const { names: resultNames, logProb, nFrame, nChord } = this.getChordTagObs(probList, triadRestriction || null);

      const dp = new Float32Array(nFrame * nChord);
      const dpMaxAt = new Int32Array(nFrame);
      const pre = new Int32Array(nFrame * nChord);

      // dp[0,1:] -= inf ; dp[0,:] += logProb[0,:]
      // -> hanya index 0 ("N") yang punya nilai dp finite di frame pertama.
      for (let c = 0; c < nChord; c++) {
        dp[c] = c === 0 ? logProb[c] : -Infinity;
      }
      dpMaxAt[0] = argmaxRow(dp, 0, nChord);
      for (let c = 0; c < nChord; c++) pre[c] = -1;

      for (let t = 1; t < nFrame; t++) {
        const rowOff = t * nChord;
        const prevOff = (t - 1) * nChord;
        if (beatArr[t]) {
          const penalty = beatArr[t] === 1 ? this.diffTransPenalty : this.beatTransPenalty[beatArr[t] - 2];
          const diffTrans = dp[prevOff + dpMaxAt[t - 1]] - penalty;
          for (let c = 0; c < nChord; c++) {
            const sameTrans = dp[prevOff + c];
            const useSame = sameTrans > diffTrans;
            const best = Math.max(diffTrans, sameTrans);
            dp[rowOff + c] = best + logProb[rowOff + c];
            pre[rowOff + c] = useSame ? c : dpMaxAt[t - 1];
          }
        } else {
          for (let c = 0; c < nChord; c++) {
            dp[rowOff + c] = dp[prevOff + c] + logProb[rowOff + c];
            pre[rowOff + c] = c;
          }
        }
        dpMaxAt[t] = argmaxRow(dp, rowOff, nChord);
      }

      const decodeIds = new Int32Array(nFrame);
      decodeIds[nFrame - 1] = dpMaxAt[nFrame - 1];
      for (let t = nFrame - 2; t >= 0; t--) {
        decodeIds[t] = pre[(t + 1) * nChord + decodeIds[t + 1]];
      }

      const out = new Array(nFrame);
      for (let t = 0; t < nFrame; t++) out[t] = resultNames[decodeIds[t]];
      return out;
    }

    // setara __get_beat_arr() untuk kasus use_beats=False (satu-satunya yang
    // dipakai chord_recognition.py). beat_arr konstan 1 di semua frame.
    _getBeatArr(length) {
      return new Int8Array(length).fill(1);
    }

    // setara decode_to_chordlab(entry, probs, use_layer_decode=False)
    decodeToChordlab(probList, useLayerDecode) {
      if (useLayerDecode) {
        throw new Error('layer_decode belum diport (tidak dipakai pipeline saat ini: chord_recognition.py selalu panggil use_layer_decode=False)');
      }
      const nFrame = probList[0].shape[0];
      const beatArr = this._getBeatArr(nFrame);
      const deltaTime = this.hopLength / this.sr;
      const decodeTags = this.decode(probList, beatArr, null);

      const result = [];
      let lastFrame = 0;
      const n = decodeTags.length;
      for (let i = 0; i < n; i++) {
        if (i + 1 === n || decodeTags[i + 1] !== decodeTags[i]) {
          result.push([lastFrame * deltaTime, (i + 1) * deltaTime, decodeTags[i]]);
          lastFrame = i + 1;
        }
      }
      return result;
    }
  }

  const XHMM = { XHMMDecoder, Matrix2D };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = XHMM;
  } else {
    global.XHMM = XHMM;
  }
})(typeof self !== 'undefined' ? self : this);
