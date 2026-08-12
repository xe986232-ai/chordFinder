// audio-decode.js
// ============================================================
// Tahap 1 — Ekstraksi PCM mentah dari file audio via Web Audio API,
// disamakan sedekat mungkin dengan `librosa.load(path, sr=22050, mono=True)`.
// ============================================================
//
// CATATAN PENTING — baca sebelum pakai fungsi ini di pipeline produksi
// (Fase 4):
//
// decodeAudioFileToPCM() di bawah TIDAK melakukan trim/shift/koreksi
// offset apa pun secara default. Ini SENGAJA. Kita belum tau apakah
// browser/OS kamu nambahin silence padding di awal decode (khas masalah
// "encoder delay" MP3 -- lihat catatan di planning-porting-chord-engine.md
// Fase 2: "Waspada silence padding dari Web Audio API"), dan kalau iya,
// berapa sample persisnya. Itu HARUS diukur, bukan ditebak.
//
// Cara ukur: pakai fase1b_pcm_validation.html (satu paket sama file ini).
// Itu bandingin PCM hasil decodeAudioFileToPCM() vs reference_pcm.json
// (hasil librosa.load() di Python, dari export_full_reference.py yang
// sudah kamu punya). Hasil pengukuran itu yang nentuin:
//   - offset = 0, diff kecil          -> aman, TIDAK perlu trim
//   - offset = N (konsisten), diff kecil setelah digeser -> perlu trim N
//     sample di awal, tambahkan itu secara eksplisit di pipeline
//   - diff masih besar walau sudah dicoba semua offset di jendela
//     pencarian -> ada masalah lain (bukan cuma silence padding), jangan
//     asal terapkan trim, laporkan hasilnya dulu buat didiagnosis lebih
//     lanjut.
//
// JANGAN import fungsi trim otomatis ke Fase 4 sebelum langkah pengukuran
// di atas dijalankan dan hasilnya jelas.

(function (global) {
  'use strict';

  /**
   * Decode file audio (mp3/wav/m4a/flac/ogg) jadi PCM mono Float32Array
   * pada sample rate target, pakai Web Audio API.
   *
   * Downmix ke mono = rata-rata semua channel per-sample (sama seperti
   * librosa.to_mono(), BUKAN cuma ambil channel kiri).
   *
   * Resample ke targetSr terjadi otomatis di dalam decodeAudioData() --
   * OfflineAudioContext dipakai murni sebagai wadah buat itu; panjang
   * render (argumen ke-2) tidak memengaruhi hasil decode.
   *
   * @param {File|Blob} file
   * @param {number} [targetSr=22050]
   * @returns {Promise<{sr:number, data:Float32Array, numChannelsOriginal:number, durationSec:number}>}
   */
  async function decodeAudioFileToPCM(file, targetSr) {
    targetSr = targetSr || 22050;
    const arrayBuffer = await file.arrayBuffer();

    const OfflineCtx = global.OfflineAudioContext || global.webkitOfflineAudioContext;
    if (!OfflineCtx) throw new Error('Browser ini tidak dukung OfflineAudioContext.');

    // Panjang render (arg ke-2) cuma placeholder minimal -- decodeAudioData
    // tidak dibatasi olehnya, dia decode+resample seluruh file.
    const offlineCtx = new OfflineCtx(1, 1, targetSr);
    const audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);

    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;

    let mono;
    if (numChannels === 1) {
      mono = audioBuffer.getChannelData(0).slice();
    } else {
      mono = new Float32Array(length);
      const chans = [];
      for (let c = 0; c < numChannels; c++) chans.push(audioBuffer.getChannelData(c));
      for (let i = 0; i < length; i++) {
        let sum = 0;
        for (let c = 0; c < numChannels; c++) sum += chans[c][i];
        mono[i] = sum / numChannels;
      }
    }

    return {
      sr: audioBuffer.sampleRate,
      data: mono,
      numChannelsOriginal: numChannels,
      durationSec: audioBuffer.duration,
    };
  }

  /**
   * Cari offset sample terbaik antara dua PCM array pakai normalized
   * cross-correlation di jendela awal saja (buat diagnosis silence
   * padding/encoder delay). MURNI DIAGNOSIS -- tidak mengubah data apa pun.
   *
   * offset > 0 artinya pcmB "lebih maju" dibanding pcmA (pcmB perlu
   * digeser mundur / dipotong offset sample di depan biar align ke pcmA).
   * offset < 0 artinya sebaliknya.
   *
   * @param {Float32Array} pcmA - PCM referensi (misal dari librosa)
   * @param {Float32Array} pcmB - PCM yang mau dicek (misal dari browser)
   * @param {number} [maxSearchSamples=4410] - batas geser (~200ms @ 22050Hz)
   * @returns {{bestOffset:number, scoreAtBest:number, scores:Array<{offset:number,score:number}>}}
   */
  function findBestAlignment(pcmA, pcmB, maxSearchSamples) {
    maxSearchSamples = maxSearchSamples != null ? maxSearchSamples : 4410;
    const compareLen = Math.min(pcmA.length, pcmB.length) - maxSearchSamples * 2;
    if (compareLen <= 0) {
      throw new Error('PCM terlalu pendek untuk jendela pencarian ini -- perkecil maxSearchSamples.');
    }

    let bestOffset = 0;
    let bestScore = -Infinity;
    const scores = [];

    for (let offset = -maxSearchSamples; offset <= maxSearchSamples; offset++) {
      let dot = 0, normA = 0, normB = 0;
      const startA = Math.max(0, -offset);
      const startB = Math.max(0, offset);
      const n = Math.min(compareLen, pcmA.length - startA, pcmB.length - startB);
      for (let i = 0; i < n; i++) {
        const a = pcmA[startA + i];
        const b = pcmB[startB + i];
        dot += a * b;
        normA += a * a;
        normB += b * b;
      }
      const denom = Math.sqrt(normA * normB) || 1e-12;
      const score = dot / denom;
      scores.push({ offset, score });
      if (score > bestScore) { bestScore = score; bestOffset = offset; }
    }

    return { bestOffset, scoreAtBest: bestScore, scores };
  }

  /**
   * Index sample pertama yang "bukan diam" (|x| > threshold). Dipakai buat
   * bandingin titik mulai bunyi pertama antar dua PCM, sesuai metode uji
   * wajib di planning-porting-chord-engine.md Fase 2.
   */
  function firstNonSilentIndex(pcm, threshold) {
    threshold = threshold != null ? threshold : 1e-4;
    for (let i = 0; i < pcm.length; i++) {
      if (Math.abs(pcm[i]) > threshold) return i;
    }
    return -1;
  }

  global.AudioDecode = { decodeAudioFileToPCM, findBestAlignment, firstNonSilentIndex };

})(typeof window !== 'undefined' ? window : this);
