# Fase 4 — Pipeline Penuh di Browser

## Apa yang berubah dari sebelumnya

`index.html` lama manggil `fetch('/api/analyze')` ke Flask. Sekarang diganti
total jadi pipeline lokal:

```
File audio (input dropzone)
  -> audio-decode.js   (Tahap 1, sudah divalidasi: offset 0, korelasi 0.9999)
  -> hybrid_cqt.js      (Fase 2, sudah divalidasi: 99.04% match chord ke baseline)
  -> onnxruntime-web     ensemble 5 model ChordNet_s0..s4.onnx (Fase 1)
  -> viterbi-worker.js  (Fase 3, jalan di Web Worker biar UI gak freeze)
  -> render UI (index.html, logic lama dipertahankan apa adanya)
```

Fitur share-link (`?job=<id>`) dari versi Flask **sengaja dihapus** — itu
butuh server nyimpen hasil, bertentangan sama tujuan porting ini.

## Isi paket

| File | Peran |
|---|---|
| `index.html` | UI (player, transpose, diagram, export MIDI — semua logic lama dipertahankan, cuma bagian analyze yang diganti) |
| `pipeline.js` | **Baru.** Orkestrator: decode → CQT → inference ensemble → kirim ke worker → BPM |
| `viterbi-worker.js` | **Baru.** Web Worker isolasi decode Viterbi |
| `audio-decode.js` | Tahap 1 (sudah tervalidasi kemarin) |
| `hybrid_cqt.js` | Fase 2 (sudah tervalidasi) |
| `xhmm_decoder.js`, `complex_chord.js` | Fase 3 (sudah tervalidasi via Fase 2b) |
| `submission_chord_list.txt` | Template chord dictionary |
| `models/ChordNet_s0.onnx` … `s4.onnx` | Fase 1 (dari `chord-engine-fase1-output.zip` kamu) |

Semua parameter (SR=22050, hop=512, fmin=F#0, n_bins=288, bins_per_octave=36,
SPEC_DIM=252, OFFSET=18) disamakan persis dengan `fase2b_full_pipeline_test.js`
yang sudah terbukti 99.04% match ke baseline Python.

## Cara jalanin (WAJIB pakai server lokal, bukan double-click file)

Buka `index.html` langsung lewat `file://` **tidak akan jalan** — browser
modern nge-block `fetch()` (buat `submission_chord_list.txt` dan file
`.onnx`) dan `new Worker(...)` di bawah protokol `file://`. Ini bukan bug,
ini pembatasan keamanan browser yang sama untuk semua static site.

Jalankan static server sederhana dari dalam folder ini:

```bash
# Python (biasanya sudah ada)
python -m http.server 8000

# atau kalau ada Node
npx serve .
```

Lalu buka `http://localhost:8000` di browser.

**Ini BUKAN "butuh server buat inference"** — server di atas cuma nge-serve
file statis (persis prinsip yang sama kayak GitHub Pages/Vercel/Netlify nanti
di Fase 7). Semua komputasi (CQT, inference ONNX, Viterbi) tetap 100% jalan
di browser/CPU kamu, bukan di server itu.

## Yang perlu kamu tau sebelum lanjut ke Fase 6

### 1. `onnxruntime-web` di-load dari CDN (jsdelivr)

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js"></script>
```

Ini artinya app butuh koneksi internet buat load library-nya (bukan buat
inference — inference-nya tetap lokal). Kalau kamu mau bener-bener 100%
offline-capable (misal buat testing tanpa internet, atau menghindari
dependency ke pihak ketiga sama sekali), unduh manual folder `dist/` dari
paket npm `onnxruntime-web`, taruh di `vendor/onnxruntime-web/`, lalu ganti
`<script src>` dan `ort.env.wasm.wasmPaths` di `index.html` ke path lokal
itu. Saya belum lakuin ini sekarang karena belum tau preferensi kamu — kasih
tau kalau mau saya siapin versi self-hosted-nya.

### 2. Estimasi BPM itu APROKSIMASI, belum divalidasi

`pipeline.js` punya `estimateBpm()` — estimasi tempo pakai onset envelope +
autocorrelation sederhana, **bukan port dari `librosa.beat.beat_track()`**
(algoritmanya beda total, dan saya belum bandingkan angkanya ke baseline
Python). Ini saya tulis sendiri karena backend Python lama (`detect_bpm()`
di `app.py`) pakai `librosa.beat.beat_track()` yang gak ada equivalent
langsung yang gampang di-port 1:1 ke JS tanpa effort validasi terpisah.

**Ini TIDAK memengaruhi akurasi deteksi chord** — `XHMMDecoder` di pipeline
ini jalan dengan `use_beats=False` (lihat komentar di `xhmm_decoder.js`),
jadi BPM sama sekali gak dipakai di proses decode chord. BPM cuma dipakai
buat:
- Badge angka BPM di UI
- Tempo default file MIDI (fallback 120 BPM kalau gagal deteksi)

Kalau kamu butuh BPM yang presisi/tervalidasi (misal buat DAW sync yang
ketat), ini bagian yang perlu paket validasi terpisah dulu (generate
baseline BPM dari `librosa.beat.beat_track()` beberapa lagu, bandingkan ke
`estimateBpm()`) — mirip pola Fase 2b kemarin. Kasih tau kalau mau saya
siapin itu.

### 3. Belum ada progress bar granular per-detik

`onProgress` callback sekarang cuma laporan per-tahap (decode → CQT →
inference model ke-N/5 → Viterbi → BPM), bukan persentase halus. Untuk lagu
pendek (~20 detik kayak yang dites Fase 2b) ini kerasa instan. Untuk lagu
3-5 menit, inference 5 model ONNX + CQT bisa makan beberapa detik sampai
puluhan detik tergantung CPU — kalau kamu rasa UX-nya kurang informatif
setelah dicoba, kabari saya, saya bisa tambah progress lebih detail.

## Belum tersentuh (sesuai scope Fase 4 di planning)

- Fase 6 (validasi akurasi end-to-end pakai lagu-lagu tes) — belum dites
  sama sekali dengan pipeline utuh ini (yang kemarin di Fase 2b itu simulasi
  Node.js, bukan lewat `index.html` beneran di browser).
- Fase 7 (deploy: cek ukuran `.onnx` — total 5 file sekitar 9.4MB gabungan,
  jauh di bawah limit 100MB GitHub Pages, kemungkinan besar gak perlu
  quantization; compression; caching model di IndexedDB/Cache API) — belum
  dikerjakan.

## Langkah selanjutnya yang saya sarankan

Jalankan `index.html` ini via local server (lihat "Cara jalanin" di atas),
upload lagu yang sama yang dipakai buat baseline Fase 2b/3 kemarin, terus
bandingkan hasil chord di UI dengan `hasil_baseline.lab`. Kalau cocok,
lanjut resmi ke Fase 6 (kumpulin beberapa lagu tes lain buat validasi lebih
luas) sebelum Fase 7 (deploy).
