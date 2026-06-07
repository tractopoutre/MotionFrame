// MotionFrame web bench harness.
// Generates synthetic TGA frames in JS, posts them to the wasm worker, and
// times the round trip from postMessage(Generate) to MsgFromWorker::Done.
// Reports median over K iterations after a warmup run.
//
// Read results from `document.title` (set to `BENCH:DONE:median=...`) or
// from the `<pre id="out">` element. Console also gets each line.

const params = new URLSearchParams(location.search);
const W = parseInt(params.get('w') || '256', 10);
const H = parseInt(params.get('h') || '256', 10);
const N_FRAMES = parseInt(params.get('n') || '64', 10);
const ATLAS_COLS = parseInt(params.get('cols') || '8', 10);
const ATLAS_ROWS = parseInt(params.get('rows') || '8', 10);
const ATLAS_PX = parseInt(params.get('aw') || '128', 10);
const RUNS = parseInt(params.get('runs') || '5', 10);
const WARMUP = parseInt(params.get('warmup') || '1', 10);
const FRAME_SKIP = parseInt(params.get('skip') || '0', 10);

const statusEl = document.getElementById('status');
const outEl = document.getElementById('out');

function log(line) {
  console.log(line);
  outEl.textContent += line + '\n';
}
function setStatus(s, cls = '') {
  statusEl.textContent = s;
  statusEl.className = cls;
}

// Encode an RGBA frame as uncompressed TGA (type 2, top-down BGRA32).
// The Rust side decodes via image::open, which accepts this layout.
function encodeTga(w, h, rgba) {
  const out = new Uint8Array(18 + w * h * 4);
  out[2] = 2;                              // image_type: uncompressed true-color
  out[12] = w & 0xff; out[13] = (w >> 8) & 0xff;
  out[14] = h & 0xff; out[15] = (h >> 8) & 0xff;
  out[16] = 32;                            // 32 bits per pixel
  out[17] = 0x28;                          // top-down, 8 alpha bits
  let p = 18;
  for (let i = 0; i < rgba.length; i += 4) {
    // RGBA in input → BGRA in TGA
    out[p++] = rgba[i + 2];
    out[p++] = rgba[i + 1];
    out[p++] = rgba[i + 0];
    out[p++] = rgba[i + 3];
  }
  return out;
}

// Synthetic gradient frames with horizontal motion. Designed to give Farneback
// real flow to compute (not all-zero, not random noise) so the bench reflects
// realistic compute cost.
function makeFrames(w, h, n) {
  const frames = [];
  for (let f = 0; f < n; f++) {
    const offset = f * 2.0;
    const rgba = new Uint8Array(w * h * 4);
    let i = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = Math.min(255, ((x + offset) / w) * 255) | 0;
        rgba[i++] = v;                       // R
        rgba[i++] = ((y / h) * 255) | 0;     // G
        rgba[i++] = 0;                       // B
        rgba[i++] = 255;                     // A
      }
    }
    frames.push({
      name: `synth-${String(f).padStart(3, '0')}.tga`,
      bytes: encodeTga(w, h, rgba),
    });
  }
  return frames;
}

// GenerateOptions matching the Rust default-ish but with bench-specific dims.
function makeOptions() {
  return {
    output_frames: ATLAS_COLS * ATLAS_ROWS,
    frame_skip: FRAME_SKIP,
    tile_pixel_width: ATLAS_PX,
    atlas_dims: [ATLAS_COLS, ATLAS_ROWS],
    stagger_pack: false,
    analyze_skipped_frames: true,
    premultiplied_alpha: false,
    alpha_threshold: 8,
    apply_alpha_post_mask: true,
    farneback: {
      pyr_scale: 0.5,
      levels: 8,
      winsize: 15,
      iterations: 5,
      poly_n: 5,
      poly_sigma: 1.5,
      use_gaussian: false,
    },
    motion_vector_encoding: 'R8G8Remap01',
    is_loop: false,
    halve_motion_vector: false,
    extrude: 0,
    resize_algorithm: 'Cubic',
    non_loop_tail: 'Hold',
    integration: 'Heun',
    remap_interp: 'Bicubic',
  };
}

function spawnWorker() {
  return new Worker('./worker/worker.js', { type: 'module' });
}

// Run one generate and resolve the elapsed ms (postMessage → Done).
function runOnce(worker, frames, options) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const onMsg = (ev) => {
      const m = ev.data;
      if (!m || !m.type) return;
      if (m.type === 'Done') {
        worker.removeEventListener('message', onMsg);
        resolve(performance.now() - t0);
      } else if (m.type === 'Error') {
        worker.removeEventListener('message', onMsg);
        reject(new Error(m.data || 'worker error'));
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'Generate', frames, options });
  });
}

function stats(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((s, v) => s + v, 0) / n;
  const variance = sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const stddev = Math.sqrt(variance);
  const median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  return {
    median,
    mean,
    stddev,
    min: sorted[0],
    max: sorted[n - 1],
    p25: sorted[Math.floor(n * 0.25)],
    p75: sorted[Math.floor(n * 0.75)],
  };
}

async function main() {
  document.title = 'BENCH:STARTING';
  log(`config: ${N_FRAMES}f × ${W}×${H}, atlas ${ATLAS_COLS}×${ATLAS_ROWS} @ ${ATLAS_PX}px, skip=${FRAME_SKIP}, runs=${RUNS} (+${WARMUP} warmup)`);
  log(`hardwareConcurrency: ${navigator.hardwareConcurrency}`);
  log(`crossOriginIsolated: ${self.crossOriginIsolated}`);

  if (!self.crossOriginIsolated) {
    setStatus('cross-origin isolation missing — SharedArrayBuffer unavailable; rayon will single-thread', 'bad');
  }

  setStatus('generating frames…');
  const tGen0 = performance.now();
  const frames = makeFrames(W, H, N_FRAMES);
  const tGen = performance.now() - tGen0;
  log(`encoded ${frames.length} TGA frames in ${tGen.toFixed(1)} ms (${(frames[0].bytes.byteLength / 1024).toFixed(1)} KB/frame)`);

  const options = makeOptions();
  const worker = spawnWorker();

  setStatus('running warmup…');
  for (let i = 0; i < WARMUP; i++) {
    const ms = await runOnce(worker, frames, options);
    log(`warmup ${i + 1}/${WARMUP}: ${ms.toFixed(2)} ms`);
  }

  const samples = [];
  for (let i = 0; i < RUNS; i++) {
    setStatus(`run ${i + 1}/${RUNS}…`);
    const ms = await runOnce(worker, frames, options);
    samples.push(ms);
    log(`run ${i + 1}/${RUNS}: ${ms.toFixed(2)} ms`);
    document.title = `BENCH:RUNNING:${i + 1}/${RUNS}`;
  }

  const s = stats(samples);
  log('');
  log(`samples: ${samples.map((v) => v.toFixed(1)).join(', ')} ms`);
  log(`median:  ${s.median.toFixed(2)} ms`);
  log(`mean:    ${s.mean.toFixed(2)} ms (±${s.stddev.toFixed(2)})`);
  log(`min/p25/p75/max: ${s.min.toFixed(2)} / ${s.p25.toFixed(2)} / ${s.p75.toFixed(2)} / ${s.max.toFixed(2)}`);

  setStatus('done', 'ok');
  document.title = `BENCH:DONE:median=${s.median.toFixed(2)}:mean=${s.mean.toFixed(2)}:stddev=${s.stddev.toFixed(2)}:n=${RUNS}`;
  worker.terminate();
}

main().catch((e) => {
  console.error(e);
  setStatus(`error: ${e.message || e}`, 'bad');
  document.title = `BENCH:ERROR:${e.message || e}`;
});
