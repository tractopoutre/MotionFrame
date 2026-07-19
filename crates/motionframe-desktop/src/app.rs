//! Desktop platform adapter for the shared `MotionFrameApp`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, OnceLock};

use motionframe_engine::io::{sequence, slice_atlas, InMemoryFrames};
use motionframe_engine::pipeline::run::{run_pipeline, run_pipeline_with_gpu};
use motionframe_engine::pipeline::{GenerateOptions, ImageRgba8, PipelineError, Progress};
use motionframe_ui::i18n::{fmt, t, Key, Lang};
use motionframe_ui::platform::{EncodedFrame, GenerationEvent, Platform};

use crate::locale;

const THIRD_PARTY_LICENSES_DEFLATED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/licenses.deflate"));

fn third_party_licenses() -> &'static str {
    static CACHE: OnceLock<String> = OnceLock::new();
    CACHE.get_or_init(|| {
        let bytes = miniz_oxide::inflate::decompress_to_vec(THIRD_PARTY_LICENSES_DEFLATED)
            .expect("THIRD-PARTY-LICENSES blob compressed at build time");
        String::from_utf8(bytes).expect("THIRD-PARTY-LICENSES.md is UTF-8")
    })
}

/// Desktop IO and worker-thread implementation for the shared app.
pub struct DesktopPlatform {
    pending_pick: Option<Vec<EncodedFrame>>,
    gen_rx: Option<mpsc::Receiver<GenerationEvent>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    worker_thread: Option<std::thread::JoinHandle<()>>,
    initial_lang: Lang,
    jp_font_available: bool,
}

impl DesktopPlatform {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let stored_lang = locale::load_or_detect(cc.storage);
        let jp_font_path = locale::probe_jp_font();
        let jp_font_available = jp_font_path.is_some();
        locale::install_fonts(&cc.egui_ctx, jp_font_path.as_deref());

        let initial_lang = if stored_lang == Lang::Ja && !jp_font_available {
            log::warn!(
                "persisted language is Japanese but no JP font found; falling back to English for this session"
            );
            Lang::En
        } else {
            stored_lang
        };

        Self {
            pending_pick: None,
            gen_rx: None,
            cancel_flag: None,
            worker_thread: None,
            initial_lang,
            jp_font_available,
        }
    }

    fn collect_sequence_bytes(path: &Path) -> Option<Vec<EncodedFrame>> {
        let seed = sequence::resolve_seed_file(path)?;
        let filename = seed.file_name().and_then(|n| n.to_str()).unwrap_or("");
        match sequence::detect_pattern(filename) {
            Some((prefix, num_digits, ext)) => {
                let dir = seed.parent().unwrap_or_else(|| Path::new("."));
                let files = sequence::collect_sequence_files(dir, &prefix, num_digits, &ext);
                if files.is_empty() {
                    return Self::load_single(&seed);
                }
                let mut out = Vec::with_capacity(files.len());
                for p in files {
                    let Ok(bytes) = std::fs::read(&p) else {
                        return None;
                    };
                    let name = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("frame")
                        .to_string();
                    out.push(EncodedFrame { name, bytes });
                }
                Some(out)
            }
            None => Self::load_single(&seed),
        }
    }

    fn load_single(path: &Path) -> Option<Vec<EncodedFrame>> {
        let bytes = std::fs::read(path).ok()?;
        let name = path.file_name().and_then(|n| n.to_str())?.to_string();
        Some(vec![EncodedFrame { name, bytes }])
    }

    fn spawn_worker(&mut self, frames: Vec<EncodedFrame>, opts: GenerateOptions) {
        self.cancel_generation();

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_for_worker = Arc::clone(&cancel_flag);
        let (tx, rx) = mpsc::channel::<GenerationEvent>();
        let worker_lang = self.initial_lang;

        let handle = std::thread::spawn(move || {
            let gpu = motionframe_engine::gpu::GpuPipeline::try_init();
            if gpu.is_some() {
                eprintln!("[gpu] using GPU pipeline");
                let _ = tx.send(GenerationEvent::Progress(Progress::Stage {
                    name: "GPU pipeline".into(),
                    fraction: 0.0,
                }));
            } else {
                eprintln!("[gpu] no GPU device, using CPU pipeline");
                let _ = tx.send(GenerationEvent::Progress(Progress::Stage {
                    name: "CPU pipeline".into(),
                    fraction: 0.0,
                }));
            }

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if cancel_for_worker.load(Ordering::Relaxed) {
                    return Err(PipelineError::Cancelled);
                }
                let decoded: Vec<ImageRgba8> = if let Some((cols, rows)) = opts.input_atlas_dims {
                    if frames.len() != 1 {
                        return Err(PipelineError::AtlasFrameCount(frames.len()));
                    }
                    let ef = &frames[0];
                    let path = std::path::PathBuf::from(&ef.name);
                    let src = motionframe_engine::io::decode_image_from_bytes(&ef.name, &ef.bytes)
                        .map_err(|e| PipelineError::DecodeFailed(path, e))?;
                    slice_atlas(&src, cols, rows)?
                } else {
                    let mut v: Vec<ImageRgba8> = Vec::with_capacity(frames.len());
                    for ef in &frames {
                        let path = std::path::PathBuf::from(&ef.name);
                        match motionframe_engine::io::decode_image_from_bytes(&ef.name, &ef.bytes) {
                            Ok(img) => v.push(img),
                            Err(e) => return Err(PipelineError::DecodeFailed(path, e)),
                        }
                    }
                    v
                };
                let source = InMemoryFrames::new(decoded)
                    .map_err(|e| PipelineError::Other(format!("frame source: {e}")))?;
                let tx_progress_lock = std::sync::Mutex::new(tx.clone());
                let progress_fn = |p: Progress| {
                    let _ = tx_progress_lock
                        .lock()
                        .unwrap()
                        .send(GenerationEvent::Progress(p));
                };
                let cancel_fn = || cancel_for_worker.load(Ordering::Relaxed);
                if let Some(gpu) = gpu.as_ref() {
                    run_pipeline_with_gpu(&source, &opts, &progress_fn, &cancel_fn, Some(gpu))
                } else {
                    run_pipeline(&source, &opts, &progress_fn, &cancel_fn)
                }
            }));
            let event = match result {
                Ok(Ok(encode_result)) => GenerationEvent::Done(encode_result),
                Ok(Err(PipelineError::Cancelled)) => GenerationEvent::Cancelled,
                Ok(Err(e)) => GenerationEvent::Error(format!("{e}")),
                Err(panic_payload) => {
                    let msg = panic_payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "Unknown panic".to_string());
                    GenerationEvent::Error(fmt(t(worker_lang, Key::ErrWorkerPanic), &[&msg]))
                }
            };
            let _ = tx.send(event);
        });

        self.cancel_flag = Some(cancel_flag);
        self.gen_rx = Some(rx);
        self.worker_thread = Some(handle);
    }
}

impl Platform for DesktopPlatform {
    fn initial_lang(&self) -> Lang {
        self.initial_lang
    }

    fn jp_font_available(&self) -> bool {
        self.jp_font_available
    }

    fn save_app_state(&mut self, storage: &mut dyn eframe::Storage, lang: Lang) {
        locale::save(storage, lang);
        self.initial_lang = lang;
    }

    fn third_party_licenses(&self) -> Option<&'static str> {
        Some(third_party_licenses())
    }

    fn start_folder_pick(&mut self, _ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new().pick_folder() else {
            self.pending_pick = Some(Vec::new());
            return;
        };
        self.pending_pick = Some(Self::collect_sequence_bytes(&path).unwrap_or_default());
    }

    fn take_folder_pick(&mut self) -> Option<Vec<EncodedFrame>> {
        self.pending_pick.take()
    }

    fn save_outputs(
        &mut self,
        prefix: &str,
        color_atlas: &ImageRgba8,
        motion_atlas: &ImageRgba8,
        metadata_json: &str,
    ) -> Result<(), String> {
        let mut dialog =
            rfd::FileDialog::new().set_title(t(self.initial_lang, Key::SaveOutputsDialogTitle));
        if !prefix.is_empty() {
            dialog = dialog.set_file_name(prefix);
        }
        let Some(path) = dialog.save_file() else {
            return Ok(()); // user cancelled — nothing to report
        };
        let prefix_path = path.with_extension("");
        let prefix_str = prefix_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("output");
        let dir = prefix_path.parent().unwrap_or_else(|| Path::new("."));
        let color_path = dir.join(format!("{prefix_str}_color_atlas.tga"));
        let motion_path = dir.join(format!("{prefix_str}_motion_atlas.tga"));
        let meta_path = dir.join(format!("{prefix_str}_meta.json"));

        // Fail on the first error so a read-only dir / full disk surfaces to the
        // user instead of looking like a successful save.
        motionframe_engine::io::tga::save_tga(&color_path, color_atlas)
            .map_err(|e| format!("{}: {e}", color_path.display()))?;
        motionframe_engine::io::tga::save_tga(&motion_path, motion_atlas)
            .map_err(|e| format!("{}: {e}", motion_path.display()))?;
        std::fs::write(&meta_path, metadata_json)
            .map_err(|e| format!("{}: {e}", meta_path.display()))?;
        Ok(())
    }

    fn start_generation(
        &mut self,
        frames: Vec<EncodedFrame>,
        options: GenerateOptions,
        _ctx: &egui::Context,
    ) {
        self.spawn_worker(frames, options);
    }

    fn take_generation_event(&mut self) -> Option<GenerationEvent> {
        let ev = self.gen_rx.as_ref()?.try_recv().ok()?;
        let is_terminal = !matches!(ev, GenerationEvent::Progress(_));
        if is_terminal {
            if let Some(handle) = self.worker_thread.take() {
                let _ = handle.join();
            }
            self.gen_rx = None;
            self.cancel_flag = None;
        }
        Some(ev)
    }

    fn cancel_generation(&mut self) {
        if let Some(ref f) = self.cancel_flag {
            f.store(true, Ordering::Relaxed);
        }
        // Detach the worker (drop the handle) rather than join()ing on the UI
        // thread: joining would freeze the window until the worker reaches its
        // next cancel checkpoint. The worker observes the flag and exits on its
        // own; its result is discarded because we drop the receiver.
        self.worker_thread = None;
        self.gen_rx = None;
        self.cancel_flag = None;
    }

    fn handle_dropped_files(&mut self, dropped: Vec<egui::DroppedFile>) -> Vec<EncodedFrame> {
        let mut paths: Vec<std::path::PathBuf> =
            dropped.into_iter().filter_map(|d| d.path).collect();
        match paths.len() {
            0 => Vec::new(),
            // A single dropped path: a folder is walked, and a lone file
            // triggers sequence detection from its siblings (drag one frame to
            // load the whole detected sequence).
            1 => Self::collect_sequence_bytes(&paths[0]).unwrap_or_default(),
            // Multiple explicit files: load exactly those (sorted by name),
            // instead of dir-scanning from the first and ignoring the rest.
            _ => {
                paths.sort();
                let mut out = Vec::with_capacity(paths.len());
                for p in &paths {
                    if p.is_dir() {
                        continue;
                    }
                    if let Some(frame) = Self::load_single(p) {
                        out.extend(frame);
                    }
                }
                out
            }
        }
    }
}

type MotionFrameApp = motionframe_ui::MotionFrameApp<DesktopPlatform>;

/// Launch the desktop GUI.
pub fn run_gui() -> Result<(), eframe::Error> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "MotionFrame",
        opts,
        Box::new(|cc| {
            let platform = DesktopPlatform::new(cc);
            Ok(Box::new(MotionFrameApp::new(platform, cc)))
        }),
    )
}
