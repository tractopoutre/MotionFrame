//! Main-side wrapper around the Web Worker.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use motionframe_engine::pipeline::run::{EncodeResult, PackMode};
use motionframe_engine::pipeline::{Flow, GenerateOptions, ImageRgba8, Progress};
use serde::{Deserialize, Serialize};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[derive(Debug, Serialize, Deserialize)]
/// Encoded source frame sent to the wasm worker.
pub struct EncodedFrameMsg {
    pub name: String,
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
/// Commands posted from the main thread to the wasm worker.
pub enum CmdToWorker {
    Generate {
        frames: Vec<EncodedFrameMsg>,
        options: GenerateOptions,
    },
    Cancel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// Pack mode in a serde-friendly form.
pub enum PackModeMsg {
    Staggered,
    Normal,
}

/// Scalar half of the Done payload. Heavy buffers come in alongside via
/// transferable typed arrays — see `parse_done_message`.
#[derive(Debug, Deserialize)]
pub struct DoneMeta {
    pub color_atlas_w: u32,
    pub color_atlas_h: u32,
    pub motion_atlas_w: u32,
    pub motion_atlas_h: u32,
    pub strength: f64,
    pub total_frames: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub columns: u32,
    pub rows: u32,
    pub pack_mode: PackModeMsg,
    pub is_loop: bool,
    pub premultiplied_alpha: bool,
}

/// The non-Done variants. `Done` is parsed via `parse_done_message` so the
/// main thread never serde-walks tens of MB of pixel data.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MsgFromWorker {
    Progress(Progress),
    Error(String),
    Cancelled,
}

/// Events drained from the wasm worker inbox by the app shell.
pub enum WorkerEvent {
    Progress(Progress),
    Done(EncodeResult),
    Error(String),
    Cancelled,
}

/// Main-thread owner for the browser `Worker` and its event inbox.
pub struct WorkerClient {
    worker: web_sys::Worker,
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_error: Closure<dyn FnMut(web_sys::ErrorEvent)>,
    inbox: Rc<RefCell<VecDeque<WorkerEvent>>>,
}

impl WorkerClient {
    pub fn spawn() -> Result<Self, JsValue> {
        let opts = web_sys::WorkerOptions::new();
        opts.set_type(web_sys::WorkerType::Module);
        // WORKER_DIR is the content-hashed worker directory injected by build.rs
        // (e.g. "worker-1a2b3c4d"), so a worker change is a new URL the browser
        // hasn't cached — no stale-worker / new-app skew.
        let worker = web_sys::Worker::new_with_options(
            concat!("./static/", env!("WORKER_DIR"), "/worker.js"),
            &opts,
        )?;

        let inbox: Rc<RefCell<VecDeque<WorkerEvent>>> = Rc::new(RefCell::new(VecDeque::new()));
        let inbox_clone = Rc::clone(&inbox);
        let on_message = Closure::wrap(Box::new(move |ev: web_sys::MessageEvent| {
            let data = ev.data();
            // Sniff the discriminator first so the heavy Done payload bypasses
            // serde-wasm-bindgen entirely (it walks each field synchronously
            // and froze the UI for ~1 s on tens of MB).
            let type_field = js_sys::Reflect::get(&data, &JsValue::from_str("type"))
                .ok()
                .and_then(|v| v.as_string());
            if type_field.as_deref() == Some("Done") {
                match parse_done_message(&data) {
                    Ok(result) => {
                        inbox_clone
                            .borrow_mut()
                            .push_back(WorkerEvent::Done(result));
                    }
                    Err(e) => {
                        inbox_clone
                            .borrow_mut()
                            .push_back(WorkerEvent::Error(format!("done parse: {e}")));
                    }
                }
                return;
            }
            match serde_wasm_bindgen::from_value::<MsgFromWorker>(data) {
                Ok(MsgFromWorker::Progress(p)) => {
                    inbox_clone.borrow_mut().push_back(WorkerEvent::Progress(p));
                }
                Ok(MsgFromWorker::Error(e)) => {
                    inbox_clone.borrow_mut().push_back(WorkerEvent::Error(e));
                }
                Ok(MsgFromWorker::Cancelled) => {
                    inbox_clone.borrow_mut().push_back(WorkerEvent::Cancelled);
                }
                Err(e) => {
                    inbox_clone
                        .borrow_mut()
                        .push_back(WorkerEvent::Error(format!("worker msg parse: {e}")));
                }
            }
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // A worker-level error (module load failure, or a wasm trap not caught
        // by worker.js's try/catch) would otherwise leave the app waiting for a
        // Done that never arrives. Surface it as a terminal Error event so the
        // app clears its generating state and shows the message.
        let inbox_err = Rc::clone(&inbox);
        let on_error = Closure::wrap(Box::new(move |ev: web_sys::ErrorEvent| {
            let msg = ev.message();
            let msg = if msg.is_empty() {
                "worker crashed".to_string()
            } else {
                format!("worker crashed: {msg}")
            };
            inbox_err.borrow_mut().push_back(WorkerEvent::Error(msg));
        }) as Box<dyn FnMut(web_sys::ErrorEvent)>);
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        Ok(Self {
            worker,
            _on_message: on_message,
            _on_error: on_error,
            inbox,
        })
    }

    pub fn send_generate(
        &self,
        frames: Vec<(String, Vec<u8>)>,
        options: GenerateOptions,
    ) -> Result<(), JsValue> {
        let msg = CmdToWorker::Generate {
            frames: frames
                .into_iter()
                .map(|(name, bytes)| EncodedFrameMsg { name, bytes })
                .collect(),
            options,
        };
        let js = serde_wasm_bindgen::to_value(&msg)
            .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))?;
        self.worker.post_message(&js)
    }

    pub fn send_cancel(&self) -> Result<(), JsValue> {
        let msg = CmdToWorker::Cancel;
        let js = serde_wasm_bindgen::to_value(&msg)
            .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))?;
        self.worker.post_message(&js)
    }

    pub fn drain_events(&self) -> Vec<WorkerEvent> {
        let drained: VecDeque<_> = std::mem::take(&mut *self.inbox.borrow_mut());
        drained.into_iter().collect()
    }

    /// Queue an event at the back.
    pub fn push_event(&self, ev: WorkerEvent) {
        self.inbox.borrow_mut().push_back(ev);
    }

    /// Push an event back to the front of the queue.
    pub fn push_front(&self, ev: WorkerEvent) {
        self.inbox.borrow_mut().push_front(ev);
    }
}

/// Copy an RGBA typed array into a `Vec`, verifying its length matches
/// `width * height * 4` first.
fn validated_rgba(
    arr: &js_sys::Uint8Array,
    width: u32,
    height: u32,
    label: &str,
) -> Result<Vec<u8>, String> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| format!("{label} atlas dims {width}x{height} overflow"))?;
    let actual = arr.length() as usize;
    if actual != expected {
        return Err(format!(
            "{label} atlas buffer len {actual} != expected {expected} for {width}x{height}"
        ));
    }
    Ok(arr.to_vec())
}

fn get_field(obj: &JsValue, key: &str) -> Result<JsValue, String> {
    js_sys::Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|e| format!("missing field `{key}`: {e:?}"))
}

/// Reconstruct an `EncodeResult` from the manually-built Done JS object.
/// Each typed-array slice is copied once into a fresh `Vec` — no slab
/// intermediate, no serde walk — so a Done with tens of MB completes well
/// under one frame.
fn parse_done_message(data: &JsValue) -> Result<EncodeResult, String> {
    let meta_js = get_field(data, "meta")?;
    let meta: DoneMeta =
        serde_wasm_bindgen::from_value(meta_js).map_err(|e| format!("meta deserialize: {e}"))?;

    let color: js_sys::Uint8Array = get_field(data, "color")?
        .dyn_into()
        .map_err(|_| "color not Uint8Array".to_string())?;
    let motion: js_sys::Uint8Array = get_field(data, "motion")?
        .dyn_into()
        .map_err(|_| "motion not Uint8Array".to_string())?;
    let flow_dims: js_sys::Uint32Array = get_field(data, "flow_dims")?
        .dyn_into()
        .map_err(|_| "flow_dims not Uint32Array".to_string())?;
    let flow_data: js_sys::Float32Array = get_field(data, "flow_data")?
        .dyn_into()
        .map_err(|_| "flow_data not Float32Array".to_string())?;
    // Validate buffer lengths against declared dims: downstream consumers index
    // these by width*height*4 and would OOB-panic on a short/detached buffer.
    let color_data = validated_rgba(&color, meta.color_atlas_w, meta.color_atlas_h, "color")?;
    let motion_data = validated_rgba(&motion, meta.motion_atlas_w, meta.motion_atlas_h, "motion")?;
    let color_atlas = ImageRgba8 {
        width: meta.color_atlas_w,
        height: meta.color_atlas_h,
        data: color_data,
    };
    let motion_atlas = ImageRgba8 {
        width: meta.motion_atlas_w,
        height: meta.motion_atlas_h,
        data: motion_data,
    };

    let flows = parse_flows(&flow_dims, &flow_data)?;

    let pack_mode = match meta.pack_mode {
        PackModeMsg::Staggered => PackMode::Staggered,
        PackModeMsg::Normal => PackMode::Normal,
    };

    Ok(EncodeResult {
        color_atlas,
        motion_atlas,
        strength: meta.strength,
        total_frames: meta.total_frames,
        atlas_width: meta.atlas_width,
        atlas_height: meta.atlas_height,
        columns: meta.columns,
        rows: meta.rows,
        pack_mode,
        is_loop: meta.is_loop,
        premultiplied_alpha: meta.premultiplied_alpha,
        flows,
    })
}

/// Split the concatenated `flow_data` into per-flow `Vec<[f32; 2]>` using
/// `flow_dims` as `[w0, h0, w1, h1, ...]`. Validates u32 overflow and that
/// `flow_data.length()` exactly matches the sum of `w*h*2`.
fn parse_flows(
    flow_dims: &js_sys::Uint32Array,
    flow_data: &js_sys::Float32Array,
) -> Result<Vec<Flow>, String> {
    let dims = flow_dims.to_vec();
    if !dims.len().is_multiple_of(2) {
        return Err(format!("flow_dims len {} not even", dims.len()));
    }
    let flow_count = dims.len() / 2;
    let total_flow_floats = flow_data.length();
    let mut flows: Vec<Flow> = Vec::with_capacity(flow_count);
    let mut foffset: u32 = 0;
    for i in 0..flow_count {
        let w = dims[i * 2];
        let h = dims[i * 2 + 1];
        // u64 mul guards against w*h overflowing usize on wasm32 (where
        // usize == u32). A flow whose total f32 count doesn't fit in u32
        // can't have been produced by the worker (Float32Array length is u32),
        // so this is purely a defense against a malformed message.
        let float_count: u32 = u64::from(w)
            .checked_mul(u64::from(h))
            .and_then(|p| p.checked_mul(2))
            .and_then(|f| u32::try_from(f).ok())
            .ok_or_else(|| format!("flow {i} dims {w}x{h} overflow u32"))?;
        let end = foffset
            .checked_add(float_count)
            .ok_or_else(|| format!("flow {i} cursor overflow"))?;
        if end > total_flow_floats {
            return Err(format!("flow_data shorter than dims claim at flow {i}"));
        }
        // One memcpy from JS into a fresh `Vec<f32>`, then `as_chunks::<2>`
        // reinterprets as `&[[f32; 2]]` and `to_vec` copies into the final
        // `Vec<[f32; 2]>`.
        let floats = flow_data.subarray(foffset, end).to_vec();
        let (pairs, rest) = floats.as_chunks::<2>();
        debug_assert!(rest.is_empty(), "float_count is even by construction");
        flows.push(Flow {
            width: w,
            height: h,
            data: pairs.to_vec(),
        });
        foffset = end;
    }
    if foffset != total_flow_floats {
        return Err(format!(
            "flow_data has {} extra trailing floats",
            total_flow_floats - foffset
        ));
    }
    Ok(flows)
}
