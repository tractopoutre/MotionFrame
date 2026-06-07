//! Worker entry: handles incoming messages from main thread.

use std::cell::Cell;

use motionframe_engine::pipeline::run::run_pipeline;
use motionframe_engine::pipeline::Progress;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

use motionframe_engine::io::{decode_image_from_bytes, slice_atlas, InMemoryFrames};

use crate::protocol::{CmdToWorker, DoneMeta, EncodedFrameMsg, MsgFromWorker, PackModeMsg};
use crate::streaming::{EncodedFrame, StreamingFrames};

thread_local! {
    /// Worker-local cancel flag. Set by Cancel message; checked by pipeline.
    static CANCEL_FLAG: Cell<bool> = const { Cell::new(false) };
    /// Worker-local "currently-generating" flag for ignore-stale-message logic.
    static GENERATING: Cell<bool> = const { Cell::new(false) };
}

fn worker_scope() -> web_sys::DedicatedWorkerGlobalScope {
    js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>()
}

fn post(msg: &MsgFromWorker) {
    match serde_wasm_bindgen::to_value(msg) {
        Ok(js) => {
            let _ = worker_scope().post_message(&js);
        }
        Err(e) => {
            log::error!("serialize msg from worker: {e:?}");
        }
    }
}

/// Entry: registered as the worker's onmessage handler.
#[wasm_bindgen]
pub fn handle_message(msg: JsValue) {
    let cmd: CmdToWorker = match serde_wasm_bindgen::from_value(msg) {
        Ok(c) => c,
        Err(e) => {
            post(&MsgFromWorker::Error(format!("bad command: {e}")));
            return;
        }
    };
    match cmd {
        CmdToWorker::Cancel => {
            CANCEL_FLAG.with(|c| c.set(true));
        }
        CmdToWorker::Generate { frames, options } => {
            if GENERATING.with(Cell::get) {
                post(&MsgFromWorker::Error("already generating".to_string()));
                return;
            }
            CANCEL_FLAG.with(|c| c.set(false));
            GENERATING.with(|g| g.set(true));
            spawn_local(async move {
                let result = run_generate(frames, options).await;
                GENERATING.with(|g| g.set(false));
                match result {
                    Ok(()) => {} // run_generate posts Done itself
                    Err(GenerateErr::Cancelled) => post(&MsgFromWorker::Cancelled),
                    Err(GenerateErr::Other(e)) => post(&MsgFromWorker::Error(e)),
                }
            });
        }
    }
}

enum GenerateErr {
    Cancelled,
    Other(String),
}

async fn run_generate(
    frames: Vec<EncodedFrameMsg>,
    options: motionframe_engine::pipeline::GenerateOptions,
) -> Result<(), GenerateErr> {
    let encoded: Vec<EncodedFrame> = frames
        .into_iter()
        .map(|f| EncodedFrame {
            name: f.name,
            bytes: f.bytes,
        })
        .collect();

    // Atlas mode: single sprite-sheet image → slice into tiles → InMemoryFrames.
    // Sequence mode: keep the existing streaming (lazy LRU) path.
    if let Some((cols, rows)) = options.input_atlas_dims {
        if encoded.len() != 1 {
            return Err(GenerateErr::Other(
                motionframe_engine::pipeline::PipelineError::AtlasFrameCount(encoded.len())
                    .to_string(),
            ));
        }
        let ef = &encoded[0];
        let src = decode_image_from_bytes(&ef.name, &ef.bytes)
            .map_err(|e| GenerateErr::Other(format!("atlas decode: {e}")))?;
        let tiles = slice_atlas(&src, cols, rows).map_err(|e| {
            GenerateErr::Other(motionframe_engine::pipeline::PipelineError::Atlas(e).to_string())
        })?;
        let source = InMemoryFrames::new(tiles)
            .map_err(|e| GenerateErr::Other(format!("frame source: {e}")))?;

        let progress_fn = |p: Progress| {
            post(&MsgFromWorker::Progress(p));
        };
        let cancel_fn = || CANCEL_FLAG.with(Cell::get);

        let result = run_pipeline(&source, &options, &progress_fn, &cancel_fn);

        return match result {
            Ok(encode_result) => {
                post_done(&encode_result);
                Ok(())
            }
            Err(motionframe_engine::pipeline::PipelineError::Cancelled) => {
                Err(GenerateErr::Cancelled)
            }
            Err(e) => Err(GenerateErr::Other(format!("{e}"))),
        };
    }

    let source = StreamingFrames::new(encoded)
        .map_err(|e| GenerateErr::Other(format!("frame source: {e}")))?;

    let progress_fn = |p: Progress| {
        post(&MsgFromWorker::Progress(p));
    };
    let cancel_fn = || CANCEL_FLAG.with(Cell::get);

    let result = run_pipeline(&source, &options, &progress_fn, &cancel_fn);

    match result {
        Ok(encode_result) => {
            post_done(&encode_result);
            Ok(())
        }
        Err(motionframe_engine::pipeline::PipelineError::Cancelled) => Err(GenerateErr::Cancelled),
        Err(e) => Err(GenerateErr::Other(format!("{e}"))),
    }
}

/// Build and post the Done message manually so that the heavy buffers
/// (atlases, flows, gray frames) cross the worker→main boundary as
/// transferable `ArrayBuffer`s instead of being walked by
/// `serde_wasm_bindgen::from_value` on the main thread.
fn post_done(encode_result: &motionframe_engine::pipeline::run::EncodeResult) {
    let pack_mode = match encode_result.pack_mode {
        motionframe_engine::pipeline::run::PackMode::Staggered => PackModeMsg::Staggered,
        motionframe_engine::pipeline::run::PackMode::Normal => PackModeMsg::Normal,
    };

    let meta = DoneMeta {
        color_atlas_w: encode_result.color_atlas.width,
        color_atlas_h: encode_result.color_atlas.height,
        motion_atlas_w: encode_result.motion_atlas.width,
        motion_atlas_h: encode_result.motion_atlas.height,
        strength: encode_result.strength,
        total_frames: encode_result.total_frames,
        atlas_width: encode_result.atlas_width,
        atlas_height: encode_result.atlas_height,
        pack_mode,
        is_loop: encode_result.is_loop,
        premultiplied_alpha: encode_result.premultiplied_alpha,
    };
    let meta_js = match serde_wasm_bindgen::to_value(&meta) {
        Ok(v) => v,
        Err(e) => {
            log::error!("serialize done meta: {e:?}");
            return;
        }
    };

    let color = u8_to_js(&encode_result.color_atlas.data);
    let motion = u8_to_js(&encode_result.motion_atlas.data);

    // Pack flows: dims as [w0,h0,w1,h1,...], data as concatenated f32 pairs.
    let flow_count = encode_result.flows.len();
    let flow_dims = js_sys::Uint32Array::new_with_length(len_u32(flow_count * 2));
    let mut total_flow_floats: usize = 0;
    {
        let mut dims_buf: Vec<u32> = Vec::with_capacity(flow_count * 2);
        for f in &encode_result.flows {
            dims_buf.push(f.width);
            dims_buf.push(f.height);
            total_flow_floats += f.data.len() * 2;
        }
        flow_dims.copy_from(&dims_buf);
    }
    let flow_data = js_sys::Float32Array::new_with_length(len_u32(total_flow_floats));
    {
        let mut offset: u32 = 0;
        for f in &encode_result.flows {
            // `as_flattened()` reinterprets `&[[f32; 2]]` as `&[f32]` with no copy.
            let flat: &[f32] = f.data.as_flattened();
            let len = len_u32(flat.len());
            if len > 0 {
                flow_data.subarray(offset, offset + len).copy_from(flat);
                offset += len;
            }
        }
    }

    let payload = js_sys::Object::new();
    set_field(&payload, "type", &JsValue::from_str("Done"));
    set_field(&payload, "meta", &meta_js);
    set_field(&payload, "color", &color);
    set_field(&payload, "motion", &motion);
    set_field(&payload, "flow_dims", &flow_dims);
    set_field(&payload, "flow_data", &flow_data);

    // Transfer list: hand off ownership of all underlying ArrayBuffers so the
    // main thread receives them without a structured-clone deep copy.
    let transfer = js_sys::Array::new();
    transfer.push(&color.buffer());
    transfer.push(&motion.buffer());
    transfer.push(&flow_dims.buffer());
    transfer.push(&flow_data.buffer());

    if let Err(e) = worker_scope().post_message_with_transfer(&payload, &transfer) {
        log::error!("post_message_with_transfer Done: {e:?}");
    }
}

/// Allocate a fresh JS `Uint8Array` (transferable) and copy `src` into it.
fn u8_to_js(src: &[u8]) -> js_sys::Uint8Array {
    let arr = js_sys::Uint8Array::new_with_length(len_u32(src.len()));
    arr.copy_from(src);
    arr
}

/// Convert a `usize` length to `u32` for typed-array sizing. On `wasm32`,
/// `usize` IS `u32`, so this never fails — but `.expect` makes the intent
/// explicit and would panic loudly if the target ever moves to wasm64,
/// instead of silently producing zero-length buffers.
fn len_u32(n: usize) -> u32 {
    u32::try_from(n).expect("length fits u32 on wasm32")
}

/// Set a string-keyed field on a JS object, ignoring the `Result` since
/// `Reflect::set` on a fresh `Object` cannot fail.
fn set_field(obj: &js_sys::Object, key: &str, val: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), val);
}
