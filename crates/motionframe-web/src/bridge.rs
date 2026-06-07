//! Invisible-HTML bridges for browser APIs egui doesn't expose:
//! - folder picker via hidden `<input webkitdirectory>`
//! - download trigger via hidden `<a download>`

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use motionframe_ui::platform::EncodedFrame;

/// True if the filename ends in `.png` or `.tga`, case-insensitive.
pub(crate) fn is_image_filename(name: &str) -> bool {
    std::path::Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("png") || ext.eq_ignore_ascii_case("tga"))
}

/// Open a folder picker synchronously (preserving the user-gesture window).
/// When the user makes a selection, read the file bytes and write into
/// `pending`. Then call `ctx.request_repaint()` so the egui app picks them up.
pub fn pick_directory_into(pending: &Rc<RefCell<Option<Vec<EncodedFrame>>>>, ctx: &egui::Context) {
    let document = web_sys::window().unwrap().document().unwrap();
    let input = document
        .create_element("input")
        .unwrap()
        .dyn_into::<web_sys::HtmlInputElement>()
        .unwrap();
    input.set_type("file");
    input.set_multiple(true);
    input.set_accept(".png,.tga");
    let _ = input.set_attribute("webkitdirectory", "");
    input.style().set_property("display", "none").ok();
    document.body().unwrap().append_child(&input).ok();

    // The listener captures `pending` + `ctx` and reads files asynchronously
    // when the user chooses. We forget the closure so it stays alive for the
    // lifetime of the input element.
    let input_clone = input.clone();
    let pending_clone = Rc::clone(pending);
    let ctx_clone = ctx.clone();
    let on_change = Closure::wrap(Box::new(move |_: web_sys::Event| {
        let pending_inner = Rc::clone(&pending_clone);
        let ctx_inner = ctx_clone.clone();
        let files = input_clone.files();
        let count = files.as_ref().map_or(0, web_sys::FileList::length);
        log::info!("folder picker change: {count} entries selected");
        let input_for_cleanup = input_clone.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let mut out: Vec<EncodedFrame> = Vec::with_capacity(count as usize);
            if let Some(file_list) = files {
                for i in 0..file_list.length() {
                    let Some(file) = file_list.get(i) else {
                        continue;
                    };
                    let name = file.name();
                    if !is_image_filename(&name) {
                        continue;
                    }
                    match read_file_bytes(&file).await {
                        Ok(bytes) => out.push(EncodedFrame { name, bytes }),
                        Err(e) => log::warn!("read {name} failed: {e:?}"),
                    }
                }
            }
            // Browser FileList order from `<input webkitdirectory>` is not
            // sorted — Chrome and Safari surface OS directory-entry order,
            // which is effectively undefined. Mirror desktop's lexicographic
            // sort (engine `sequence::collect_sequence_files` does the same)
            // so zero-padded sequences play in numeric order.
            out.sort_by(|a, b| a.name.cmp(&b.name));
            log::info!("folder picker delivered {} usable frames", out.len());
            *pending_inner.borrow_mut() = Some(out);
            ctx_inner.request_repaint();
            // Remove the temporary input element from DOM.
            if let Some(parent) = input_for_cleanup.parent_node() {
                let node: &web_sys::Node = input_for_cleanup.as_ref();
                let _ = parent.remove_child(node);
            }
        });
    }) as Box<dyn FnMut(web_sys::Event)>);
    input
        .add_event_listener_with_callback("change", on_change.as_ref().unchecked_ref())
        .unwrap();
    on_change.forget();

    // Click while still inside the user gesture.
    input.click();
}

/// Install document-level drag-and-drop handlers that walk dropped folders
/// (which `dataTransfer.files` doesn't expose) and route the resulting frames
/// through the same `pending` channel as the Browse picker.
///
/// Why bypass eframe's drop handling: eframe reads `dataTransfer.files`, and
/// browsers do not expand directories in that list — folder drops surface as
/// zero entries. The standards-compliant path uses `webkitGetAsEntry()` on
/// each `DataTransferItem`, then walks `FileSystemDirectoryEntry` trees via
/// `createReader().readEntries()` (which returns chunks; loop until empty).
///
/// Listeners are attached at the document level in the capture phase. On
/// `dragover` we `preventDefault` (required for `drop` to fire) but let it
/// propagate so eframe can still update its hover state. On `drop` we
/// `preventDefault` and `stopImmediatePropagation` so the event doesn't reach
/// eframe; eframe would just see an empty file list anyway, but skipping the
/// path avoids any spurious "drop accepted" feedback.
pub fn install_drop_handler(pending: &Rc<RefCell<Option<Vec<EncodedFrame>>>>, ctx: &egui::Context) {
    let document = web_sys::window().unwrap().document().unwrap();

    let on_dragover = Closure::wrap(Box::new(|ev: web_sys::DragEvent| {
        ev.prevent_default();
    }) as Box<dyn FnMut(web_sys::DragEvent)>);
    document
        .add_event_listener_with_callback_and_bool(
            "dragover",
            on_dragover.as_ref().unchecked_ref(),
            true,
        )
        .unwrap();
    on_dragover.forget();

    let pending_clone = Rc::clone(pending);
    let ctx_clone = ctx.clone();
    let on_drop = Closure::wrap(Box::new(move |ev: web_sys::DragEvent| {
        ev.prevent_default();
        ev.stop_immediate_propagation();
        let Some(dt) = ev.data_transfer() else {
            return;
        };
        // Capture FileSystemEntries synchronously: DataTransfer is invalidated
        // once the event handler returns, but FileSystemEntry handles persist.
        let items = dt.items();
        let mut roots: Vec<web_sys::FileSystemEntry> = Vec::new();
        for i in 0..items.length() {
            let Some(item) = items.get(i) else { continue };
            if let Ok(Some(entry)) = item.webkit_get_as_entry() {
                roots.push(entry);
            }
        }
        if roots.is_empty() {
            return;
        }
        let pending_inner = Rc::clone(&pending_clone);
        let ctx_inner = ctx_clone.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let mut out = collect_entries(roots).await;
            // Mirror desktop's lex sort for zero-padded sequence ordering.
            out.sort_by(|a, b| a.name.cmp(&b.name));
            log::info!("drop handler delivered {} usable frames", out.len());
            *pending_inner.borrow_mut() = Some(out);
            ctx_inner.request_repaint();
        });
    }) as Box<dyn FnMut(web_sys::DragEvent)>);
    document
        .add_event_listener_with_callback_and_bool("drop", on_drop.as_ref().unchecked_ref(), true)
        .unwrap();
    on_drop.forget();
}

/// Stack-based DFS over a forest of `FileSystemEntry`. Reads files lazily
/// (one bytes-blob at a time) so a giant directory doesn't pin every File
/// object in memory at once.
async fn collect_entries(roots: Vec<web_sys::FileSystemEntry>) -> Vec<EncodedFrame> {
    let mut stack = roots;
    let mut out: Vec<EncodedFrame> = Vec::new();
    while let Some(entry) = stack.pop() {
        if entry.is_file() {
            let fe: web_sys::FileSystemFileEntry = entry.unchecked_into();
            match entry_to_file(&fe).await {
                Ok(file) => {
                    let name = file.name();
                    if !is_image_filename(&name) {
                        continue;
                    }
                    match read_file_bytes(&file).await {
                        Ok(bytes) => out.push(EncodedFrame { name, bytes }),
                        Err(e) => log::warn!("read {name}: {e:?}"),
                    }
                }
                Err(e) => log::warn!("entry_to_file: {e:?}"),
            }
        } else if entry.is_directory() {
            let de: web_sys::FileSystemDirectoryEntry = entry.unchecked_into();
            let reader = de.create_reader();
            // readEntries returns chunks (often 100 at a time); spec requires
            // calling repeatedly until an empty array signals end-of-directory.
            loop {
                match read_entries_chunk(&reader).await {
                    Ok(batch) if batch.is_empty() => break,
                    Ok(batch) => stack.extend(batch),
                    Err(e) => {
                        log::warn!("readEntries: {e:?}");
                        break;
                    }
                }
            }
        }
    }
    out
}

/// Promisify `FileSystemFileEntry.file(success, error)`.
async fn entry_to_file(entry: &web_sys::FileSystemFileEntry) -> Result<web_sys::File, JsValue> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        entry.file_with_callback_and_callback(&resolve, &reject);
    });
    let val = wasm_bindgen_futures::JsFuture::from(promise).await?;
    val.dyn_into::<web_sys::File>()
}

/// Promisify a single `readEntries` call. Returns one chunk; caller loops.
async fn read_entries_chunk(
    reader: &web_sys::FileSystemDirectoryReader,
) -> Result<Vec<web_sys::FileSystemEntry>, JsValue> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        // `read_entries_with_callback_and_callback` returns Result; surface
        // a synchronous failure by rejecting the promise so callers see the
        // same error path as a later async rejection.
        if let Err(e) = reader.read_entries_with_callback_and_callback(&resolve, &reject) {
            let _ = reject.call1(&JsValue::NULL, &e);
        }
    });
    let val = wasm_bindgen_futures::JsFuture::from(promise).await?;
    let array = js_sys::Array::from(&val);
    Ok(array
        .iter()
        .map(wasm_bindgen::JsCast::unchecked_into::<web_sys::FileSystemEntry>)
        .collect())
}

/// Read a `File` blob into a fresh `Vec<u8>`.
///
/// # Errors
/// Returns the underlying `JsValue` if reading the blob's `ArrayBuffer` rejects.
pub async fn read_file_bytes(file: &web_sys::File) -> Result<Vec<u8>, JsValue> {
    let buf_promise = file.array_buffer();
    let buf = wasm_bindgen_futures::JsFuture::from(buf_promise).await?;
    let array = js_sys::Uint8Array::new(&buf);
    let mut out = vec![0u8; array.length() as usize];
    array.copy_to(&mut out);
    Ok(out)
}

/// Trigger a browser download of `bytes` named `filename` with the given MIME.
pub fn trigger_download(filename: &str, bytes: &[u8], mime: &str) {
    let document = web_sys::window().unwrap().document().unwrap();
    let len = u32::try_from(bytes.len()).expect("download bytes len fits u32");
    let array = js_sys::Uint8Array::new_with_length(len);
    array.copy_from(bytes);
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&array.buffer());

    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&blob_parts, &bag).unwrap();

    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();

    let a = document
        .create_element("a")
        .unwrap()
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .unwrap();
    a.set_href(&url);
    a.set_download(filename);
    a.style().set_property("display", "none").ok();
    document.body().unwrap().append_child(&a).ok();
    a.click();
    document.body().unwrap().remove_child(&a).ok();

    // Revoke after a short delay (let download start).
    let url_to_revoke = url.clone();
    let cb = Closure::once(Box::new(move || {
        let _ = web_sys::Url::revoke_object_url(&url_to_revoke);
    }) as Box<dyn FnOnce()>);
    web_sys::window()
        .unwrap()
        .set_timeout_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), 5000)
        .unwrap();
    cb.forget();
}
