// In release builds, use the Windows GUI subsystem so double-clicking the
// executable does not spawn a stray console window behind the egui app. Debug
// builds keep the console subsystem so developer output is always visible.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;
use env_logger::Env;
use motionframe_desktop::cli::{run, Cli};

/// Reattach stdio to the parent console when launched from a terminal.
///
/// Under the `windows` subsystem the process starts with no console, so CLI
/// output (`convert`, `--help`, errors) would be lost. `AttachConsole` rebinds
/// stdout/stderr to the launching terminal; it is a no-op when double-clicked
/// from Explorer (no parent console), preserving the clean GUI launch.
#[cfg(windows)]
#[allow(
    unsafe_code,
    reason = "single Win32 FFI call to reattach the parent console"
)]
fn attach_parent_console() {
    use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    // SAFETY: FFI call with no preconditions. A failed attach (GUI launch)
    // returns 0 and is intentionally ignored.
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

fn main() {
    #[cfg(windows)]
    attach_parent_console();

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(e.exit_code());
    }
}
