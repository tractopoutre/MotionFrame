// Integration tests: CLI `convert` subcommand.

use std::process::Command;

#[test]
fn convert_help_is_available() {
    let output = Command::new(env!("CARGO_BIN_EXE_motionframe"))
        .arg("convert")
        .arg("--help")
        .output()
        .expect("run motionframe convert --help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("--input"));
    assert!(stdout.contains("--output"));
}

#[test]
fn headless_flag_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_motionframe"))
        .arg("--headless")
        .output()
        .expect("run motionframe --headless");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).expect("stderr is utf8");
    assert!(stderr.contains("unexpected argument '--headless'"));
}

#[test]
fn convert_explosion00_produces_outputs() {
    let out_prefix = std::env::temp_dir().join("motionframe_convert_test");
    let color = out_prefix.with_file_name("motionframe_convert_test_color_atlas.tga");
    let motion = out_prefix.with_file_name("motionframe_convert_test_motion_atlas.tga");
    let meta = out_prefix.with_file_name("motionframe_convert_test_meta.json");
    let _ = std::fs::remove_file(&color);
    let _ = std::fs::remove_file(&motion);
    let _ = std::fs::remove_file(&meta);

    let input_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("explosion00");

    let status = Command::new(env!("CARGO_BIN_EXE_motionframe"))
        .arg("convert")
        .arg("--input")
        .arg(&input_dir)
        .arg("--output")
        .arg(&out_prefix)
        .arg("--overwrite")
        .status()
        .expect("run motionframe convert");

    assert!(status.success(), "convert exited with failure");
    assert!(color.exists(), "color atlas not found: {}", color.display());
    assert!(
        motion.exists(),
        "motion atlas not found: {}",
        motion.display()
    );
    assert!(meta.exists(), "metadata not found: {}", meta.display());

    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta).expect("read metadata"))
            .expect("metadata json");
    assert!(parsed["strength"].is_number());
    assert!(parsed["total_frames"].is_number());

    let _ = std::fs::remove_file(&color);
    let _ = std::fs::remove_file(&motion);
    let _ = std::fs::remove_file(&meta);
}

#[test]
fn convert_refuses_existing_outputs_without_overwrite() {
    let out_prefix = std::env::temp_dir().join("motionframe_convert_existing");
    let color = out_prefix.with_file_name("motionframe_convert_existing_color_atlas.tga");
    std::fs::write(&color, b"existing").expect("write existing output");

    let input_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("explosion00");

    let output = Command::new(env!("CARGO_BIN_EXE_motionframe"))
        .arg("convert")
        .arg("--input")
        .arg(&input_dir)
        .arg("--output")
        .arg(&out_prefix)
        .output()
        .expect("run motionframe convert");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).expect("stderr is utf8");
    assert!(stderr.contains("already exists"));

    let _ = std::fs::remove_file(color);
}

#[test]
fn convert_config_with_cli_override_changes_metadata() {
    let dir = std::env::temp_dir();
    let config_path = dir.join("motionframe_convert_config.toml");
    let out_prefix = dir.join("motionframe_convert_config_out");
    let color = out_prefix.with_file_name("motionframe_convert_config_out_color_atlas.tga");
    let motion = out_prefix.with_file_name("motionframe_convert_config_out_motion_atlas.tga");
    let meta = out_prefix.with_file_name("motionframe_convert_config_out_meta.json");
    let _ = std::fs::remove_file(&color);
    let _ = std::fs::remove_file(&motion);
    let _ = std::fs::remove_file(&meta);

    let input_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("explosion00");

    std::fs::write(
        &config_path,
        format!(
            r#"
input = "{}"
output = "unused/from-config"
output_count = 4
layout = "auto"
progress = "none"
"#,
            input_dir.display()
        ),
    )
    .expect("write config");

    let status = Command::new(env!("CARGO_BIN_EXE_motionframe"))
        .arg("convert")
        .arg("--config")
        .arg(&config_path)
        .arg("--output")
        .arg(&out_prefix)
        .arg("--output-count")
        .arg("2")
        .arg("--overwrite")
        .status()
        .expect("run motionframe convert");

    assert!(status.success(), "convert exited with failure");
    let parsed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta).expect("read metadata"))
            .expect("metadata json");
    assert_eq!(parsed["total_frames"], 2);

    let _ = std::fs::remove_file(config_path);
    let _ = std::fs::remove_file(&color);
    let _ = std::fs::remove_file(&motion);
    let _ = std::fs::remove_file(&meta);
}
