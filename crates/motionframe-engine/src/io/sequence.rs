//! Two-stage sequence discovery: detect numbering pattern, then strict-load
//! all matching files in the directory. Gaps in numbering are silently
//! ignored.

use std::path::{Path, PathBuf};

/// Accepted image file extensions for sequence scanning (case-insensitive match).
pub const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tga"];

/// Detect a frame numbering pattern from a single filename.
///
/// Returns `(prefix, num_digits, extension)` if the filename matches
/// `^(.*?)(\d+)\.(\w+)$`, i.e. has trailing digits before the extension.
pub fn detect_pattern(filename: &str) -> Option<(String, u32, String)> {
    let dot = filename.rfind('.')?;
    let (stem, ext_with_dot) = filename.split_at(dot);
    let ext = &ext_with_dot[1..];
    if ext.is_empty() || !ext.chars().all(char::is_alphanumeric) {
        return None;
    }

    let digits_end = stem.len();
    let digits_start = stem
        .bytes()
        .rposition(|b| !b.is_ascii_digit())
        .map_or(0, |i| i + 1);
    if digits_start == digits_end {
        return None;
    }

    let prefix = &stem[..digits_start];
    let num_digits = (digits_end - digits_start) as u32;
    Some((prefix.to_string(), num_digits, ext.to_string()))
}

/// Check whether `filename` matches the strict pattern `{prefix}\d{num_digits}.{ext}`.
fn matches_pattern(filename: &str, prefix: &str, num_digits: u32, ext: &str) -> bool {
    let Some(rest) = filename.strip_prefix(prefix) else {
        return false;
    };
    // Strip ".ext" suffix without allocating (avoids per-call format! allocation).
    let Some(without_dot) = rest.strip_suffix(ext) else {
        return false;
    };
    let Some(num_part) = without_dot.strip_suffix('.') else {
        return false;
    };
    num_part.len() == num_digits as usize && num_part.bytes().all(|b| b.is_ascii_digit())
}

/// Resolve a dropped path to a seed image file.
///
/// - If `path` is a file, returns it directly.
/// - If `path` is a directory, scans for the first file (alphabetically) whose
///   extension matches [`SUPPORTED_EXTENSIONS`] (case-insensitive).
/// - Returns `None` if no supported image is found or the path doesn't exist.
pub fn resolve_seed_file(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    if !path.is_dir() {
        return None;
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .ok()?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        let lower = ext.to_ascii_lowercase();
                        SUPPORTED_EXTENSIONS.iter().any(|&s| s == lower)
                    })
        })
        .collect();
    entries.sort();
    entries.into_iter().next()
}

/// Collect all files in `dir` matching the strict pattern
/// `{prefix}\d{num_digits}.{ext}`, sorted alphabetically.
///
/// Gaps in numbering are silently ignored — only existing files are returned.
pub fn collect_sequence_files(
    dir: &Path,
    prefix: &str,
    num_digits: u32,
    ext: &str,
) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut matched: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|name| matches_pattern(name, prefix, num_digits, ext))
        })
        .map(|e| e.path())
        .collect();
    matched.sort();
    matched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_explosion_sample() {
        let p = detect_pattern("explosion00-frame001.tga");
        assert_eq!(p, Some(("explosion00-frame".into(), 3, "tga".into())));
    }

    #[test]
    fn detect_no_digits() {
        assert!(detect_pattern("notes.txt").is_none());
    }

    #[test]
    fn match_strict() {
        assert!(matches_pattern("a012.png", "a", 3, "png"));
        assert!(!matches_pattern("a12.png", "a", 3, "png"));
        assert!(!matches_pattern("a012.jpg", "a", 3, "png"));
    }

    #[test]
    fn detect_all_digits_stem() {
        let p = detect_pattern("001.tga");
        assert_eq!(p, Some((String::new(), 3, "tga".into())));
    }

    #[test]
    fn detect_no_extension() {
        assert!(detect_pattern("frame001").is_none());
    }

    #[test]
    fn detect_multiple_dots() {
        let p = detect_pattern("my.seq.003.png");
        assert_eq!(p, Some(("my.seq.".into(), 3, "png".into())));
    }

    #[test]
    fn match_wrong_prefix() {
        assert!(!matches_pattern("b012.png", "a", 3, "png"));
    }

    // --- resolve_seed_file tests ---

    #[test]
    fn resolve_seed_file_with_existing_file() {
        // Cargo.toml is a known file in the repo root.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cargo_toml = root.join("Cargo.toml");
        assert_eq!(resolve_seed_file(&cargo_toml), Some(cargo_toml));
    }

    #[test]
    fn resolve_seed_file_nonexistent_returns_none() {
        let bogus = PathBuf::from("/nonexistent/path/to/nothing.tga");
        assert_eq!(resolve_seed_file(&bogus), None);
    }

    #[test]
    fn resolve_seed_file_dir_finds_image() {
        // Create a temp dir inside the project with a supported image file.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tmp_dir = root.join("_test_resolve_seed");
        let _ = std::fs::create_dir(&tmp_dir);
        let img = tmp_dir.join("frame001.tga");
        std::fs::write(&img, b"fake").unwrap();

        let result = resolve_seed_file(&tmp_dir);
        assert_eq!(result, Some(img.clone()));

        // Cleanup.
        let _ = std::fs::remove_file(&img);
        let _ = std::fs::remove_dir(&tmp_dir);
    }

    #[test]
    fn resolve_seed_file_dir_no_images() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tmp_dir = root.join("_test_resolve_seed_empty");
        let _ = std::fs::create_dir(&tmp_dir);
        let txt = tmp_dir.join("readme.txt");
        std::fs::write(&txt, b"not an image").unwrap();

        let result = resolve_seed_file(&tmp_dir);
        assert_eq!(result, None);

        let _ = std::fs::remove_file(&txt);
        let _ = std::fs::remove_dir(&tmp_dir);
    }

    // --- collect_sequence_files tests ---

    #[test]
    fn collect_sequence_files_filters_and_sorts() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tmp_dir = root.join("_test_collect_seq");
        let _ = std::fs::create_dir(&tmp_dir);

        // Create matching files (with a gap at 002).
        for name in &["frame001.tga", "frame003.tga", "frame004.tga"] {
            std::fs::write(tmp_dir.join(name), b"x").unwrap();
        }
        // Non-matching files.
        std::fs::write(tmp_dir.join("frame01.tga"), b"x").unwrap(); // wrong digit count
        std::fs::write(tmp_dir.join("other001.tga"), b"x").unwrap(); // wrong prefix

        let result = collect_sequence_files(&tmp_dir, "frame", 3, "tga");
        let names: Vec<&str> = result
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect();
        assert_eq!(names, vec!["frame001.tga", "frame003.tga", "frame004.tga"]);

        // Cleanup.
        for entry in std::fs::read_dir(&tmp_dir).unwrap() {
            let _ = std::fs::remove_file(entry.unwrap().path());
        }
        let _ = std::fs::remove_dir(&tmp_dir);
    }

    #[test]
    fn collect_sequence_files_nonexistent_dir() {
        let result = collect_sequence_files(Path::new("/no/such/dir"), "a", 3, "png");
        assert!(result.is_empty());
    }
}
