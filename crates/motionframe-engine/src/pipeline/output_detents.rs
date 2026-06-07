//! Slider-detent set for the output-count slider in the desktop UI.
//!
//! Given `n_input` source frames, enumerate the strictly-decreasing-skip
//! sequence of `(output_count, frame_skip)` pairs where each step yields
//! a fresh distinct output count and `output_count >= 2`.

use crate::pipeline::run::calculate_required_frames;

/// One slider stop: an output frame count and the smallest `frame_skip`
/// that produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetentEntry {
    pub output_count: u32,
    pub frame_skip: u32,
    pub ignored_tail_frames: u32,
}

/// Build the slider's canonical detent set for `n_input` input frames.
///
/// Returns entries in ascending order of `output_count`. Excludes
/// `output_count == 1` (single-frame output is not useful). For
/// `n_input < 2`, returns an empty vec.
#[must_use]
pub fn build_canonical_layouts(n_input: u32) -> Vec<DetentEntry> {
    build_output_count_detents(n_input, false)
}

/// Build the slider's output-count detent set for `n_input` input frames.
///
/// When `trim_tail_for_exact_output_count` is true, every output count that can
/// be produced by an exact input prefix is included, with the fewest possible
/// ignored tail frames for that count.
#[must_use]
pub fn build_output_count_detents(
    n_input: u32,
    trim_tail_for_exact_output_count: bool,
) -> Vec<DetentEntry> {
    if trim_tail_for_exact_output_count {
        build_trimmed_layouts(n_input)
    } else {
        build_untrimmed_layouts(n_input)
    }
}

fn build_untrimmed_layouts(n_input: u32) -> Vec<DetentEntry> {
    if n_input < 2 {
        return Vec::new();
    }
    let mut out: Vec<DetentEntry> = Vec::new();
    // Walk skip from 0 upward. As skip grows, output_count strictly
    // decreases (or stays the same — those positions are dups). The first
    // appearance of each count is at its smallest-producing skip; that's
    // what we record. Skip values that drop output_count below 2 stop
    // the walk.
    for skip in 0..n_input {
        let count = calculate_required_frames(n_input as usize, skip) as u32;
        if count < 2 {
            break;
        }
        let fresh = out.last().is_none_or(|e| e.output_count != count);
        if fresh {
            out.push(DetentEntry {
                output_count: count,
                frame_skip: skip,
                ignored_tail_frames: 0,
            });
        }
    }
    // Currently in descending count order; reverse to ascending.
    out.reverse();
    out
}

fn build_trimmed_layouts(n_input: u32) -> Vec<DetentEntry> {
    if n_input < 2 {
        return Vec::new();
    }

    let untrimmed = build_untrimmed_layouts(n_input);
    let mut out = Vec::new();
    for output_count in 2..=n_input {
        if let Some(entry) = untrimmed
            .iter()
            .find(|entry| entry.output_count == output_count)
            .copied()
            .or_else(|| best_trimmed_entry(n_input, output_count))
        {
            out.push(entry);
        }
    }
    out
}

const fn best_trimmed_entry(n_input: u32, output_count: u32) -> Option<DetentEntry> {
    let step = n_input / output_count;
    if step == 0 {
        return None;
    }

    let used_frames = output_count * step;
    Some(DetentEntry {
        output_count,
        frame_skip: step - 1,
        ignored_tail_frames: n_input - used_frames,
    })
}

/// Snap a possibly-stale `frame_skip` to the canonical detent for the
/// current input frame count. Returns the canonicalized `frame_skip`.
///
/// Behavior:
/// - If `frame_skip` already produces a count `>= 2` that matches a layout
///   entry, return that entry's (minimal) `frame_skip`. Idempotent on
///   already-canonical inputs.
/// - If the produced count is `< 2` (e.g., source got smaller, leaving a
///   too-large skip from a prior session), return the first layout's skip
///   (the smallest-count detent).
/// - If `layouts` is empty or `n_input < 2`, return the input unchanged.
#[must_use]
pub fn snap_to_canonical_skip(layouts: &[DetentEntry], n_input: u32, frame_skip: u32) -> u32 {
    if layouts.is_empty() || n_input < 2 {
        return frame_skip;
    }
    let count_now = calculate_required_frames(n_input as usize, frame_skip) as u32;
    if let Some(entry) = layouts.iter().find(|e| e.output_count == count_now) {
        return entry.frame_skip;
    }
    layouts.first().map_or(frame_skip, |e| e.frame_skip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_below_two_inputs() {
        assert_eq!(build_canonical_layouts(0), vec![]);
        assert_eq!(build_canonical_layouts(1), vec![]);
    }

    #[test]
    fn n_two_yields_single_entry() {
        assert_eq!(
            build_canonical_layouts(2),
            vec![DetentEntry {
                output_count: 2,
                frame_skip: 0,
                ignored_tail_frames: 0,
            }]
        );
    }

    #[test]
    fn n_three_yields_two_entries() {
        assert_eq!(
            build_canonical_layouts(3),
            vec![
                DetentEntry {
                    output_count: 2,
                    frame_skip: 1,
                    ignored_tail_frames: 0,
                },
                DetentEntry {
                    output_count: 3,
                    frame_skip: 0,
                    ignored_tail_frames: 0,
                },
            ]
        );
    }

    #[test]
    fn n_ten_has_expected_layout() {
        let layouts = build_canonical_layouts(10);
        assert_eq!(
            layouts,
            vec![
                DetentEntry {
                    output_count: 2,
                    frame_skip: 4,
                    ignored_tail_frames: 0,
                },
                DetentEntry {
                    output_count: 3,
                    frame_skip: 3,
                    ignored_tail_frames: 0,
                },
                DetentEntry {
                    output_count: 4,
                    frame_skip: 2,
                    ignored_tail_frames: 0,
                },
                DetentEntry {
                    output_count: 5,
                    frame_skip: 1,
                    ignored_tail_frames: 0,
                },
                DetentEntry {
                    output_count: 10,
                    frame_skip: 0,
                    ignored_tail_frames: 0,
                },
            ]
        );
    }

    #[test]
    fn n_hundred_has_expected_endpoints_and_count() {
        let layouts = build_canonical_layouts(100);
        // Distinct output counts ≥ 2 for N=100:
        // {2,3,4,5,6,7,8,9,10,12,13,15,17,20,25,34,50,100} → 18 entries.
        // (No 11/14/16/18/19/21..24/26..33/35..49/51..99 are reachable.)
        assert_eq!(layouts.len(), 18);
        assert_eq!(
            layouts.first().copied(),
            Some(DetentEntry {
                output_count: 2,
                frame_skip: 49,
                ignored_tail_frames: 0,
            })
        );
        assert_eq!(
            layouts.last().copied(),
            Some(DetentEntry {
                output_count: 100,
                frame_skip: 0,
                ignored_tail_frames: 0,
            })
        );
    }

    #[test]
    fn hundred_without_trim_does_not_offer_sixteen() {
        let layouts = build_canonical_layouts(100);

        assert!(!layouts.iter().any(|entry| entry.output_count == 16));
    }

    #[test]
    fn hundred_with_trim_offers_sixteen_from_first_ninety_six() {
        let layouts = build_output_count_detents(100, true);
        let entry = layouts
            .iter()
            .find(|entry| entry.output_count == 16)
            .expect("16-frame trimmed detent");

        assert_eq!(entry.frame_skip, 5);
        assert_eq!(entry.ignored_tail_frames, 4);
    }

    #[test]
    fn untrimmed_counts_report_zero_ignored_tail_frames() {
        let layouts = build_output_count_detents(100, true);
        let entry = layouts
            .iter()
            .find(|entry| entry.output_count == 20)
            .expect("20-frame untrimmed detent");

        assert_eq!(entry.frame_skip, 4);
        assert_eq!(entry.ignored_tail_frames, 0);
    }

    #[test]
    fn trim_mode_preserves_existing_untrimmed_detents() {
        let layouts = build_output_count_detents(100, true);
        let entry = layouts
            .iter()
            .find(|entry| entry.output_count == 17)
            .expect("17-frame untrimmed detent");

        assert_eq!(entry.frame_skip, 5);
        assert_eq!(entry.ignored_tail_frames, 0);
    }

    #[test]
    fn trimmed_detents_prefer_fewest_ignored_tail_frames() {
        let layouts = build_output_count_detents(100, true);
        let entry = layouts
            .iter()
            .find(|entry| entry.output_count == 33)
            .expect("33-frame trimmed detent");

        assert_eq!(entry.frame_skip, 2);
        assert_eq!(entry.ignored_tail_frames, 1);
    }

    #[test]
    fn postconditions_hold_across_sizes() {
        for n in [2u32, 3, 10, 100, 1000] {
            let layouts = build_canonical_layouts(n);
            assert!(!layouts.is_empty(), "n={n}: empty layouts");

            // Counts strictly increasing.
            for w in layouts.windows(2) {
                assert!(
                    w[0].output_count < w[1].output_count,
                    "n={n}: counts not increasing"
                );
                assert!(
                    w[0].frame_skip > w[1].frame_skip,
                    "n={n}: skips not decreasing"
                );
            }

            // Endpoints.
            assert!(layouts.first().unwrap().output_count >= 2);
            assert_eq!(layouts.last().unwrap().output_count, n);
            assert_eq!(layouts.last().unwrap().frame_skip, 0);

            // Each entry's count matches calculate_required_frames(n, skip).
            for entry in &layouts {
                let actual = calculate_required_frames(n as usize, entry.frame_skip) as u32;
                assert_eq!(
                    actual, entry.output_count,
                    "n={n}: count mismatch at skip={}",
                    entry.frame_skip
                );
            }

            // Each entry's frame_skip is the smallest skip producing its count.
            for entry in &layouts {
                if entry.frame_skip > 0 {
                    let smaller =
                        calculate_required_frames(n as usize, entry.frame_skip - 1) as u32;
                    assert!(
                        smaller > entry.output_count,
                        "n={n}: skip={} not minimal for count={}",
                        entry.frame_skip,
                        entry.output_count
                    );
                }
            }
        }
    }

    #[test]
    fn snap_already_canonical_is_idempotent() {
        let l = build_canonical_layouts(10);
        // skip=3 → count=3, canonical for n=10.
        assert_eq!(snap_to_canonical_skip(&l, 10, 3), 3);
        // skip=4 → count=2, canonical for n=10.
        assert_eq!(snap_to_canonical_skip(&l, 10, 4), 4);
        // skip=0 → count=10, canonical.
        assert_eq!(snap_to_canonical_skip(&l, 10, 0), 0);
    }

    #[test]
    fn snap_non_minimal_to_minimal() {
        let l = build_canonical_layouts(10);
        // skip=5,6,7,8 all produce count=2 in n=10; minimal is skip=4.
        assert_eq!(snap_to_canonical_skip(&l, 10, 5), 4);
        assert_eq!(snap_to_canonical_skip(&l, 10, 8), 4);
    }

    #[test]
    fn snap_count_lt_two_falls_back_to_first() {
        let l = build_canonical_layouts(10);
        // skip=9 → count=1, excluded. Snap to layouts[0].frame_skip = 4.
        assert_eq!(snap_to_canonical_skip(&l, 10, 9), 4);
    }

    #[test]
    fn snap_input_atlas_shrink_recovers() {
        // Simulates input_atlas_dims change: previously n=10 with skip=4
        // (canonical for count=2). Source shrinks to n=4. skip=4 now
        // produces count=ceil(4/5)=1 — falls below the canonical floor.
        // Snap to first layout for n=4 (count=2, skip=1).
        let l = build_canonical_layouts(4);
        assert_eq!(snap_to_canonical_skip(&l, 4, 4), 1);
    }

    #[test]
    fn snap_empty_layouts_passthrough() {
        assert_eq!(snap_to_canonical_skip(&[], 0, 7), 7);
        assert_eq!(snap_to_canonical_skip(&[], 1, 7), 7);
    }

    #[test]
    fn snap_n_input_lt_two_passthrough() {
        let l = build_canonical_layouts(10);
        assert_eq!(snap_to_canonical_skip(&l, 1, 5), 5);
        assert_eq!(snap_to_canonical_skip(&l, 0, 5), 5);
    }
}
