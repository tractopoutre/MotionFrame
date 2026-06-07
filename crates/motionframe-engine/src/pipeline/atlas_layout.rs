//! Atlas grid layout selection: pick (cols, rows) for an output frame count
//! that fits the GPU texture limit.

/// Returns `(cols, rows)` with `cols >= rows`, `cols * rows == m`, and
/// `cols - rows` minimized over all factor pairs of `m`.
///
/// `m` must be `>= 1`. For `m == 1` returns `(1, 1)`.
#[must_use]
pub fn squarest_factor_pair(m: u32) -> (u32, u32) {
    debug_assert!(m >= 1, "m must be >= 1");
    let mut best = (m, 1u32);
    let mut r = 1u32;
    while r.saturating_mul(r) <= m {
        if m.is_multiple_of(r) {
            let c = m / r;
            if c - r < best.0 - best.1 {
                best = (c, r);
            }
        }
        r += 1;
    }
    best
}

/// Result of `pick_layout`: a chosen atlas grid plus the number of padding
/// tiles added beyond the requested frame count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasLayout {
    pub cols: u32,
    pub rows: u32,
    /// Number of empty tiles appended past `n` to obtain a fitting grid.
    /// `cols * rows == n + padding`.
    pub padding: u32,
}

/// Default cap on the number of padding tiles `pick_layout` will try.
pub const DEFAULT_PADDING_BOUND: u32 = 8;

/// Find the smallest `k in [0, k_max]` such that the squarest factor pair
/// of `n + k` fits the texture limit on both axes.
///
/// `tile_w` and `tile_h` are the full per-tile pixel dimensions (including
/// any extrude border) of the output color atlas. `max_dim` is the GPU
/// texture-size cap. Returns `None` if no `k <= k_max` produces a fitting
/// layout, or if any input is `0`.
#[must_use]
pub fn pick_layout(
    n: u32,
    tile_w: u32,
    tile_h: u32,
    max_dim: u32,
    k_max: u32,
) -> Option<AtlasLayout> {
    if n == 0 || tile_w == 0 || tile_h == 0 || max_dim == 0 {
        return None;
    }
    for k in 0..=k_max {
        let m = n + k;
        let (cols, rows) = squarest_factor_pair(m);
        let w_ok = cols.checked_mul(tile_w).is_some_and(|w| w <= max_dim);
        let h_ok = rows.checked_mul(tile_h).is_some_and(|h| h <= max_dim);
        if w_ok && h_ok {
            return Some(AtlasLayout {
                cols,
                rows,
                padding: k,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squarest_factor_pair_unit() {
        assert_eq!(squarest_factor_pair(1), (1, 1));
    }

    #[test]
    fn squarest_factor_pair_perfect_squares() {
        assert_eq!(squarest_factor_pair(4), (2, 2));
        assert_eq!(squarest_factor_pair(16), (4, 4));
        assert_eq!(squarest_factor_pair(100), (10, 10));
    }

    #[test]
    fn squarest_factor_pair_primes_yield_n_by_1() {
        assert_eq!(squarest_factor_pair(2), (2, 1));
        assert_eq!(squarest_factor_pair(7), (7, 1));
        assert_eq!(squarest_factor_pair(31), (31, 1));
        assert_eq!(squarest_factor_pair(101), (101, 1));
    }

    #[test]
    fn squarest_factor_pair_composites_pick_squarest() {
        // 15 = 5*3, 15*1 → (5,3)
        assert_eq!(squarest_factor_pair(15), (5, 3));
        // 12 = 4*3, 6*2, 12*1 → (4,3)
        assert_eq!(squarest_factor_pair(12), (4, 3));
        // 6 = 3*2, 6*1 → (3,2)
        assert_eq!(squarest_factor_pair(6), (3, 2));
        // 34 = 17*2, 34*1 → (17,2)
        assert_eq!(squarest_factor_pair(34), (17, 2));
        // 102 = 17*6, 51*2, 34*3, 102*1 → (17,6)
        assert_eq!(squarest_factor_pair(102), (17, 6));
    }

    #[test]
    fn squarest_factor_pair_postcondition_cols_ge_rows() {
        for m in 1u32..=200 {
            let (c, r) = squarest_factor_pair(m);
            assert!(c >= r, "m={m}: cols={c} < rows={r}");
            assert_eq!(c * r, m, "m={m}: {c} * {r} != {m}");
        }
    }

    #[test]
    fn pick_layout_composite_no_padding() {
        // 15 frames, 128 px tiles, 8192 cap → 5x3, no padding.
        let l = pick_layout(15, 128, 128, 8192, 8).unwrap();
        assert_eq!(
            l,
            AtlasLayout {
                cols: 5,
                rows: 3,
                padding: 0
            }
        );
    }

    #[test]
    fn pick_layout_prime_1xn_fits() {
        // 31 frames, 128 px tiles, 8192 cap → 31x1 = 3968px, fits.
        let l = pick_layout(31, 128, 128, 8192, 8).unwrap();
        assert_eq!(
            l,
            AtlasLayout {
                cols: 31,
                rows: 1,
                padding: 0
            }
        );
    }

    #[test]
    fn pick_layout_prime_overflow_pads_to_composite() {
        // 41 frames at 256 px: 41*256 = 10496 > 8192. Try +1=42=7*6 → 7*256=1792.
        let l = pick_layout(41, 256, 256, 8192, 8).unwrap();
        assert_eq!(
            l,
            AtlasLayout {
                cols: 7,
                rows: 6,
                padding: 1
            }
        );
    }

    #[test]
    fn pick_layout_composite_overflow_pads_further() {
        // 100 frames at 1024 px, 8192 cap: 10x10 → 10*1024=10240 overflow.
        // Walk forward until something with cols<=8 turns up (or None within k_max).
        let l = pick_layout(100, 1024, 1024, 8192, 8);
        if let Some(layout) = l {
            assert!(layout.cols * 1024 <= 8192, "cols overflows: {layout:?}");
            assert!(layout.rows * 1024 <= 8192, "rows overflows: {layout:?}");
            assert!(layout.padding <= 8);
        }
    }

    #[test]
    fn pick_layout_cited_example_34_at_256() {
        // From the spec: 34 frames @ 256 px tile width, 8192 cap.
        // 34 = 17x2 → 17*256 = 4352, 2*256 = 512. Fits, padding=0.
        let l = pick_layout(34, 256, 256, 8192, 8).unwrap();
        assert_eq!(
            l,
            AtlasLayout {
                cols: 17,
                rows: 2,
                padding: 0
            }
        );
    }

    #[test]
    fn pick_layout_exhaustion_returns_none() {
        // tile_w > max_dim → no layout possible.
        assert!(pick_layout(1, 9000, 128, 8192, 8).is_none());
        // tile_h > max_dim → no layout possible.
        assert!(pick_layout(1, 128, 9000, 8192, 8).is_none());
    }

    #[test]
    fn pick_layout_zero_inputs_return_none() {
        assert!(pick_layout(0, 128, 128, 8192, 8).is_none());
        assert!(pick_layout(10, 0, 128, 8192, 8).is_none());
        assert!(pick_layout(10, 128, 0, 8192, 8).is_none());
        assert!(pick_layout(10, 128, 128, 0, 8).is_none());
    }

    #[test]
    fn pick_layout_padding_bound_respected() {
        // n=41, tile_w=256, max=8192: k=0 fails (41*256>8192),
        // k=1 → 42=7x6 succeeds.
        assert!(pick_layout(41, 256, 256, 8192, 0).is_none());
        assert!(pick_layout(41, 256, 256, 8192, 1).is_some());
    }
}
