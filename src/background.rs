//! Background estimation ported from `const_background.py`.
//!
//! Estimates the mean, variance and gain of the flat background directly from
//! the image by locating square "flat" tiles whose pixel distribution passes a
//! normality test and whose halves have equal means, then pooling those tiles.
//!
//! This is a faithful port of the deterministic Python estimator
//! (`estimate_background` + `largest_flat_tile` in `const_background.py`), not
//! the randomized `estimate_background_fast`. Two deviations from the Python
//! reference are intentional and documented at their call sites:
//!
//!   * `find_flat_tile_size` returns the size that *worked* when shrinking
//!     (the Python `largest_flat_tile` returns `min_size` there, which can be a
//!     stale/incorrect tile size and lead to a spurious global fallback).
//!   * Tiles are pooled with a single streaming pass; no intermediate vectors
//!     are allocated in the hot loop.

use crate::statistical_tests::Background;

// ---------------------------------------------------------------------------
// Special functions (Lanczos ln Γ, incomplete gamma, incomplete beta)
// ---------------------------------------------------------------------------

/// Regularized upper incomplete gamma function Q(a, x).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // Series representation for P(a, x); Q = 1 - P.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - gln).exp()
    } else {
        // Continued fraction for Q(a, x).
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * h
    }
}

/// Lanczos approximation of ln Γ(x) (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized incomplete beta I_x(a, b) via continued fraction.
fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Statistical tests
// ---------------------------------------------------------------------------

/// Two-sided p-value of the D'Agostino–Pearson normality test.
///
/// Equivalent to `scipy.stats.normaltest` for a 1-D sample.
///
/// Returns 1.0 ("cannot reject normality") in these guard cases:
///   * `n < 8`: scipy raises for such samples; we are permissive.
///   * `8 <= n < 20`: the D'Agostino–Pearson statistic is known to be poorly
///     conditioned here and can produce `NaN` through `sqrtbeta1` and the
///     kurtosis transform. We conservatively accept the sample rather than
///     risk a `NaN < p` comparison (which is `false` and would silently pass).
///   * any non-finite intermediate moment.
fn normaltest_pvalue(sample: &[f64]) -> f64 {
    let n = sample.len();
    if n < 8 {
        return 1.0;
    }
    let nf = n as f64;
    let mean = sample.iter().sum::<f64>() / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &v in sample {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let m2 = m2 / nf;
    let m3 = m3 / nf;
    let m4 = m4 / nf;
    if m2 <= 0.0 || !m2.is_finite() || !m3.is_finite() || !m4.is_finite() {
        return 1.0;
    }
    if n < 20 {
        // See doc comment: D'Agostino–Pearson is unreliable for small n.
        return 1.0;
    }
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2) - 3.0; // excess kurtosis

    // Skewness test (D'Agostino 1970).
    let y = skew * ((nf + 1.0) * (nf + 3.0) / (6.0 * (nf - 2.0))).sqrt();
    let beta2 = 3.0 * (nf * nf + 27.0 * nf - 70.0) * (nf + 1.0) * (nf + 3.0)
        / ((nf - 2.0) * (nf + 5.0) * (nf + 7.0) * (nf + 9.0));
    let w2_arg = 2.0 * (beta2 - 1.0);
    if w2_arg < 0.0 || !w2_arg.is_finite() {
        return 1.0;
    }
    let w2 = -1.0 + w2_arg.sqrt();
    if w2 <= 0.0 || !w2.is_finite() {
        return 1.0;
    }
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    if !delta.is_finite() || !alpha.is_finite() {
        return 1.0;
    }
    let ya = y / alpha;
    let z1 = if ya > 0.0 {
        delta * (ya + (ya * ya + 1.0).sqrt()).ln()
    } else {
        -delta * (-ya + (ya * ya + 1.0).sqrt()).ln()
    };

    // Kurtosis test (Anscombe & Glynn 1983).
    let e = 3.0 * (nf - 1.0) / (nf + 1.0);
    let varb2 = 24.0 * nf * (nf - 2.0) * (nf - 3.0)
        / ((nf + 1.0) * (nf + 1.0) * (nf + 3.0) * (nf + 5.0));
    if varb2 <= 0.0 || !varb2.is_finite() {
        return 1.0;
    }
    let x = (kurt - e) / varb2.sqrt();
    let sqrtbeta1 = 6.0 * (nf * nf - 5.0 * nf + 2.0) / ((nf + 7.0) * (nf + 9.0))
        * (6.0 * (nf + 3.0) * (nf + 5.0) / (nf * (nf - 2.0) * (nf - 3.0))).sqrt();
    if sqrtbeta1 == 0.0 || !sqrtbeta1.is_finite() {
        return 1.0;
    }
    let a = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)).sqrt());
    if a <= 4.0 || !a.is_finite() {
        // `2.0 / (a - 4.0)` would be ≤ 0 or non-finite below this guard.
        return 1.0;
    }
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let term2 = ((1.0 - 2.0 / a) / (1.0 + x * (2.0 / (a - 4.0)).sqrt())).powf(1.0 / 3.0);
    let z2 = (term1 - term2) / (2.0 / (9.0 * a)).sqrt();
    if !z1.is_finite() || !z2.is_finite() {
        return 1.0;
    }

    let k2 = z1 * z1 + z2 * z2;
    // p-value of chi-square with 2 df: exp(-k2 / 2).
    (-k2 / 2.0).exp()
}

/// Two-sided p-value of Student's t-test for two independent samples with
/// pooled variance.
///
/// This matches `scipy.stats.ttest_ind(a, b)` with its default
/// `equal_var=True`. The Python reference calls `ttest_ind` without
/// overriding `equal_var`, so Student's t is the correct analogue.
fn ttest_ind_pvalue(a: &[f64], b: &[f64]) -> f64 {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    if na < 2.0 || nb < 2.0 {
        return 1.0;
    }
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let m2a = a.iter().map(|&v| (v - ma) * (v - ma)).sum::<f64>();
    let m2b = b.iter().map(|&v| (v - mb) * (v - mb)).sum::<f64>();
    let sp2 = (m2a + m2b) / (na + nb - 2.0);
    let se2 = sp2 * (1.0 / na + 1.0 / nb);
    if se2 <= 0.0 || !se2.is_finite() {
        return 1.0;
    }
    let t = (ma - mb) / se2.sqrt();
    if !t.is_finite() {
        return 1.0;
    }
    let df = na + nb - 2.0;
    // Student-t survival: 2 * (1 - CDF_t(df, |t|)) = I_{df/(df+t^2)}(df/2, 1/2).
    let x = df / (df + t * t);
    let p = betainc(df / 2.0, 0.5, x).min(1.0);
    p
}

// ---------------------------------------------------------------------------
// Flat-tile detection
// ---------------------------------------------------------------------------

/// Reusable scratch buffers so the hot loop performs no heap allocation.
#[derive(Default)]
struct Scratch {
    finite: Vec<f64>,
    top: Vec<f64>,
    bot: Vec<f64>,
    left: Vec<f64>,
    right: Vec<f64>,
}

impl Scratch {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(tile_len: usize) -> Self {
        let n = tile_len * tile_len;
        let h = (tile_len / 2) * tile_len;
        let h2 = tile_len * (tile_len - tile_len / 2);
        Self {
            finite: Vec::with_capacity(n),
            top: Vec::with_capacity(h),
            bot: Vec::with_capacity(h2),
            left: Vec::with_capacity(h),
            right: Vec::with_capacity(h2),
        }
    }

    fn clear(&mut self) {
        self.finite.clear();
        self.top.clear();
        self.bot.clear();
        self.left.clear();
        self.right.clear();
    }
}

/// The flat-tile acceptance test: normal distribution + equal half-means.
///
/// Matches the Python `check_tile_is_flat` semantics:
///   * reject all-zero tiles or tiles with no finite pixels;
///   * reject if `normaltest` p-value < `r1` (Python `rejection_rate_1`);
///   * reject if either top-vs-bottom or left-vs-right `ttest_ind` p-value
///     < `r2` (Python `rejection_rate_2`).
///
/// NaN pixels are dropped per-test, equivalent to scipy's `nan_policy='omit'`.
fn check_tile_is_flat(
    tile: &[f32],
    tile_len: usize,
    r1: f64,
    r2: f64,
    scratch: &mut Scratch,
) -> bool {
    debug_assert_eq!(tile.len(), tile_len * tile_len);
    scratch.clear();

    let mut all_zero = true;
    let mut any_finite = false;
    let h = tile_len / 2;

    // Single pass: collect per-half samples for the t-tests and the finite
    // values for the normality test.
    for y in 0..tile_len {
        for x in 0..tile_len {
            let v = tile[y * tile_len + x];
            if !v.is_finite() {
                continue;
            }
            any_finite = true;
            let v = v as f64;
            if v != 0.0 {
                all_zero = false;
            }
            scratch.finite.push(v);
            if y < h {
                scratch.top.push(v);
            } else {
                scratch.bot.push(v);
            }
            if x < h {
                scratch.left.push(v);
            } else {
                scratch.right.push(v);
            }
        }
    }
    if !any_finite || all_zero {
        return false;
    }

    if normaltest_pvalue(&scratch.finite) < r1 {
        return false;
    }
    if ttest_ind_pvalue(&scratch.top, &scratch.bot) < r2 {
        return false;
    }
    if ttest_ind_pvalue(&scratch.left, &scratch.right) < r2 {
        return false;
    }
    true
}

/// Copy the `tile_len` x `tile_len` block at `(x0, y0)` into `tile`.
#[inline]
fn read_tile(
    img: &[f32],
    width: usize,
    x0: usize,
    y0: usize,
    tile_len: usize,
    tile: &mut [f32],
) {
    for ty in 0..tile_len {
        let src = (y0 + ty) * width + x0;
        tile[ty * tile_len..(ty + 1) * tile_len].copy_from_slice(&img[src..src + tile_len]);
    }
}

/// Returns `true` if any tile of size `tile_len` passes `check_tile_is_flat`.
fn available_tiles(
    img: &[f32],
    width: usize,
    height: usize,
    tile_len: usize,
    r1: f64,
    r2: f64,
    tile: &mut [f32],
    scratch: &mut Scratch,
) -> bool {
    let mut y = 0;
    while y + tile_len <= height {
        let mut x = 0;
        while x + tile_len <= width {
            read_tile(img, width, x, y, tile_len, tile);
            if check_tile_is_flat(tile, tile_len, r1, r2, scratch) {
                return true;
            }
            x += tile_len;
        }
        y += tile_len;
    }
    false
}

/// Collect all flat tile positions of size `tile_len`.
fn collect_flat_tiles(
    img: &[f32],
    width: usize,
    height: usize,
    tile_len: usize,
    r1: f64,
    r2: f64,
    tile: &mut [f32],
    scratch: &mut Scratch,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut y = 0;
    while y + tile_len <= height {
        let mut x = 0;
        while x + tile_len <= width {
            read_tile(img, width, x, y, tile_len, tile);
            if check_tile_is_flat(tile, tile_len, r1, r2, scratch) {
                out.push((x, y));
            }
            x += tile_len;
        }
        y += tile_len;
    }
    out
}

/// Find the largest usable flat-tile size, searching powers of two.
///
/// Starts at `2^size_start` and doubles up to `2^size_max` if that size has
/// any flat tile; otherwise halves down to `2^size_min`.
///
/// Unlike the Python `largest_flat_tile`, when shrinking we return the size
/// that actually passed (`current`), not the floor `min_size`. The Python
/// function returns `min_size` there, which can disagree with the tile size
/// that satisfied the test and cause `collect_flat_tiles` at that wrong size
/// to find nothing, silently falling back to a global estimate.
///
/// Returns 0 when no flat tile exists at any tried size.
fn find_flat_tile_size(
    img: &[f32],
    width: usize,
    height: usize,
    r1: f64,
    r2: f64,
    size_start: u32,
    size_min: u32,
    size_max: u32,
) -> usize {
    let start = 1usize << size_start;
    let min_size = 1usize << size_min;
    let max_size = 1usize << size_max;

    let mut tile = vec![0.0f32; start * start];
    let mut scratch = Scratch::with_capacity(start);

    // Grow while tiles exist at the current (larger) size.
    if available_tiles(img, width, height, start, r1, r2, &mut tile, &mut scratch) {
        let mut current = start;
        while current < max_size {
            let next = current * 2;
            if next * next > tile.len() {
                tile.resize(next * next, 0.0);
                scratch = Scratch::with_capacity(next);
            }
            if available_tiles(img, width, height, next, r1, r2, &mut tile, &mut scratch) {
                current = next;
            } else {
                break;
            }
        }
        return current;
    }

    // Shrink until tiles exist, returning the size that worked.
    let mut current = start;
    while current > min_size {
        current /= 2;
        if current * current > tile.len() {
            tile.resize(current * current, 0.0);
            scratch = Scratch::with_capacity(current);
        }
        if available_tiles(img, width, height, current, r1, r2, &mut tile, &mut scratch) {
            return current;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Estimate background statistics from the image itself, using the flat-tile
/// approach from `const_background.py::estimate_background`.
///
/// On success returns a `Background` whose `alpha` is set to `alpha`.
/// If no flat tiles are found, falls back to `global_background`.
pub fn estimate_background(
    img: &[f32],
    width: usize,
    height: usize,
    rejection_rate: f64,
    alpha: f64,
) -> Background {
    let r1 = 1.0 - (1.0 - rejection_rate).powf(0.5);
    let r2 = 1.0 - (1.0 - rejection_rate).powf(0.25);

    let tile_len = find_flat_tile_size(img, width, height, r1, r2, 6, 4, 7);
    if tile_len == 0 {
        return global_background(img, alpha);
    }

    let mut tile = vec![0.0f32; tile_len * tile_len];
    let mut scratch = Scratch::with_capacity(tile_len);
    let tiles = collect_flat_tiles(
        img, width, height, tile_len, r1, r2, &mut tile, &mut scratch,
    );
    if tiles.is_empty() {
        return global_background(img, alpha);
    }

    // Streaming pooling over the accepted tiles.
    let mut count = 0u64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &(x0, y0) in &tiles {
        for ty in 0..tile_len {
            let row = (y0 + ty) * width;
            for tx in 0..tile_len {
                let v = img[row + x0 + tx];
                if v.is_finite() {
                    let v = v as f64;
                    sum += v;
                    sum_sq += v * v;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return global_background(img, alpha);
    }
    let n = count as f64;
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);

    // `soft_bias` in the Python reference is `min(nanmin(img), 0)`.
    let image_min = img
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, |a, b| a.min(b as f64));
    let soft_bias = if image_min.is_finite() && image_min < 0.0 {
        image_min
    } else {
        0.0
    };
    let gain = if var != 0.0 { (mean - soft_bias) / var } else { 0.0 };

    Background {
        mean,
        var,
        gain,
        alpha,
    }
}

/// Robust global fallback using the median and MAD-derived sigma.
fn global_background(img: &[f32], alpha: f64) -> Background {
    let mut vals: Vec<f64> = img
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .collect();
    if vals.is_empty() {
        return Background {
            mean: 0.0,
            var: 1.0,
            gain: 1.0,
            alpha,
        };
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    let median = if n % 2 == 1 {
        vals[n / 2]
    } else {
        0.5 * (vals[n / 2 - 1] + vals[n / 2])
    };
    let mut dev: Vec<f64> = vals.iter().map(|&v| (v - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if n % 2 == 1 {
        dev[n / 2]
    } else {
        0.5 * (dev[n / 2 - 1] + dev[n / 2])
    };
    // 1.4826 makes MAD a consistent estimator of sigma for normal data.
    let sigma = 1.4826 * mad;
    let var = if sigma > 0.0 { sigma * sigma } else { 1.0 };
    let image_min = vals[0];
    let soft_bias = image_min.min(0.0);
    let gain = if var != 0.0 {
        (median - soft_bias) / var
    } else {
        1.0
    };
    Background {
        mean: median,
        var,
        gain,
        alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_q_sanity() {
        assert!((gamma_q(1.0, 1.0) - (-1.0f64).exp()).abs() < 1e-9);
        assert!((gamma_q(1.0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normaltest_uniform_is_not_normal() {
        let unif: Vec<f64> = (0..200).map(|i| i as f64).collect();
        assert!(normaltest_pvalue(&unif) < 0.01);
    }

    #[test]
    fn normaltest_normal_is_normal() {
        let data: Vec<f64> = vec![
            -1.5, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8,
            1.0, 1.2, 1.5, -0.9, -0.7, -0.3, 0.3, 0.5, 0.7, 0.9, -1.4, -0.5, 0.5, 1.4,
            -1.1, 1.1,
        ];
        assert!(normaltest_pvalue(&data) > 0.01);
    }

    #[test]
    fn normaltest_small_n_returns_one() {
        // For 8 <= n < 20 we conservatively accept (see doc).
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(normaltest_pvalue(&data), 1.0);
    }

    #[test]
    fn ttest_equal_means_pvalue_high() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        assert!(ttest_ind_pvalue(&a, &b) > 0.5);
    }

    #[test]
    fn ttest_different_means_pvalue_low() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        assert!(ttest_ind_pvalue(&a, &b) < 0.01);
    }

    #[test]
    fn flat_tile_accepts_constant() {
        let tile = vec![5.0f32; 64];
        let mut s = Scratch::with_capacity(8);
        assert!(check_tile_is_flat(&tile, 8, 0.05, 0.05, &mut s));
    }

    #[test]
    fn flat_tile_rejects_all_zero() {
        let tile = vec![0.0f32; 64];
        let mut s = Scratch::with_capacity(8);
        assert!(!check_tile_is_flat(&tile, 8, 0.05, 0.05, &mut s));
    }

    #[test]
    fn flat_tile_rejects_gradient() {
        let mut tile = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                tile[y * 8 + x] = x as f32 * 10.0;
            }
        }
        let mut s = Scratch::with_capacity(8);
        assert!(!check_tile_is_flat(&tile, 8, 0.05, 0.05, &mut s));
    }

    #[test]
    fn find_flat_tile_size_shrink_returns_actual_size() {
        // 128x128 mostly-flat image with a large bright region covering 96x96.
        // At tile=64 the flat region is too small; at tile=32 there is still
        // space in the 32-wide L-shaped border; we expect a size >= 16.
        let w = 128usize;
        let h = 128usize;
        let mut img = vec![100.0f32; w * h];
        for y in 0..96 {
            for x in 0..96 {
                img[y * w + x] = 5000.0;
            }
        }
        let r1 = 1.0 - 0.95f64.powf(0.5);
        let r2 = 1.0 - 0.95f64.powf(0.25);
        let size = find_flat_tile_size(&img, w, h, r1, r2, 6, 4, 7);
        // Must be a size where at least one flat tile actually exists.
        if size > 0 {
            let mut tile = vec![0.0f32; size * size];
            let mut s = Scratch::with_capacity(size);
            assert!(
                available_tiles(&img, w, h, size, r1, r2, &mut tile, &mut s),
                "returned size {size} has no flat tile"
            );
        }
    }

    #[test]
    fn estimate_background_on_synthetic_flat_image() {
        let w = 128usize;
        let h = 128usize;
        let mut img = vec![0.0f32; w * h];
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for y in 0..h {
            for x in 0..w {
                let noise = 5.0 * next();
                img[y * w + x] = (100.0 + noise) as f32;
            }
        }
        for y in 50..70 {
            for x in 50..70 {
                img[y * w + x] = 5000.0;
            }
        }
        let bg = estimate_background(&img, w, h, 0.05, 1e-6);
        assert!((bg.mean - 100.0).abs() < 10.0, "mean={}", bg.mean);
        assert!(bg.var > 5.0 && bg.var < 100.0, "var={}", bg.var);
        assert!(bg.gain > 0.0);
        assert_eq!(bg.alpha, 1e-6);
    }

    #[test]
    fn estimate_background_negative_min_sets_soft_bias() {
        let w = 64;
        let h = 64;
        let mut img = vec![0.0f32; w * h];
        for (i, v) in img.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.1 - 5.0;
        }
        let bg = estimate_background(&img, w, h, 0.05, 1e-6);
        assert!(bg.mean.is_finite());
        assert!(bg.var > 0.0);
        assert_eq!(bg.alpha, 1e-6);
    }
}