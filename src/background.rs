//! Background estimation ported from `const_background.py`.
//!
//! Estimates the mean, variance and gain of the flat background directly from
//! the image by locating square "flat" tiles whose pixel distribution passes a
//! D'Agostino-Pearson normality test and whose halves have equal means, then
//! pooling those tiles.
//!
//! Faithful port of the deterministic Python estimator (`estimate_background`
//! + `largest_flat_tile`), with these deliberate differences:
//!
//!   * The kurtosis test uses the NON-excess kurtosis `m4/m2^2`, exactly like
//!     `scipy.stats.kurtosistest`. (The previous version subtracted 3 twice,
//!     which pushed the statistic to NaN and silently accepted nearly every
//!     tile.) Degenerate statistics now REJECT the tile instead of accepting.
//!   * Constant (zero-variance) tiles are rejected; they are not background
//!     (typically NaN-filled or saturated regions).
//!   * An optional `bad_mask` excludes tiles that touch invalid pixels.
//!   * `find_flat_tile_size` returns the size that actually worked when
//!     shrinking (Python returns `min_size`).
//!   * Failure returns `None` (Python raises) so the caller can switch to the
//!     morphological background, as `mto2.py` does.
//!   * Tiles are analysed with no heap allocation: two passes over the image
//!     rows, central moments and per-half sums, pooled with Chan's algorithm.

use crate::statistical_tests::Background;

/// scipy's kurtosis test needs n >= 20 to be meaningful.
const MIN_TILE_PIXELS: usize = 20;

// ---------------------------------------------------------------------------
// Special functions (Lanczos ln Γ, regularized incomplete beta)
// ---------------------------------------------------------------------------

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
    for m in 1..500 {
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

/// D'Agostino-Pearson K² statistic from central moments (population moments:
/// `m2 = Σd²/n`, etc.). Mirrors `scipy.stats.skewtest` + `kurtosistest`.
///
/// Returns `None` for degenerate input (n too small, zero variance, any
/// non-finite intermediate). Callers treat `None` as "reject the tile".
fn dagostino_k2(n: f64, m2: f64, m3: f64, m4: f64) -> Option<f64> {
    if n < MIN_TILE_PIXELS as f64
        || !(m2 > 0.0)
        || !m2.is_finite()
        || !m3.is_finite()
        || !m4.is_finite()
    {
        return None;
    }
    let skew = m3 / m2.powf(1.5);
    let b2 = m4 / (m2 * m2); // NON-excess kurtosis (scipy: fisher=False)

    // Skewness test (D'Agostino 1970).
    let y = skew * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let z1 = delta * (y / alpha).asinh();

    // Kurtosis test (Anscombe & Glynn 1983).
    let e = 3.0 * (n - 1.0) / (n + 1.0);
    let varb2 =
        24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    let x = (b2 - e) / varb2.sqrt();
    let sqrtbeta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)).sqrt());
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let denom = 1.0 + x * (2.0 / (a - 4.0)).sqrt();
    if denom == 0.0 {
        return None;
    }
    // scipy: sign(denom) * (|(1 - 2/A) / denom|)^(1/3)
    let term2 = denom.signum() * ((1.0 - 2.0 / a) / denom.abs()).powf(1.0 / 3.0);
    let z2 = (term1 - term2) / (2.0 / (9.0 * a)).sqrt();

    let k2 = z1 * z1 + z2 * z2;
    k2.is_finite().then_some(k2)
}

/// Two-sided Student-t p-value (pooled variance, like `scipy.stats.ttest_ind`)
/// from per-group `[n, Σd, Σd²]`, where `d` is shifted by a COMMON constant.
fn ttest_pvalue(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (na, nb) = (a[0], b[0]);
    if na < 2.0 || nb < 2.0 {
        return 1.0;
    }
    let (ma, mb) = (a[1] / na, b[1] / nb);
    let ssa = (a[2] - a[1] * a[1] / na).max(0.0);
    let ssb = (b[2] - b[1] * b[1] / nb).max(0.0);
    let df = na + nb - 2.0;
    let se2 = (ssa + ssb) / df * (1.0 / na + 1.0 / nb);
    if !(se2 > 0.0) || !se2.is_finite() {
        return if ma == mb { 1.0 } else { 0.0 };
    }
    let t = (ma - mb) / se2.sqrt();
    // Two-sided p = I_{df/(df+t²)}(df/2, 1/2).
    betainc(df / 2.0, 0.5, df / (df + t * t)).min(1.0)
}

// ---------------------------------------------------------------------------
// Flat-tile detection
// ---------------------------------------------------------------------------

/// Summary of an accepted tile (mean and sum of squared deviations).
#[derive(Clone, Copy)]
struct TileStats {
    n: u64,
    mean: f64,
    m2: f64,
}

struct Ctx<'a> {
    img: &'a [f32],
    width: usize,
    height: usize,
    /// `true` = pixel invalid; any tile touching one is rejected.
    bad: Option<&'a [bool]>,
    r1: f64,
    r2: f64,
}

impl Ctx<'_> {
    /// Flat-tile test. Returns the tile summary if the tile is accepted.
    ///
    /// Semantics follow Python `check_tile_is_flat`:
    ///   * reject all-zero tiles / tiles without finite pixels;
    ///   * reject if normaltest p < r1;
    ///   * reject if top/bottom or left/right t-test p < r2.
    /// Non-finite pixels are omitted (`nan_policy='omit'`).
    fn analyze_tile(&self, x0: usize, y0: usize, len: usize) -> Option<TileStats> {
        let w = self.width;

        // Pass 1: count, sum, validity.
        let mut n = 0usize;
        let mut sum = 0.0f64;
        let mut nonzero = false;
        for y in y0..y0 + len {
            let row = y * w + x0;
            if let Some(b) = self.bad {
                if b[row..row + len].iter().any(|&m| m) {
                    return None;
                }
            }
            for &v in &self.img[row..row + len] {
                if v.is_finite() {
                    n += 1;
                    sum += v as f64;
                    nonzero |= v != 0.0;
                }
            }
        }
        if n < MIN_TILE_PIXELS || !nonzero {
            return None;
        }
        let nf = n as f64;
        let mean = sum / nf;

        // Pass 2: central moments + per-half sums [n, Σd, Σd²].
        // Index: 0 = top, 1 = bottom, 2 = left, 3 = right.
        let h = len / 2;
        let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
        let mut hs = [[0.0f64; 3]; 4];
        for ty in 0..len {
            let row = (y0 + ty) * w + x0;
            let r = if ty < h { 0 } else { 1 };
            for (tx, &v) in self.img[row..row + len].iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                let d = v as f64 - mean;
                let d2 = d * d;
                m2 += d2;
                m3 += d2 * d;
                m4 += d2 * d2;
                let c = if tx < h { 2 } else { 3 };
                hs[r][0] += 1.0;
                hs[r][1] += d;
                hs[r][2] += d2;
                hs[c][0] += 1.0;
                hs[c][1] += d;
                hs[c][2] += d2;
            }
        }

        let k2 = dagostino_k2(nf, m2 / nf, m3 / nf, m4 / nf)?;
        if (-0.5 * k2).exp() < self.r1 {
            return None; // chi-square(2) survival is exp(-k2/2)
        }
        if ttest_pvalue(hs[0], hs[1]) < self.r2 || ttest_pvalue(hs[2], hs[3]) < self.r2 {
            return None;
        }
        Some(TileStats { n: n as u64, mean, m2 })
    }

    /// Visit accepted tiles on the non-overlapping grid; `f` returns `true`
    /// to stop early.
    fn for_each_flat_tile(&self, len: usize, mut f: impl FnMut(TileStats) -> bool) {
        if len == 0 {
            return;
        }
        let mut y = 0;
        while y + len <= self.height {
            let mut x = 0;
            while x + len <= self.width {
                if let Some(t) = self.analyze_tile(x, y, len) {
                    if f(t) {
                        return;
                    }
                }
                x += len;
            }
            y += len;
        }
    }

    fn any_flat(&self, len: usize) -> bool {
        let mut found = false;
        self.for_each_flat_tile(len, |_| {
            found = true;
            true
        });
        found
    }

    /// Largest usable flat-tile size among powers of two.
    ///
    /// Starts at `2^size_start`; if a flat tile exists, doubles up to
    /// `2^size_max`, otherwise halves down to `2^size_min`. When shrinking,
    /// returns the size that worked. Returns 0 if nothing works.
    fn find_flat_tile_size(&self, size_start: u32, size_min: u32, size_max: u32) -> usize {
        let start = 1usize << size_start;
        let min_size = 1usize << size_min;
        let max_size = 1usize << size_max;

        if self.any_flat(start) {
            let mut current = start;
            while current < max_size && self.any_flat(current * 2) {
                current *= 2;
            }
            return current;
        }
        let mut current = start;
        while current > min_size {
            current /= 2;
            if self.any_flat(current) {
                return current;
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Estimate background statistics with the flat-tile method of
/// `const_background.py::estimate_background`.
///
/// `bad_mask` (same length as `img`, `true` = invalid) excludes any tile that
/// touches an invalid pixel. Returns `None` if no flat tiles exist, so the
/// caller can fall back to the morphological background.
pub fn estimate_background(
    img: &[f32],
    width: usize,
    height: usize,
    rejection_rate: f64,
    alpha: f64,
    bad_mask: Option<&[bool]>,
) -> Option<Background> {
    assert_eq!(img.len(), width * height, "image size mismatch");
    if let Some(m) = bad_mask {
        assert_eq!(m.len(), img.len(), "mask size mismatch");
    }

    let ctx = Ctx {
        img,
        width,
        height,
        bad: bad_mask,
        r1: 1.0 - (1.0 - rejection_rate).powf(0.5),
        r2: 1.0 - (1.0 - rejection_rate).powf(0.25),
    };

    let tile_len = ctx.find_flat_tile_size(6, 4, 7);
    if tile_len == 0 {
        return None;
    }

    // Pool accepted tiles with Chan's parallel update (stable mean / M2).
    let (mut n, mut mean, mut m2) = (0.0f64, 0.0f64, 0.0f64);
    ctx.for_each_flat_tile(tile_len, |t| {
        let nb = t.n as f64;
        let tot = n + nb;
        let delta = t.mean - mean;
        mean += delta * nb / tot;
        m2 += t.m2 + delta * delta * n * nb / tot;
        n = tot;
        false
    });
    if n == 0.0 {
        return None;
    }
    let var = m2 / n; // population variance, like np.nanvar

    // soft_bias = nanmin(img) if negative else 0.
    let image_min = img
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min) as f64;
    let soft_bias = if image_min.is_finite() && image_min < 0.0 {
        image_min
    } else {
        0.0
    };
    let gain = if var != 0.0 { (mean - soft_bias) / var } else { 0.0 };

    Some(Background { mean, var, gain, alpha })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn k2_of(s: &[f64]) -> Option<f64> {
        let n = s.len() as f64;
        let mean = s.iter().sum::<f64>() / n;
        let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
        for &v in s {
            let d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
        }
        dagostino_k2(n, m2 / n, m3 / n, m4 / n)
    }

    fn acc(v: &[f64]) -> [f64; 3] {
        [
            v.len() as f64,
            v.iter().sum(),
            v.iter().map(|x| x * x).sum(),
        ]
    }

    fn noise_image(w: usize, h: usize, mean: f64, sigma: f64, seed: u64) -> Vec<f32> {
        let mut g = Lcg(seed);
        (0..w * h).map(|_| (mean + sigma * g.gauss()) as f32).collect()
    }

    #[test]
    fn gaussian_k2_is_finite_and_small() {
        // Regression: the old excess-kurtosis bug made z2 NaN (silently accepted).
        let mut g = Lcg(1);
        let s: Vec<f64> = (0..4096).map(|_| g.gauss()).collect();
        let k2 = k2_of(&s).expect("k2 must be finite for Gaussian data");
        assert!(k2 < 25.0, "k2={k2}");
        assert!((-0.5 * k2).exp() > 1e-5);
    }

    #[test]
    fn uniform_is_rejected_by_kurtosis() {
        // Skewness ~ 0, so only a correct kurtosis term can reject this.
        let s: Vec<f64> = (0..4096).map(|i| i as f64 / 4096.0).collect();
        let k2 = k2_of(&s).unwrap();
        assert!((-0.5 * k2).exp() < 1e-6, "k2={k2}");
    }

    #[test]
    fn outliers_are_rejected() {
        let mut g = Lcg(2);
        let s: Vec<f64> = (0..4096)
            .map(|i| if i % 20 == 0 { 5000.0 } else { g.gauss() })
            .collect();
        let k2 = k2_of(&s).unwrap();
        assert!((-0.5 * k2).exp() < 1e-6, "k2={k2}");
    }

    #[test]
    fn degenerate_inputs_return_none() {
        assert!(dagostino_k2(10.0, 1.0, 0.0, 3.0).is_none()); // n too small
        assert!(dagostino_k2(100.0, 0.0, 0.0, 0.0).is_none()); // zero variance
        assert!(dagostino_k2(100.0, f64::NAN, 0.0, 3.0).is_none());
    }

    #[test]
    fn ttest_equal_means_pvalue_high() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.1, 2.1, 3.1, 4.1, 5.1];
        assert!(ttest_pvalue(acc(&a), acc(&b)) > 0.5);
    }

    #[test]
    fn ttest_different_means_pvalue_low() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [11.0, 12.0, 13.0, 14.0, 15.0];
        assert!(ttest_pvalue(acc(&a), acc(&b)) < 0.01);
    }

    fn ctx<'a>(img: &'a [f32], w: usize, h: usize, bad: Option<&'a [bool]>) -> Ctx<'a> {
        Ctx { img, width: w, height: h, bad, r1: 1e-4, r2: 1e-4 }
    }

    #[test]
    fn tile_accepts_gaussian_noise() {
        let img = noise_image(16, 16, 100.0, 5.0, 7);
        assert!(ctx(&img, 16, 16, None).analyze_tile(0, 0, 16).is_some());
    }

    #[test]
    fn tile_rejects_constant_and_zero() {
        let c = vec![5.0f32; 256];
        let z = vec![0.0f32; 256];
        assert!(ctx(&c, 16, 16, None).analyze_tile(0, 0, 16).is_none());
        assert!(ctx(&z, 16, 16, None).analyze_tile(0, 0, 16).is_none());
    }

    #[test]
    fn tile_rejects_gradient() {
        let mut t = vec![0.0f32; 256];
        for y in 0..16 {
            for x in 0..16 {
                t[y * 16 + x] = x as f32 * 10.0;
            }
        }
        assert!(ctx(&t, 16, 16, None).analyze_tile(0, 0, 16).is_none());
    }

    #[test]
    fn tile_rejects_masked_pixels() {
        let img = noise_image(16, 16, 100.0, 5.0, 7);
        let mut mask = vec![false; 256];
        mask[37] = true;
        assert!(ctx(&img, 16, 16, Some(&mask)).analyze_tile(0, 0, 16).is_none());
    }

    #[test]
    fn find_flat_tile_size_shrinks_to_working_size() {
        // 96x96 bright block: no clean 64-tile, but clean 32-tiles in the border.
        let (w, h) = (128usize, 128usize);
        let mut img = noise_image(w, h, 100.0, 5.0, 11);
        for y in 0..96 {
            for x in 0..96 {
                img[y * w + x] += 5000.0;
            }
        }
        let c = Ctx { img: &img, width: w, height: h, bad: None,
                      r1: 1.0 - 0.95f64.powf(0.5), r2: 1.0 - 0.95f64.powf(0.25) };
        let size = c.find_flat_tile_size(6, 4, 7);
        assert_eq!(size, 32);
        assert!(c.any_flat(size));
    }

    #[test]
    fn estimate_background_on_synthetic_flat_image() {
        let (w, h) = (128usize, 128usize);
        let mut img = noise_image(w, h, 100.0, 5.0, 12345);
        for y in 50..70 {
            for x in 50..70 {
                img[y * w + x] = 5000.0;
            }
        }
        let bg = estimate_background(&img, w, h, 0.05, 1e-6, None).expect("flat tiles exist");
        assert!((bg.mean - 100.0).abs() < 1.0, "mean={}", bg.mean);
        assert!(bg.var > 20.0 && bg.var < 30.0, "var={}", bg.var);
        assert!(bg.gain > 0.0);
        assert_eq!(bg.alpha, 1e-6);
    }

    #[test]
    fn estimate_background_negative_min_uses_soft_bias() {
        let (w, h) = (64usize, 64usize);
        let img = noise_image(w, h, -5.0, 0.3, 99);
        let bg = estimate_background(&img, w, h, 0.05, 1e-6, None).expect("flat tiles exist");
        assert!((bg.mean + 5.0).abs() < 0.2, "mean={}", bg.mean);
        assert!(bg.var > 0.0 && bg.gain > 0.0);
    }

    #[test]
    fn fully_masked_image_returns_none() {
        let (w, h) = (64usize, 64usize);
        let img = noise_image(w, h, 100.0, 5.0, 5);
        let mask = vec![true; w * h];
        assert!(estimate_background(&img, w, h, 0.05, 1e-6, Some(&mask)).is_none());
    }
}