use crate::SpectralGrid;
use std::f64::consts::TAU;

pub fn add_vorticity_ripple(
    n: usize,
    u: &[f64],
    v: &[f64],
    coefficient: f64,
    kx: i32,
    ky: i32,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !coefficient.is_finite() {
        return Err("ripple coefficient must be finite".into());
    }
    if kx.unsigned_abs() as usize >= n / 2 || ky.unsigned_abs() as usize >= n / 2 {
        return Err("ripple wavenumbers must be below the Nyquist frequency".into());
    }
    let grid = SpectralGrid::new(n)?;
    if u.len() != n * n || v.len() != n * n {
        return Err(format!("u and v must each contain {} values", n * n));
    }
    let scale = u
        .iter()
        .chain(v)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let dv_dx = grid.derivative_x(v)?;
    let du_dy = grid.derivative_y(u)?;
    let mut omega: Vec<_> = dv_dx.iter().zip(du_dy).map(|(dv, du)| dv - du).collect();
    for (index, value) in omega.iter_mut().enumerate() {
        let x = TAU * (index % n) as f64 / n as f64;
        let y = TAU * (index / n) as f64 / n as f64;
        *value += coefficient * scale * (kx as f64 * x).cos() * (ky as f64 * y).cos();
    }
    grid.velocity_from_vorticity(&omega)
}
