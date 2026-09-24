use serde::Serialize;
use spectral_fluid::{CenteredDifferenceGrid, SpectralGrid};
use std::error::Error;
use std::f64::consts::TAU;

#[derive(Serialize)]
struct StudyDocument {
    derivatives: Vec<DerivativeRow>,
}

#[derive(Serialize)]
struct DerivativeRow {
    derivative: &'static str,
    finite_difference_n32: f64,
    finite_difference_n64: f64,
    fourier_n32: f64,
}

fn comparison_wave(n: usize) -> Vec<f64> {
    (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                let x = TAU * ix as f64 / n as f64;
                let y = TAU * iy as f64 / n as f64;
                (3.0 * x).sin() * (2.0 * y).cos()
            })
        })
        .collect()
}

fn maximum_errors(n: usize, derivatives: &[Vec<f64>; 4]) -> [f64; 4] {
    std::array::from_fn(|derivative_index| {
        derivatives[derivative_index]
            .iter()
            .enumerate()
            .map(|(index, &actual)| {
                let x = TAU * (index % n) as f64 / n as f64;
                let y = TAU * (index / n) as f64 / n as f64;
                let g = (3.0 * x).sin() * (2.0 * y).cos();
                let expected = match derivative_index {
                    0 => 3.0 * (3.0 * x).cos() * (2.0 * y).cos(),
                    1 => -9.0 * g,
                    2 => -6.0 * (3.0 * x).cos() * (2.0 * y).sin(),
                    3 => -13.0 * g,
                    _ => unreachable!(),
                };
                (actual - expected).abs()
            })
            .fold(0.0, f64::max)
    })
}

fn fourier_errors(n: usize) -> Result<[f64; 4], String> {
    let grid = SpectralGrid::new(n)?;
    let values = comparison_wave(n);
    Ok(maximum_errors(
        n,
        &[
            grid.derivative_x(&values)?,
            grid.second_derivative_x(&values)?,
            grid.mixed_derivative_xy(&values)?,
            grid.laplacian(&values)?,
        ],
    ))
}

fn centered_errors(n: usize) -> Result<[f64; 4], String> {
    let grid = CenteredDifferenceGrid::new(n)?;
    let values = comparison_wave(n);
    Ok(maximum_errors(
        n,
        &[
            grid.derivative_x(&values)?,
            grid.second_derivative_x(&values)?,
            grid.mixed_derivative_xy(&values)?,
            grid.laplacian(&values)?,
        ],
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let finite_difference_n32 = centered_errors(32)?;
    let finite_difference_n64 = centered_errors(64)?;
    let fourier_n32 = fourier_errors(32)?;
    let names = ["dx", "dxx", "dxdy", "laplacian"];
    let derivatives = (0..4)
        .map(|index| DerivativeRow {
            derivative: names[index],
            finite_difference_n32: finite_difference_n32[index],
            finite_difference_n64: finite_difference_n64[index],
            fourier_n32: fourier_n32[index],
        })
        .collect();
    serde_json::to_writer(std::io::stdout(), &StudyDocument { derivatives })?;
    Ok(())
}
