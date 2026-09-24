use spectral_fluid::CenteredDifferenceGrid;
use std::f64::consts::TAU;

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

fn maximum_errors(n: usize) -> [f64; 4] {
    let grid = CenteredDifferenceGrid::new(n).unwrap();
    let values = comparison_wave(n);
    let derivatives = [
        grid.derivative_x(&values).unwrap(),
        grid.second_derivative_x(&values).unwrap(),
        grid.mixed_derivative_xy(&values).unwrap(),
        grid.laplacian(&values).unwrap(),
    ];
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

#[test]
fn centered_difference_errors_fall_by_four_when_the_grid_spacing_halves() {
    let coarse = maximum_errors(32);
    let fine = maximum_errors(64);
    let answer_key_coarse = [0.17050, 0.25724, 0.48534, 0.30838];
    let answer_key_fine = [0.04318, 0.06487, 0.12429, 0.07771];

    for index in 0..4 {
        assert!((coarse[index] - answer_key_coarse[index]).abs() < 1.0e-5);
        assert!((fine[index] - answer_key_fine[index]).abs() < 1.0e-5);
        let ratio = coarse[index] / fine[index];
        assert!(
            (3.8..4.1).contains(&ratio),
            "derivative {index} error ratio was {ratio}"
        );
    }
}
