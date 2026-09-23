use spectral_fluid::{FlowSolver, Method, SpectralGrid, kinetic_energy};
use std::f64::consts::TAU;

#[test]
fn spectral_derivative_of_sin_3x_is_3cos_3x() {
    let n = 32;
    let grid = SpectralGrid::new(n).unwrap();
    let values: Vec<f64> = (0..n)
        .flat_map(|_| (0..n).map(|ix| (3.0 * TAU * ix as f64 / n as f64).sin()))
        .collect();

    let derivative = grid.derivative_x(&values).unwrap();
    let max_error = derivative
        .iter()
        .enumerate()
        .map(|(index, &actual)| {
            let ix = index % n;
            let expected = 3.0 * (3.0 * TAU * ix as f64 / n as f64).cos();
            (actual - expected).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(max_error < 1.0e-12, "max error was {max_error:e}");
}

#[test]
fn poisson_solution_of_two_cos_x_cos_y_is_cos_x_cos_y() {
    let n = 32;
    let grid = SpectralGrid::new(n).unwrap();
    let omega: Vec<f64> = (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                let x = TAU * ix as f64 / n as f64;
                let y = TAU * iy as f64 / n as f64;
                2.0 * x.cos() * y.cos()
            })
        })
        .collect();

    let psi = grid.poisson_streamfunction(&omega).unwrap();
    let max_error = psi
        .iter()
        .enumerate()
        .map(|(index, &actual)| {
            let ix = index % n;
            let iy = index / n;
            let expected = (TAU * ix as f64 / n as f64).cos() * (TAU * iy as f64 / n as f64).cos();
            (actual - expected).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(max_error < 1.0e-12, "max error was {max_error:e}");
}

#[test]
fn rk4_taylor_green_energy_and_divergence_match_exact_solution() {
    let n = 64;
    let viscosity = 0.1;
    let dt = 0.01;
    let omega: Vec<f64> = (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                let x = TAU * ix as f64 / n as f64;
                let y = TAU * iy as f64 / n as f64;
                -2.0 * x.cos() * y.cos()
            })
        })
        .collect();
    let mut solver = FlowSolver::new(n, viscosity, &omega).unwrap();

    for _ in 0..100 {
        solver.step(Method::Rk4, dt).unwrap();
    }

    let fields = solver.fields();
    let energy = kinetic_energy(&fields.u, &fields.v).unwrap();
    let expected = 0.25 * (-0.4_f64).exp();
    assert!(
        (energy - expected).abs() < 1.0e-6,
        "energy {energy:.15e}, expected {expected:.15e}"
    );

    let grid = SpectralGrid::new(n).unwrap();
    let divergence = grid.divergence(&fields.u, &fields.v).unwrap();
    let max_divergence = divergence
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    assert!(
        max_divergence < 1.0e-10,
        "maximum divergence was {max_divergence:e}"
    );
}

#[test]
fn first_derivative_discards_the_even_grid_nyquist_line() {
    let n = 16;
    let grid = SpectralGrid::new(n).unwrap();
    let checkerboard: Vec<f64> = (0..n)
        .flat_map(|_| (0..n).map(|ix| if ix % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();

    let derivative = grid.derivative_x(&checkerboard).unwrap();
    assert!(derivative.iter().all(|value| value.abs() < 1.0e-12));
}

#[test]
fn poisson_zero_mode_fixes_psi_mean_without_creating_velocity_mean() {
    let n = 16;
    let grid = SpectralGrid::new(n).unwrap();
    let omega: Vec<f64> = (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                7.0 + (TAU * ix as f64 / n as f64).cos() * (TAU * iy as f64 / n as f64).cos()
            })
        })
        .collect();

    let psi = grid.poisson_streamfunction(&omega).unwrap();
    let (u, v) = grid.velocity_from_vorticity(&omega).unwrap();
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    assert!(mean(&psi).abs() < 1.0e-12);
    assert!(mean(&u).abs() < 1.0e-12);
    assert!(mean(&v).abs() < 1.0e-12);
}

#[test]
fn euler_and_explicit_midpoint_rk2_use_their_expected_diffusion_polynomials() {
    let n = 16;
    let viscosity = 0.1;
    let dt = 0.1;
    let omega: Vec<f64> = (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                let x = TAU * ix as f64 / n as f64;
                let y = TAU * iy as f64 / n as f64;
                -2.0 * x.cos() * y.cos()
            })
        })
        .collect();
    let decay_argument = 2.0 * viscosity * dt;

    for (method, expected_factor) in [
        (Method::Euler, 1.0 - decay_argument),
        (
            Method::MidpointRk2,
            1.0 - decay_argument + 0.5 * decay_argument * decay_argument,
        ),
    ] {
        let mut solver = FlowSolver::new(n, viscosity, &omega).unwrap();
        solver.step(method, dt).unwrap();
        let actual = solver.fields().omega[0];
        let expected = -2.0 * expected_factor;
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{method:?}: got {actual:.15e}, expected {expected:.15e}"
        );
    }
}

#[test]
fn initial_vorticity_is_filtered_by_the_two_thirds_mask() {
    let n = 16;
    let unresolved_by_mask: Vec<f64> = (0..n)
        .flat_map(|_| (0..n).map(|ix| (6.0 * TAU * ix as f64 / n as f64).cos()))
        .collect();

    let solver = FlowSolver::new(n, 0.0, &unresolved_by_mask).unwrap();
    assert!(
        solver
            .fields()
            .omega
            .iter()
            .all(|value| value.abs() < 1.0e-12)
    );
}

#[test]
fn velocity_round_trip_preserves_taylor_green_orientation() {
    let n = 16;
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for iy in 0..n {
        let y = TAU * iy as f64 / n as f64;
        for ix in 0..n {
            let x = TAU * ix as f64 / n as f64;
            u.push(x.cos() * y.sin());
            v.push(-x.sin() * y.cos());
        }
    }

    let fields = FlowSolver::from_velocity(n, 0.0, &u, &v).unwrap().fields();
    let max_error = fields
        .u
        .iter()
        .zip(&u)
        .chain(fields.v.iter().zip(&v))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-12, "maximum error was {max_error:e}");
}

#[test]
fn euler_step_uses_the_vorticity_advection_and_diffusion_rhs() {
    let n = 16;
    let viscosity = 0.1;
    let dt = 0.01;
    let omega: Vec<f64> = (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                let x = TAU * ix as f64 / n as f64;
                let y = TAU * iy as f64 / n as f64;
                x.cos() + 4.0 * (2.0 * y).cos()
            })
        })
        .collect();
    let mut solver = FlowSolver::new(n, viscosity, &omega).unwrap();

    solver.step(Method::Euler, dt).unwrap();

    let actual = solver.fields().omega;
    let max_error = actual
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let x = TAU * (index % n) as f64 / n as f64;
            let y = TAU * (index / n) as f64 / n as f64;
            let initial = x.cos() + 4.0 * (2.0 * y).cos();
            let advection = 6.0 * x.sin() * (2.0 * y).sin();
            let diffusion = viscosity * (-x.cos() - 16.0 * (2.0 * y).cos());
            (value - (initial + dt * (advection + diffusion))).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-11, "maximum error was {max_error:e}");
}

#[test]
fn nonlinear_term_is_filtered_by_the_two_thirds_mask() {
    let n = 16;
    let dt = 0.01;
    let omega: Vec<f64> = (0..n)
        .flat_map(|iy| {
            (0..n).map(move |ix| {
                let x = TAU * ix as f64 / n as f64;
                let y = TAU * iy as f64 / n as f64;
                16.0 * (4.0 * x).cos() + 10.0 * (3.0 * x + y).cos()
            })
        })
        .collect();
    let mut solver = FlowSolver::new(n, 0.0, &omega).unwrap();

    solver.step(Method::Euler, dt).unwrap();

    let actual = solver.fields().omega;
    let max_error = actual
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let x = TAU * (index % n) as f64 / n as f64;
            let y = TAU * (index / n) as f64 / n as f64;
            let expected =
                16.0 * (4.0 * x).cos() + 10.0 * (3.0 * x + y).cos() - 12.0 * dt * (x - y).cos();
            (value - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1.0e-10, "maximum error was {max_error:e}");
}
