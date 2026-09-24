use spectral_fluid::{
    DerivativeMethod, Euler, ExplicitMidpoint, Integrator, LineAdvectionDiffusion, Rk4,
    integrate_trajectory, periodic_gaussian,
};
use std::f64::consts::TAU;

fn cosine_wave(n: usize, wave_number: f64, time: f64, c: f64, nu: f64) -> Vec<f64> {
    (0..n)
        .map(|index| {
            let x = TAU * index as f64 / n as f64;
            (-nu * wave_number.powi(2) * time).exp() * (wave_number * (x - c * time)).cos()
        })
        .collect()
}

#[test]
fn periodic_gaussian_returns_after_one_inviscid_lap() {
    let n = 64;
    let center = TAU / 4.0;
    let sigma = 0.25;
    let initial = periodic_gaussian(n, center, sigma, 1.0, 0.0, 0.0).unwrap();
    let after_one_lap = periodic_gaussian(n, center, sigma, 1.0, 0.0, TAU).unwrap();

    assert!(max_error(&initial, &after_one_lap) < 1.0e-14);
}

#[test]
fn trajectory_uses_the_requested_step_and_lands_on_the_final_time() {
    let n = 32;
    let model = LineAdvectionDiffusion::new(n, 0.0, 0.0, DerivativeMethod::Fourier).unwrap();
    let initial = vec![2.0; n];

    let trajectory = integrate_trajectory(&Rk4, &model, &initial, 0.3, 1.0).unwrap();

    assert_eq!(trajectory.times.len(), 5);
    assert_eq!(*trajectory.times.last().unwrap(), 1.0);
    assert!(
        trajectory
            .times
            .windows(2)
            .all(|pair| pair[1] - pair[0] <= 0.3)
    );
    assert_eq!(trajectory.states.last().unwrap(), &initial);
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

#[test]
fn fourier_rate_matches_the_single_wave_formula() {
    let n = 64;
    let c = 1.0;
    let nu = 0.05;
    let wave_number = 3.0;
    let model = LineAdvectionDiffusion::new(n, c, nu, DerivativeMethod::Fourier).unwrap();
    let state = cosine_wave(n, wave_number, 0.0, c, nu);

    let actual = model.rate(&state).unwrap();
    let expected: Vec<_> = (0..n)
        .map(|index| {
            let x = TAU * index as f64 / n as f64;
            c * wave_number * (wave_number * x).sin()
                - nu * wave_number.powi(2) * (wave_number * x).cos()
        })
        .collect();

    assert!(max_error(&actual, &expected) < 1.0e-12);
}

#[test]
fn fourier_nyquist_mode_diffuses_but_does_not_travel() {
    let n = 32;
    let c = 7.0;
    let nu = 0.125;
    let model = LineAdvectionDiffusion::new(n, c, nu, DerivativeMethod::Fourier).unwrap();
    let state: Vec<_> = (0..n)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let actual = model.rate(&state).unwrap();
    let expected_factor = -nu * (n as f64 / 2.0).powi(2);

    for (rate, value) in actual.iter().zip(&state) {
        assert!((rate - expected_factor * value).abs() < 1.0e-11);
    }
}

#[test]
fn centered_difference_rate_matches_its_discrete_single_wave_formula() {
    let n = 32;
    let c = 0.8;
    let nu = 0.03;
    let wave_number = 3.0;
    let dx = TAU / n as f64;
    let model =
        LineAdvectionDiffusion::new(n, c, nu, DerivativeMethod::CenteredDifference).unwrap();
    let state = cosine_wave(n, wave_number, 0.0, c, nu);

    let actual = model.rate(&state).unwrap();
    let expected: Vec<_> = (0..n)
        .map(|index| {
            let x = dx * index as f64;
            let first_derivative =
                -wave_number.mul_add(dx, 0.0).sin() / dx * (wave_number * x).sin();
            let second_derivative =
                2.0 * ((wave_number * dx).cos() - 1.0) / dx.powi(2) * (wave_number * x).cos();
            -c * first_derivative + nu * second_derivative
        })
        .collect();

    assert!(max_error(&actual, &expected) < 1.0e-12);
}

#[test]
fn every_integrator_advances_a_single_wave_toward_the_exact_solution() {
    let n = 64;
    let c = 0.7;
    let nu = 0.05;
    let wave_number = 3.0;
    let dt = 0.001;
    let steps = 10;
    let model = LineAdvectionDiffusion::new(n, c, nu, DerivativeMethod::Fourier).unwrap();
    let initial = cosine_wave(n, wave_number, 0.0, c, nu);
    let exact = cosine_wave(n, wave_number, dt * steps as f64, c, nu);
    let cases: [(&str, &dyn Integrator<f64>, f64); 3] = [
        ("Euler", &Euler, 3.0e-5),
        ("explicit midpoint", &ExplicitMidpoint, 3.0e-8),
        ("RK4", &Rk4, 1.0e-12),
    ];

    for (name, integrator, tolerance) in cases {
        let mut state = initial.clone();
        for _ in 0..steps {
            let mut rate = |values: &[f64]| model.rate(values);
            state = integrator.step(&state, dt, &mut rate).unwrap();
        }
        let error = max_error(&state, &exact);
        assert!(error < tolerance, "{name} error was {error:e}");
    }
}
