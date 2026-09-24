use rustfft::num_complex::Complex64;
use serde::Serialize;
use spectral_fluid::{
    DerivativeMethod, EqualWeightRk4, Euler, ExplicitMidpoint, Integrator, LineAdvectionDiffusion,
    Rk4, integrate_trajectory, periodic_gaussian,
};
use std::collections::BTreeMap;
use std::error::Error;
use std::f64::consts::{FRAC_PI_2, TAU};

#[derive(Serialize)]
struct StudyDocument {
    stability: StabilityStudy,
    accuracy: AccuracyStudy,
}

#[derive(Serialize)]
struct StabilityStudy {
    n: usize,
    nu: f64,
    c: f64,
    stable_dt: f64,
    unstable_dt: f64,
    real_axis: Vec<f64>,
    imaginary_axis: Vec<f64>,
    rk4_growth: Vec<Vec<f64>>,
    modes: Vec<ModeSet>,
    stable_pulse: PulseTrajectory,
    unstable_pulse: PulseTrajectory,
}

#[derive(Serialize)]
struct ModeSet {
    dt: f64,
    points: Vec<[f64; 2]>,
}

#[derive(Serialize)]
struct PulseTrajectory {
    dt: f64,
    x: Vec<f64>,
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct AccuracyStudy {
    profile_n: usize,
    x: Vec<f64>,
    exact_profile: Vec<f64>,
    profiles: Vec<Profile>,
    time_steps: Vec<f64>,
    convergence_errors: BTreeMap<String, Vec<f64>>,
    slopes: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct Profile {
    label: String,
    max_error: f64,
    values: Vec<f64>,
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| start + (end - start) * index as f64 / (count - 1) as f64)
        .collect()
}

fn measured_rk4_growth(z: Complex64) -> Result<f64, String> {
    let initial = [Complex64::new(1.0, 0.0)];
    let mut rate = |state: &[Complex64]| Ok(vec![z * state[0]]);
    Ok(Rk4.step(&initial, 1.0, &mut rate)?[0].norm())
}

fn line_modes(n: usize, nu: f64, c: f64, dt: f64) -> ModeSet {
    let points = (0..n)
        .map(|index| {
            let wave_number = if index <= n / 2 {
                index as f64
            } else {
                index as f64 - n as f64
            };
            let imaginary = if index == n / 2 {
                0.0
            } else {
                -c * wave_number
            };
            [-nu * wave_number * wave_number * dt, imaginary * dt]
        })
        .collect();
    ModeSet { dt, points }
}

fn pulse_trajectory(dt: f64) -> Result<PulseTrajectory, String> {
    let n = 64;
    let c = 1.0;
    let nu = 0.05;
    let model = LineAdvectionDiffusion::new(n, c, nu, DerivativeMethod::Fourier)?;
    let initial = periodic_gaussian(n, FRAC_PI_2, 0.35, c, nu, 0.0)?;
    let trajectory = integrate_trajectory(&Rk4, &model, &initial, dt, 6.0)?;
    Ok(PulseTrajectory {
        dt,
        x: model.coordinates(),
        times: trajectory.times,
        states: trajectory.states,
    })
}

fn stability_study() -> Result<StabilityStudy, String> {
    let n = 64;
    let nu = 0.05;
    let c = 1.0;
    let stable_dt = 0.045;
    let unstable_dt = 0.056;
    let real_axis = linspace(-4.0, 2.0, 241);
    let imaginary_axis = linspace(-4.0, 4.0, 321);
    let rk4_growth = imaginary_axis
        .iter()
        .map(|&imaginary| {
            real_axis
                .iter()
                .map(|&real| measured_rk4_growth(Complex64::new(real, imaginary)))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StabilityStudy {
        n,
        nu,
        c,
        stable_dt,
        unstable_dt,
        real_axis,
        imaginary_axis,
        rk4_growth,
        modes: vec![
            line_modes(n, nu, c, stable_dt),
            line_modes(n, nu, c, unstable_dt),
        ],
        stable_pulse: pulse_trajectory(stable_dt)?,
        unstable_pulse: pulse_trajectory(unstable_dt)?,
    })
}

fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn final_profile(
    integrator: &dyn Integrator<f64>,
    derivative_method: DerivativeMethod,
    sigma: f64,
    nu: f64,
    dt: f64,
    t_end: f64,
) -> Result<Vec<f64>, String> {
    let n = 64;
    let c = 1.0;
    let model = LineAdvectionDiffusion::new(n, c, nu, derivative_method)?;
    let initial = periodic_gaussian(n, FRAC_PI_2, sigma, c, nu, 0.0)?;
    let trajectory = integrate_trajectory(integrator, &model, &initial, dt, t_end)?;
    Ok(trajectory.states.last().cloned().unwrap())
}

fn fit_log_slope(time_steps: &[f64], errors: &[f64]) -> f64 {
    let log_steps: Vec<_> = time_steps.iter().map(|value| value.ln()).collect();
    let log_errors: Vec<_> = errors.iter().map(|value| value.ln()).collect();
    let mean_step = log_steps.iter().sum::<f64>() / log_steps.len() as f64;
    let mean_error = log_errors.iter().sum::<f64>() / log_errors.len() as f64;
    let numerator: f64 = log_steps
        .iter()
        .zip(&log_errors)
        .map(|(step, error)| (step - mean_step) * (error - mean_error))
        .sum();
    let denominator: f64 = log_steps
        .iter()
        .map(|step| (step - mean_step).powi(2))
        .sum();
    numerator / denominator
}

fn accuracy_study() -> Result<AccuracyStudy, String> {
    let profile_n = 64;
    let profile_sigma = 0.25;
    let profile_nu = 0.002;
    let profile_time = TAU;
    let exact_profile = periodic_gaussian(
        profile_n,
        FRAC_PI_2,
        profile_sigma,
        1.0,
        profile_nu,
        profile_time,
    )?;
    let profile_cases: [(&str, &dyn Integrator<f64>, DerivativeMethod, f64); 3] = [
        (
            "RK4, Fourier, dt=0.02",
            &Rk4,
            DerivativeMethod::Fourier,
            0.02,
        ),
        (
            "RK4, centered difference, dt=0.02",
            &Rk4,
            DerivativeMethod::CenteredDifference,
            0.02,
        ),
        (
            "Euler, Fourier, dt=0.005",
            &Euler,
            DerivativeMethod::Fourier,
            0.005,
        ),
    ];
    let profiles = profile_cases
        .iter()
        .map(|(label, integrator, derivative, dt)| {
            let values = final_profile(
                *integrator,
                *derivative,
                profile_sigma,
                profile_nu,
                *dt,
                profile_time,
            )?;
            Ok(Profile {
                label: (*label).into(),
                max_error: max_error(&values, &exact_profile),
                values,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let time_steps = vec![0.02, 0.01, 0.005, 0.0025];
    let convergence_sigma = 0.35;
    let convergence_nu = 0.05;
    let convergence_time = 1.0;
    let convergence_exact = periodic_gaussian(
        profile_n,
        FRAC_PI_2,
        convergence_sigma,
        1.0,
        convergence_nu,
        convergence_time,
    )?;
    let methods: [(&str, &dyn Integrator<f64>); 4] = [
        ("euler", &Euler),
        ("midpoint", &ExplicitMidpoint),
        ("rk4", &Rk4),
        ("equal_weight_rk4", &EqualWeightRk4),
    ];
    let mut convergence_errors = BTreeMap::new();
    let mut slopes = BTreeMap::new();
    for (name, integrator) in methods {
        let errors = time_steps
            .iter()
            .map(|&dt| {
                let values = final_profile(
                    integrator,
                    DerivativeMethod::Fourier,
                    convergence_sigma,
                    convergence_nu,
                    dt,
                    convergence_time,
                )?;
                Ok(max_error(&values, &convergence_exact))
            })
            .collect::<Result<Vec<_>, String>>()?;
        slopes.insert(name.into(), fit_log_slope(&time_steps, &errors));
        convergence_errors.insert(name.into(), errors);
    }

    Ok(AccuracyStudy {
        profile_n,
        x: (0..profile_n)
            .map(|index| TAU * index as f64 / profile_n as f64)
            .collect(),
        exact_profile,
        profiles,
        time_steps,
        convergence_errors,
        slopes,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let document = StudyDocument {
        stability: stability_study()?,
        accuracy: accuracy_study()?,
    };
    serde_json::to_writer(std::io::stdout(), &document)?;
    Ok(())
}
