use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::TAU;
use std::sync::Arc;

use crate::Integrator;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeMethod {
    Fourier,
    CenteredDifference,
}

pub struct LineAdvectionDiffusion {
    n: usize,
    c: f64,
    nu: f64,
    derivative_method: DerivativeMethod,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

pub struct LineTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
}

impl LineAdvectionDiffusion {
    pub fn new(
        n: usize,
        c: f64,
        nu: f64,
        derivative_method: DerivativeMethod,
    ) -> Result<Self, String> {
        if n < 2 || !n.is_power_of_two() {
            return Err("n must be a power of two and at least 2".into());
        }
        if !c.is_finite() {
            return Err("c must be finite".into());
        }
        if !nu.is_finite() || nu < 0.0 {
            return Err("nu must be finite and non-negative".into());
        }
        let mut planner = FftPlanner::new();
        Ok(Self {
            n,
            c,
            nu,
            derivative_method,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        })
    }

    pub fn rate(&self, state: &[f64]) -> Result<Vec<f64>, String> {
        if state.len() != self.n {
            return Err(format!(
                "expected {} line values, got {}",
                self.n,
                state.len()
            ));
        }
        match self.derivative_method {
            DerivativeMethod::Fourier => Ok(self.fourier_rate(state)),
            DerivativeMethod::CenteredDifference => Ok(self.centered_difference_rate(state)),
        }
    }

    pub fn coordinates(&self) -> Vec<f64> {
        (0..self.n)
            .map(|index| TAU * index as f64 / self.n as f64)
            .collect()
    }

    pub fn n(&self) -> usize {
        self.n
    }

    fn fourier_rate(&self, state: &[f64]) -> Vec<f64> {
        let mut spectrum: Vec<_> = state
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        self.forward.process(&mut spectrum);
        for (index, value) in spectrum.iter_mut().enumerate() {
            let wave_number = self.wave_number(index);
            let advection = if index == self.n / 2 {
                Complex64::ZERO
            } else {
                Complex64::new(0.0, -self.c * wave_number)
            };
            let diffusion = -self.nu * wave_number * wave_number;
            *value *= advection + diffusion;
        }
        self.inverse.process(&mut spectrum);
        let scale = 1.0 / self.n as f64;
        spectrum.iter().map(|value| value.re * scale).collect()
    }

    fn centered_difference_rate(&self, state: &[f64]) -> Vec<f64> {
        let dx = TAU / self.n as f64;
        (0..self.n)
            .map(|index| {
                let previous = state[(index + self.n - 1) % self.n];
                let current = state[index];
                let next = state[(index + 1) % self.n];
                let first_derivative = (next - previous) / (2.0 * dx);
                let second_derivative = (next - 2.0 * current + previous) / (dx * dx);
                -self.c * first_derivative + self.nu * second_derivative
            })
            .collect()
    }

    fn wave_number(&self, index: usize) -> f64 {
        if index <= self.n / 2 {
            index as f64
        } else {
            index as f64 - self.n as f64
        }
    }
}

pub fn periodic_gaussian(
    n: usize,
    center: f64,
    sigma: f64,
    c: f64,
    nu: f64,
    time: f64,
) -> Result<Vec<f64>, String> {
    if n < 2 {
        return Err("n must be at least 2".into());
    }
    if !center.is_finite() || !c.is_finite() {
        return Err("center and c must be finite".into());
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err("sigma must be finite and greater than zero".into());
    }
    if !nu.is_finite() || nu < 0.0 || !time.is_finite() || time < 0.0 {
        return Err("nu and time must be finite and non-negative".into());
    }

    let variance = sigma * sigma + 2.0 * nu * time;
    let amplitude = sigma / variance.sqrt();
    let advected_center = center + c * time;
    Ok((0..n)
        .map(|index| {
            let x = TAU * index as f64 / n as f64;
            let nearest_image = ((advected_center - x) / TAU).round() as i32;
            let image_sum: f64 = (nearest_image - 3..=nearest_image + 3)
                .map(|image| {
                    let displacement = x - advected_center + TAU * image as f64;
                    (-displacement * displacement / (2.0 * variance)).exp()
                })
                .sum();
            amplitude * image_sum
        })
        .collect())
}

pub fn integrate_trajectory(
    integrator: &dyn Integrator<f64>,
    model: &LineAdvectionDiffusion,
    initial: &[f64],
    dt: f64,
    t_end: f64,
) -> Result<LineTrajectory, String> {
    if initial.len() != model.n() {
        return Err(format!(
            "expected {} initial values, got {}",
            model.n(),
            initial.len()
        ));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and greater than zero".into());
    }
    if !t_end.is_finite() || t_end < 0.0 {
        return Err("t_end must be finite and non-negative".into());
    }

    let mut time = 0.0;
    let mut state = initial.to_vec();
    let mut times = vec![time];
    let mut states = vec![state.clone()];
    while time < t_end {
        let step = dt.min(t_end - time);
        let mut rate = |values: &[f64]| model.rate(values);
        state = integrator.step(&state, step, &mut rate)?;
        time += step;
        if t_end - time <= 16.0 * f64::EPSILON * t_end.max(1.0) {
            time = t_end;
        }
        times.push(time);
        states.push(state.clone());
    }
    Ok(LineTrajectory { times, states })
}
