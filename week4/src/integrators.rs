use std::ops::{Add, Mul};

pub type RateFunction<'a, T> = dyn FnMut(&[T]) -> Result<Vec<T>, String> + 'a;

pub trait Integrator<T> {
    fn step(&self, state: &[T], dt: f64, rate: &mut RateFunction<'_, T>) -> Result<Vec<T>, String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Euler;

#[derive(Clone, Copy, Debug, Default)]
pub struct ExplicitMidpoint;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rk4;

/// Four RK4 stages combined with deliberately equal weights.
///
/// This is retained only as the Part 1 order-comparison method; unlike
/// [`Rk4`], it is second-order accurate.
#[derive(Clone, Copy, Debug, Default)]
pub struct EqualWeightRk4;

fn validate_dt(dt: f64) -> Result<(), String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and greater than zero".into());
    }
    Ok(())
}

fn evaluate_rate<T>(state: &[T], rate: &mut RateFunction<'_, T>) -> Result<Vec<T>, String> {
    let values = rate(state)?;
    if values.len() != state.len() {
        return Err(format!(
            "rate returned {} values for a state of length {}",
            values.len(),
            state.len()
        ));
    }
    Ok(values)
}

fn combine<T>(base: &[T], terms: &[(f64, &[T])]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    base.iter()
        .enumerate()
        .map(|(index, &base_value)| {
            terms.iter().fold(base_value, |value, (factor, term)| {
                value + term[index] * *factor
            })
        })
        .collect()
}

impl<T> Integrator<T> for Euler
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    fn step(&self, state: &[T], dt: f64, rate: &mut RateFunction<'_, T>) -> Result<Vec<T>, String> {
        validate_dt(dt)?;
        let k1 = evaluate_rate(state, rate)?;
        Ok(combine(state, &[(dt, &k1)]))
    }
}

impl<T> Integrator<T> for ExplicitMidpoint
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    fn step(&self, state: &[T], dt: f64, rate: &mut RateFunction<'_, T>) -> Result<Vec<T>, String> {
        validate_dt(dt)?;
        let k1 = evaluate_rate(state, rate)?;
        let midpoint = combine(state, &[(0.5 * dt, &k1)]);
        let k2 = evaluate_rate(&midpoint, rate)?;
        Ok(combine(state, &[(dt, &k2)]))
    }
}

impl<T> Integrator<T> for Rk4
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    fn step(&self, state: &[T], dt: f64, rate: &mut RateFunction<'_, T>) -> Result<Vec<T>, String> {
        validate_dt(dt)?;
        let k1 = evaluate_rate(state, rate)?;
        let stage2 = combine(state, &[(0.5 * dt, &k1)]);
        let k2 = evaluate_rate(&stage2, rate)?;
        let stage3 = combine(state, &[(0.5 * dt, &k2)]);
        let k3 = evaluate_rate(&stage3, rate)?;
        let stage4 = combine(state, &[(dt, &k3)]);
        let k4 = evaluate_rate(&stage4, rate)?;
        Ok(combine(
            state,
            &[
                (dt / 6.0, &k1),
                (dt / 3.0, &k2),
                (dt / 3.0, &k3),
                (dt / 6.0, &k4),
            ],
        ))
    }
}

impl<T> Integrator<T> for EqualWeightRk4
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T>,
{
    fn step(&self, state: &[T], dt: f64, rate: &mut RateFunction<'_, T>) -> Result<Vec<T>, String> {
        validate_dt(dt)?;
        let k1 = evaluate_rate(state, rate)?;
        let stage2 = combine(state, &[(0.5 * dt, &k1)]);
        let k2 = evaluate_rate(&stage2, rate)?;
        let stage3 = combine(state, &[(0.5 * dt, &k2)]);
        let k3 = evaluate_rate(&stage3, rate)?;
        let stage4 = combine(state, &[(dt, &k3)]);
        let k4 = evaluate_rate(&stage4, rate)?;
        Ok(combine(
            state,
            &[
                (dt / 4.0, &k1),
                (dt / 4.0, &k2),
                (dt / 4.0, &k3),
                (dt / 4.0, &k4),
            ],
        ))
    }
}
