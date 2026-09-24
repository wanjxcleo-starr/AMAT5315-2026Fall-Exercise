use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct SpectralGrid {
    n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl SpectralGrid {
    pub fn new(n: usize) -> Result<Self, String> {
        if n < 2 || !n.is_power_of_two() {
            return Err("N must be a power of two and at least 2".into());
        }
        let mut planner = FftPlanner::new();
        Ok(Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        })
    }

    pub fn derivative_x(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        self.derivative(values, true)
    }

    pub fn derivative_y(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        self.derivative(values, false)
    }

    pub fn second_derivative_x(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        let mut spectrum = self.forward_real(values)?;
        for iy in 0..self.n {
            for ix in 0..self.n {
                let kx = self.wave_number(ix);
                spectrum[iy * self.n + ix] *= -kx * kx;
            }
        }
        Ok(self.inverse_real(&spectrum))
    }

    pub fn mixed_derivative_xy(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        let mut spectrum = self.forward_real(values)?;
        for iy in 0..self.n {
            let ky = self.wave_number(iy);
            for ix in 0..self.n {
                let index = iy * self.n + ix;
                spectrum[index] = if ix == self.n / 2 || iy == self.n / 2 {
                    Complex64::ZERO
                } else {
                    spectrum[index] * (-self.wave_number(ix) * ky)
                };
            }
        }
        Ok(self.inverse_real(&spectrum))
    }

    pub fn laplacian(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        let mut spectrum = self.forward_real(values)?;
        for iy in 0..self.n {
            let ky = self.wave_number(iy);
            for ix in 0..self.n {
                let kx = self.wave_number(ix);
                spectrum[iy * self.n + ix] *= -(kx * kx + ky * ky);
            }
        }
        Ok(self.inverse_real(&spectrum))
    }

    pub fn divergence(&self, u: &[f64], v: &[f64]) -> Result<Vec<f64>, String> {
        let du_dx = self.derivative_x(u)?;
        let dv_dy = self.derivative_y(v)?;
        Ok(du_dx.iter().zip(dv_dy).map(|(du, dv)| du + dv).collect())
    }

    pub fn poisson_streamfunction(&self, omega: &[f64]) -> Result<Vec<f64>, String> {
        let mut spectrum = self.forward_real(omega)?;
        for iy in 0..self.n {
            let ky = self.wave_number(iy);
            for ix in 0..self.n {
                let kx = self.wave_number(ix);
                let wave_number_squared = kx * kx + ky * ky;
                let index = iy * self.n + ix;
                spectrum[index] = if wave_number_squared == 0.0 {
                    Complex64::ZERO
                } else {
                    spectrum[index] / wave_number_squared
                };
            }
        }
        Ok(self.inverse_real(&spectrum))
    }

    pub fn velocity_from_vorticity(&self, omega: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let spectrum = self.forward_real(omega)?;
        let (u, v) = self.velocity_spectra(&spectrum);
        Ok((self.inverse_real(&u), self.inverse_real(&v)))
    }

    pub(crate) fn forward_real(&self, values: &[f64]) -> Result<Vec<Complex64>, String> {
        if values.len() != self.n * self.n {
            return Err(format!(
                "expected {} grid values, got {}",
                self.n * self.n,
                values.len()
            ));
        }
        let mut data: Vec<_> = values
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        self.transform_2d(&mut data, false);
        Ok(data)
    }

    pub(crate) fn inverse_real(&self, spectrum: &[Complex64]) -> Vec<f64> {
        let mut values = spectrum.to_vec();
        self.transform_2d(&mut values, true);
        let scale = 1.0 / (self.n * self.n) as f64;
        values.iter().map(|value| value.re * scale).collect()
    }

    pub(crate) fn velocity_spectra(&self, omega: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
        let mut u = vec![Complex64::ZERO; omega.len()];
        let mut v = vec![Complex64::ZERO; omega.len()];
        for iy in 0..self.n {
            let ky = self.wave_number(iy);
            for ix in 0..self.n {
                let kx = self.wave_number(ix);
                let index = iy * self.n + ix;
                let k_squared = kx * kx + ky * ky;
                if k_squared != 0.0 {
                    let psi = omega[index] / k_squared;
                    if iy != self.n / 2 {
                        u[index] = psi * Complex64::new(0.0, ky);
                    }
                    if ix != self.n / 2 {
                        v[index] = psi * Complex64::new(0.0, -kx);
                    }
                }
            }
        }
        (u, v)
    }

    pub(crate) fn spectral_derivative(
        &self,
        spectrum: &[Complex64],
        along_x: bool,
    ) -> Vec<Complex64> {
        let mut derivative = spectrum.to_vec();
        for iy in 0..self.n {
            for ix in 0..self.n {
                let axis_index = if along_x { ix } else { iy };
                let index = iy * self.n + ix;
                derivative[index] = if axis_index == self.n / 2 {
                    Complex64::ZERO
                } else {
                    derivative[index] * Complex64::new(0.0, self.wave_number(axis_index))
                };
            }
        }
        derivative
    }

    pub(crate) fn filter_two_thirds(&self, spectrum: &mut [Complex64]) {
        let cutoff = (self.n / 3) as f64;
        for iy in 0..self.n {
            let ky = self.wave_number(iy).abs();
            for ix in 0..self.n {
                let kx = self.wave_number(ix).abs();
                if kx > cutoff || ky > cutoff {
                    spectrum[iy * self.n + ix] = Complex64::ZERO;
                }
            }
        }
    }

    pub(crate) fn wave_number_squared(&self, index: usize) -> f64 {
        let ix = index % self.n;
        let iy = index / self.n;
        self.wave_number(ix).powi(2) + self.wave_number(iy).powi(2)
    }

    fn derivative(&self, values: &[f64], along_x: bool) -> Result<Vec<f64>, String> {
        let spectrum = self.forward_real(values)?;
        let derivative = self.spectral_derivative(&spectrum, along_x);
        Ok(self.inverse_real(&derivative))
    }

    fn transform_2d(&self, data: &mut [Complex64], inverse: bool) {
        let transform = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        for row in data.chunks_exact_mut(self.n) {
            transform.process(row);
        }
        let mut column = vec![Complex64::ZERO; self.n];
        for ix in 0..self.n {
            for iy in 0..self.n {
                column[iy] = data[iy * self.n + ix];
            }
            transform.process(&mut column);
            for iy in 0..self.n {
                data[iy * self.n + ix] = column[iy];
            }
        }
    }

    pub(crate) fn wave_number(&self, index: usize) -> f64 {
        if index <= self.n / 2 {
            index as f64
        } else {
            index as f64 - self.n as f64
        }
    }
}
