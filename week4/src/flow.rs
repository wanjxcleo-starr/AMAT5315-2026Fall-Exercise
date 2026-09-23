use crate::SpectralGrid;
use rustfft::num_complex::Complex64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Euler,
    MidpointRk2,
    Rk4,
}

pub struct FlowFields {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub omega: Vec<f64>,
}

pub struct FlowSolver {
    grid: SpectralGrid,
    viscosity: f64,
    omega: Vec<Complex64>,
}

impl FlowSolver {
    pub fn new(n: usize, viscosity: f64, omega: &[f64]) -> Result<Self, String> {
        if !viscosity.is_finite() || viscosity < 0.0 {
            return Err("viscosity must be finite and non-negative".into());
        }
        let grid = SpectralGrid::new(n)?;
        let mut omega = grid.forward_real(omega)?;
        grid.filter_two_thirds(&mut omega);
        omega[0] = Complex64::ZERO;
        Ok(Self {
            grid,
            viscosity,
            omega,
        })
    }

    pub fn from_velocity(n: usize, viscosity: f64, u: &[f64], v: &[f64]) -> Result<Self, String> {
        let grid = SpectralGrid::new(n)?;
        let dv_dx = grid.derivative_x(v)?;
        let du_dy = grid.derivative_y(u)?;
        let omega: Vec<f64> = dv_dx.iter().zip(du_dy).map(|(dv, du)| dv - du).collect();
        Self::new(n, viscosity, &omega)
    }

    pub fn step(&mut self, method: Method, dt: f64) -> Result<(), String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and greater than zero".into());
        }
        let initial = self.omega.clone();
        self.omega = match method {
            Method::Euler => {
                let k1 = self.rhs(&initial)?;
                self.combination(&initial, &[(dt, &k1)])
            }
            Method::MidpointRk2 => {
                let k1 = self.rhs(&initial)?;
                let midpoint = self.combination(&initial, &[(0.5 * dt, &k1)]);
                let k2 = self.rhs(&midpoint)?;
                self.combination(&initial, &[(dt, &k2)])
            }
            Method::Rk4 => {
                let k1 = self.rhs(&initial)?;
                let stage2 = self.combination(&initial, &[(0.5 * dt, &k1)]);
                let k2 = self.rhs(&stage2)?;
                let stage3 = self.combination(&initial, &[(0.5 * dt, &k2)]);
                let k3 = self.rhs(&stage3)?;
                let stage4 = self.combination(&initial, &[(dt, &k3)]);
                let k4 = self.rhs(&stage4)?;
                self.combination(
                    &initial,
                    &[
                        (dt / 6.0, &k1),
                        (dt / 3.0, &k2),
                        (dt / 3.0, &k3),
                        (dt / 6.0, &k4),
                    ],
                )
            }
        };
        Ok(())
    }

    pub fn fields(&self) -> FlowFields {
        let (u_spectrum, v_spectrum) = self.grid.velocity_spectra(&self.omega);
        FlowFields {
            u: self.grid.inverse_real(&u_spectrum),
            v: self.grid.inverse_real(&v_spectrum),
            omega: self.grid.inverse_real(&self.omega),
        }
    }

    fn rhs(&self, stage: &[Complex64]) -> Result<Vec<Complex64>, String> {
        let mut omega = stage.to_vec();
        self.grid.filter_two_thirds(&mut omega);
        omega[0] = Complex64::ZERO;

        let (u_spectrum, v_spectrum) = self.grid.velocity_spectra(&omega);
        let omega_x_spectrum = self.grid.spectral_derivative(&omega, true);
        let omega_y_spectrum = self.grid.spectral_derivative(&omega, false);
        let u = self.grid.inverse_real(&u_spectrum);
        let v = self.grid.inverse_real(&v_spectrum);
        let omega_x = self.grid.inverse_real(&omega_x_spectrum);
        let omega_y = self.grid.inverse_real(&omega_y_spectrum);
        let nonlinear: Vec<f64> = (0..omega.len())
            .map(|index| -(u[index] * omega_x[index] + v[index] * omega_y[index]))
            .collect();
        let mut rhs = self.grid.forward_real(&nonlinear)?;
        self.grid.filter_two_thirds(&mut rhs);
        for (index, value) in rhs.iter_mut().enumerate() {
            *value -= self.viscosity * self.grid.wave_number_squared(index) * omega[index];
        }
        rhs[0] = Complex64::ZERO;
        Ok(rhs)
    }

    fn combination(&self, base: &[Complex64], terms: &[(f64, &[Complex64])]) -> Vec<Complex64> {
        let mut result = base.to_vec();
        for &(factor, values) in terms {
            for (target, &value) in result.iter_mut().zip(values) {
                *target += factor * value;
            }
        }
        self.grid.filter_two_thirds(&mut result);
        result[0] = Complex64::ZERO;
        result
    }
}

pub fn kinetic_energy(u: &[f64], v: &[f64]) -> Result<f64, String> {
    if u.is_empty() || u.len() != v.len() {
        return Err("u and v must be non-empty arrays of equal length".into());
    }
    Ok(u.iter()
        .zip(v)
        .map(|(&u, &v)| 0.5 * (u * u + v * v))
        .sum::<f64>()
        / u.len() as f64)
}

pub fn enstrophy(omega: &[f64]) -> Result<f64, String> {
    if omega.is_empty() {
        return Err("omega must be non-empty".into());
    }
    Ok(omega.iter().map(|&value| 0.5 * value * value).sum::<f64>() / omega.len() as f64)
}
