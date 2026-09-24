use std::f64::consts::TAU;

pub struct CenteredDifferenceGrid {
    n: usize,
    dx: f64,
}

impl CenteredDifferenceGrid {
    pub fn new(n: usize) -> Result<Self, String> {
        if n < 3 {
            return Err("n must be at least 3".into());
        }
        Ok(Self {
            n,
            dx: TAU / n as f64,
        })
    }

    pub fn derivative_x(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        self.validate(values)?;
        Ok((0..self.n * self.n)
            .map(|index| {
                let ix = index % self.n;
                let iy = index / self.n;
                let previous = values[iy * self.n + (ix + self.n - 1) % self.n];
                let next = values[iy * self.n + (ix + 1) % self.n];
                (next - previous) / (2.0 * self.dx)
            })
            .collect())
    }

    pub fn second_derivative_x(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        self.validate(values)?;
        Ok((0..self.n * self.n)
            .map(|index| {
                let ix = index % self.n;
                let iy = index / self.n;
                let previous = values[iy * self.n + (ix + self.n - 1) % self.n];
                let current = values[index];
                let next = values[iy * self.n + (ix + 1) % self.n];
                (next - 2.0 * current + previous) / self.dx.powi(2)
            })
            .collect())
    }

    pub fn mixed_derivative_xy(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        self.validate(values)?;
        Ok((0..self.n * self.n)
            .map(|index| {
                let ix = index % self.n;
                let iy = index / self.n;
                let previous_x = (ix + self.n - 1) % self.n;
                let next_x = (ix + 1) % self.n;
                let previous_y = (iy + self.n - 1) % self.n;
                let next_y = (iy + 1) % self.n;
                (values[next_y * self.n + next_x]
                    - values[previous_y * self.n + next_x]
                    - values[next_y * self.n + previous_x]
                    + values[previous_y * self.n + previous_x])
                    / (4.0 * self.dx.powi(2))
            })
            .collect())
    }

    pub fn laplacian(&self, values: &[f64]) -> Result<Vec<f64>, String> {
        self.validate(values)?;
        Ok((0..self.n * self.n)
            .map(|index| {
                let ix = index % self.n;
                let iy = index / self.n;
                let previous_x = values[iy * self.n + (ix + self.n - 1) % self.n];
                let next_x = values[iy * self.n + (ix + 1) % self.n];
                let previous_y = values[((iy + self.n - 1) % self.n) * self.n + ix];
                let next_y = values[((iy + 1) % self.n) * self.n + ix];
                (previous_x + next_x + previous_y + next_y - 4.0 * values[index]) / self.dx.powi(2)
            })
            .collect())
    }

    fn validate(&self, values: &[f64]) -> Result<(), String> {
        if values.len() != self.n * self.n {
            return Err(format!(
                "expected {} grid values, got {}",
                self.n * self.n,
                values.len()
            ));
        }
        Ok(())
    }
}
