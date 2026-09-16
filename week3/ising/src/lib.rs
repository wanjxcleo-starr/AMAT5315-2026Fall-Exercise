/// A periodic square lattice with row-major spins.
pub struct Lattice {
    side: usize,
    spins: Vec<i8>,
    spin_sum: i64,
}

impl Lattice {
    pub fn all_up(side: usize) -> Result<Self, String> {
        if side < 2 {
            return Err("--l must be at least 2".into());
        }
        let sites = side
            .checked_mul(side)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or("lattice is too large")?;
        let mut spins = Vec::new();
        spins
            .try_reserve_exact(sites)
            .map_err(|_| "cannot allocate lattice")?;
        spins.resize(sites, 1);
        Ok(Self {
            side,
            spins,
            spin_sum: sites as i64,
        })
    }

    pub fn spins(&self) -> &[i8] {
        &self.spins
    }

    pub fn mean_spin(&self) -> f64 {
        self.spin_sum as f64 / self.spins.len() as f64
    }

    /// Count each right and down bond once, including periodic bonds at L = 2.
    pub fn energy_per_site(&self) -> f64 {
        let mut energy = 0_i64;
        for row in 0..self.side {
            for col in 0..self.side {
                let site = row * self.side + col;
                let right = row * self.side + (col + 1) % self.side;
                let down = ((row + 1) % self.side) * self.side + col;
                energy -= i64::from(self.spins[site])
                    * (i64::from(self.spins[right]) + i64::from(self.spins[down]));
            }
        }
        energy as f64 / self.spins.len() as f64
    }

    fn flip_energy_change(&self, site: usize) -> i32 {
        let row = site / self.side;
        let col = site % self.side;
        let up = ((row + self.side - 1) % self.side) * self.side + col;
        let down = ((row + 1) % self.side) * self.side + col;
        let left = row * self.side + (col + self.side - 1) % self.side;
        let right = row * self.side + (col + 1) % self.side;
        2 * i32::from(self.spins[site])
            * (i32::from(self.spins[up])
                + i32::from(self.spins[down])
                + i32::from(self.spins[left])
                + i32::from(self.spins[right]))
    }

    /// One sweep makes L*L independent, uniform site proposals with replacement.
    pub fn metropolis_sweep(&mut self, temperature: f64, rng: &mut SplitMix64) -> usize {
        let accept_four = (-4.0 / temperature).exp();
        let accept_eight = (-8.0 / temperature).exp();
        let mut accepted = 0;
        for _ in 0..self.spins.len() {
            let site = rng.index(self.spins.len());
            let delta = self.flip_energy_change(site);
            let should_flip = match delta {
                ..=0 => true,
                4 => rng.unit_f64() < accept_four,
                8 => rng.unit_f64() < accept_eight,
                _ => unreachable!("four-neighbor Ising flip has ΔE in -8,-4,0,4,8"),
            };
            if should_flip {
                let old = self.spins[site];
                self.spins[site] = -old;
                self.spin_sum -= 2 * i64::from(old);
                accepted += 1;
            }
        }
        accepted
    }
}

/// One deterministic random stream is carried through the full temperature ramp.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        let upper = upper as u64;
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % upper) as usize;
            }
        }
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

/// Generate temperatures from the input bounds without accumulating step error.
pub fn temperature_grid(from: f64, to: f64, step: f64) -> Result<Vec<f64>, String> {
    if !from.is_finite() || !to.is_finite() || !step.is_finite() {
        return Err("temperatures and step must be finite".into());
    }
    if from > to || step <= 0.0 {
        return Err("require --t-from <= --t-to and --t-step > 0".into());
    }

    let mut grid = Vec::new();
    let tolerance = (64.0 * f64::EPSILON * from.abs().max(to.abs()).max(1.0)).min(step * 1e-8);
    let mut index = 0_usize;
    let mut previous = f64::NEG_INFINITY;
    loop {
        let current = from + index as f64 * step;
        if current > to && current - to > tolerance {
            break;
        }
        if current <= previous {
            return Err("temperature step is too small at this scale".into());
        }
        grid.push(if (current - to).abs() <= tolerance {
            to
        } else {
            current
        });
        previous = current;
        index = index.checked_add(1).ok_or("too many temperatures")?;
    }
    Ok(grid)
}

#[cfg(test)]
mod tests {
    use super::{Lattice, SplitMix64, temperature_grid};

    #[test]
    fn periodic_energy_change_matches_a_flip_at_side_two() {
        let mut lattice = Lattice::all_up(2).unwrap();
        assert_eq!(lattice.energy_per_site(), -2.0);
        assert_eq!(lattice.flip_energy_change(0), 8);
        lattice.spins[0] = -1;
        lattice.spin_sum -= 2;
        assert_eq!(lattice.energy_per_site(), 0.0);
        assert_eq!(lattice.mean_spin(), 0.5);
    }

    #[test]
    fn energy_change_matches_energy_difference_at_side_three() {
        for pattern in 0..(1 << 9) {
            let mut lattice = Lattice::all_up(3).unwrap();
            for site in 0..9 {
                if pattern & (1 << site) != 0 {
                    lattice.spins[site] = -1;
                }
            }
            for site in 0..9 {
                let before = lattice.energy_per_site();
                let delta = lattice.flip_energy_change(site);
                lattice.spins[site] = -lattice.spins[site];
                let after = lattice.energy_per_site();
                assert!(((after - before) * 9.0 - f64::from(delta)).abs() < 1e-10);
                lattice.spins[site] = -lattice.spins[site];
            }
        }
    }

    #[test]
    fn sweep_uses_a_reproducible_stream() {
        let mut first = Lattice::all_up(4).unwrap();
        let mut second = Lattice::all_up(4).unwrap();
        let mut first_rng = SplitMix64::new(2026);
        let mut second_rng = SplitMix64::new(2026);
        for _ in 0..10 {
            assert_eq!(
                first.metropolis_sweep(2.0, &mut first_rng),
                second.metropolis_sweep(2.0, &mut second_rng)
            );
        }
        assert_eq!(first.spins(), second.spins());
    }

    #[test]
    fn temperature_grid_includes_only_reached_endpoint() {
        assert_eq!(temperature_grid(1.5, 1.6, 0.05).unwrap(), [1.5, 1.55, 1.6]);
        assert_eq!(temperature_grid(1.5, 1.59, 0.05).unwrap(), [1.5, 1.55]);
    }
}
