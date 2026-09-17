const CUTOFF: f64 = 2.5;

#[derive(Clone, Copy, Debug)]
pub struct Box2 {
    pub lx: f64,
    pub ly: f64,
}

#[derive(Clone, Debug)]
pub struct FluidState {
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub box2: Box2,
}

pub fn lattice(n: usize, rho: f64) -> Result<FluidState, String> {
    if !rho.is_finite() || rho <= 0.0 {
        return Err("rho must be positive and finite".into());
    }
    let side = (n as f64).sqrt() as usize;
    if side < 2 || side % 2 != 0 || side.checked_mul(side) != Some(n) {
        return Err("n must be an even perfect square".into());
    }
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 3.0_f64.sqrt() * a / 2.0;
    let box2 = Box2 {
        lx: side as f64 * a,
        ly: side as f64 * h,
    };
    if !box2.lx.is_finite() || !box2.ly.is_finite() || box2.lx.min(box2.ly) <= 2.0 * CUTOFF {
        return Err("box lengths must be finite and greater than twice the cutoff".into());
    }
    let mut pos = Vec::with_capacity(n);
    for row in 0..side {
        for column in 0..side {
            pos.push([(column as f64 + 0.5 * (row % 2) as f64) * a, row as f64 * h]);
        }
    }
    Ok(FluidState {
        pos,
        vel: vec![[0.0; 2]; n],
        box2,
    })
}

pub fn minimum_image(displacement: f64, length: f64) -> f64 {
    displacement - length * (displacement / length).round()
}

pub fn wrapped(position: f64, length: f64) -> f64 {
    position.rem_euclid(length)
}

pub fn shifted_pair_energy(r: f64) -> f64 {
    if r >= CUTOFF {
        0.0
    } else {
        crate::lennard_jones_energy(r) - crate::lennard_jones_energy(CUTOFF)
    }
}

pub fn forces_and_potential(state: &FluidState) -> Result<(Vec<[f64; 2]>, f64), String> {
    let mut forces = vec![[0.0; 2]; state.pos.len()];
    let mut potential = 0.0;
    for i in 0..state.pos.len() {
        for j in i + 1..state.pos.len() {
            let dx = minimum_image(state.pos[j][0] - state.pos[i][0], state.box2.lx);
            let dy = minimum_image(state.pos[j][1] - state.pos[i][1], state.box2.ly);
            let r2 = dx * dx + dy * dy;
            if !r2.is_finite() || r2 == 0.0 {
                return Err("non-finite or overlapping atom positions".into());
            }
            if r2 >= CUTOFF * CUTOFF {
                continue;
            }
            let r = r2.sqrt();
            potential += shifted_pair_energy(r);
            let scale = crate::lennard_jones_radial_force(r) / r;
            let fx = scale * dx;
            let fy = scale * dy;
            forces[i][0] -= fx;
            forces[i][1] -= fy;
            forces[j][0] += fx;
            forces[j][1] += fy;
        }
    }
    if !potential.is_finite() || forces.iter().flatten().any(|x| !x.is_finite()) {
        return Err("non-finite pair force or energy".into());
    }
    Ok((forces, potential))
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        (value >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn gaussian_pair(&mut self, standard_deviation: f64) -> [f64; 2] {
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt() * standard_deviation;
        let angle = std::f64::consts::TAU * self.uniform();
        [radius * angle.cos(), radius * angle.sin()]
    }
}

impl FluidState {
    pub fn new(n: usize, rho: f64, temperature: f64, seed: u64) -> Result<Self, String> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("temperature must be positive and finite".into());
        }
        let mut state = lattice(n, rho)?;
        let mut generator = SplitMix64(seed);
        for velocity in &mut state.vel {
            *velocity = generator.gaussian_pair(temperature.sqrt());
        }
        for axis in 0..2 {
            let mean = state.vel.iter().map(|v| v[axis]).sum::<f64>() / n as f64;
            for velocity in &mut state.vel {
                velocity[axis] -= mean;
            }
        }
        rescale_to_temperature(&mut state, temperature)?;
        Ok(state)
    }
}

pub fn kinetic_energy(state: &FluidState) -> f64 {
    state.vel.iter().flatten().map(|v| 0.5 * v * v).sum()
}

pub fn thermostat_temperature(state: &FluidState) -> f64 {
    2.0 * kinetic_energy(state) / (2 * state.vel.len() - 2) as f64
}

pub fn rescale_to_temperature(state: &mut FluidState, target: f64) -> Result<(), String> {
    let current = thermostat_temperature(state);
    if !target.is_finite() || target <= 0.0 || !current.is_finite() || current <= 0.0 {
        return Err("cannot rescale non-finite or zero temperature".into());
    }
    let scale = (target / current).sqrt();
    for velocity in &mut state.vel {
        for component in velocity {
            *component *= scale;
        }
    }
    Ok(())
}

pub fn velocity_verlet_step(state: &mut FluidState, dt: f64) -> Result<(), String> {
    let (old_forces, _) = forces_and_potential(state)?;
    for (atom, force) in old_forces.iter().enumerate() {
        for (axis, length) in [state.box2.lx, state.box2.ly].into_iter().enumerate() {
            state.pos[atom][axis] = wrapped(
                state.pos[atom][axis] + dt * state.vel[atom][axis] + 0.5 * dt * dt * force[axis],
                length,
            );
        }
    }
    let (new_forces, _) = forces_and_potential(state)?;
    for atom in 0..state.pos.len() {
        for axis in 0..2 {
            state.vel[atom][axis] += 0.5 * dt * (old_forces[atom][axis] + new_forces[atom][axis]);
            if !state.vel[atom][axis].is_finite() {
                return Err("non-finite velocity".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Box2, FluidState, forces_and_potential, kinetic_energy, lattice, minimum_image,
        shifted_pair_energy, thermostat_temperature, velocity_verlet_step, wrapped,
    };

    #[test]
    fn seeded_gaussian_initialization_removes_drift_and_sets_thermostat_temperature() {
        let a = FluidState::new(100, 0.8, 0.5, 2026).unwrap();
        let b = FluidState::new(100, 0.8, 0.5, 2026).unwrap();
        assert_eq!(a.vel, b.vel);
        assert!(a.vel.iter().flatten().all(|x| x.is_finite()));
        for axis in 0..2 {
            let mean = a.vel.iter().map(|v| v[axis]).sum::<f64>() / 100.0;
            assert!(mean.abs() < 1e-14);
        }
        assert!((thermostat_temperature(&a) - 0.5).abs() < 1e-14);
        assert!((kinetic_energy(&a) - 49.5).abs() < 1e-12);
    }

    #[test]
    fn one_verlet_step_keeps_momentum_and_wraps_positions() {
        let mut state = FluidState::new(100, 0.8, 0.5, 2026).unwrap();
        let before = [0, 1].map(|axis| state.vel.iter().map(|v| v[axis]).sum::<f64>());
        velocity_verlet_step(&mut state, 0.01).unwrap();
        let after = [0, 1].map(|axis| state.vel.iter().map(|v| v[axis]).sum::<f64>());
        for axis in 0..2 {
            assert!((after[axis] - before[axis]).abs() < 1e-11);
        }
        assert!(
            state.pos.iter().all(|p| 0.0 <= p[0]
                && p[0] < state.box2.lx
                && 0.0 <= p[1]
                && p[1] < state.box2.ly)
        );
    }

    #[test]
    fn periodic_seam_and_shifted_cutoff_preserve_pair_symmetry() {
        assert!((minimum_image(8.8, 10.0) + 1.2).abs() < 1e-12);
        assert!((wrapped(-0.1, 10.0) - 9.9).abs() < 1e-12);
        assert_eq!(shifted_pair_energy(2.5), 0.0);
        assert_eq!(shifted_pair_energy(3.0), 0.0);
        assert!(
            (shifted_pair_energy(1.2)
                - (crate::lennard_jones_energy(1.2) - crate::lennard_jones_energy(2.5)))
            .abs()
                < 1e-12
        );
        let state = FluidState {
            pos: vec![[0.1, 5.0], [8.9, 5.0]],
            vel: vec![[0.0; 2]; 2],
            box2: Box2 { lx: 10.0, ly: 10.0 },
        };
        let (forces, energy) = forces_and_potential(&state).unwrap();
        assert!((energy - shifted_pair_energy(1.2)).abs() < 1e-12);
        assert!((forces[0][0] + forces[1][0]).abs() < 1e-12);
        assert_eq!(forces[0][1], 0.0);
        assert_eq!(forces[1][1], 0.0);
        assert!(forces[1][0] > 0.0);
    }

    #[test]
    fn hundred_atoms_form_ten_staggered_rows_at_requested_density() {
        let state = lattice(100, 0.8).unwrap();
        assert_eq!(state.pos.len(), 100);
        assert_eq!(state.vel.len(), 100);
        assert!((state.box2.lx - 12.0141).abs() < 1e-4);
        assert!((state.box2.ly - 10.4045).abs() < 1e-4);
        assert!((state.box2.lx * state.box2.ly - 125.0).abs() < 1e-10);
        assert_eq!(state.pos[0], [0.0, 0.0]);
        assert!((state.pos[10][0] - state.box2.lx / 20.0).abs() < 1e-12);
        assert!((state.pos[10][1] - state.box2.ly / 10.0).abs() < 1e-12);
        assert!(state.pos.iter().all(|p| {
            0.0 <= p[0] && p[0] < state.box2.lx && 0.0 <= p[1] && p[1] < state.box2.ly
        }));
    }
}
