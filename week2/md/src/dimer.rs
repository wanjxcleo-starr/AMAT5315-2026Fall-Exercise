use crate::{lennard_jones_energy, lennard_jones_radial_force};

/// Positions and velocities of two unit-mass atoms in two dimensions.
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub positions: [[f64; 2]; 2],
    pub velocities: [[f64; 2]; 2],
}

pub fn initial_state() -> State {
    State {
        positions: [[0.0, 0.0], [1.2, 0.0]],
        velocities: [[0.0; 2]; 2],
    }
}

/// Pair forces, with the potential counted only once.
pub fn pair_forces(state: &State) -> [[f64; 2]; 2] {
    let dx = state.positions[1][0] - state.positions[0][0];
    let dy = state.positions[1][1] - state.positions[0][1];
    let r = dx.hypot(dy);
    let scale = lennard_jones_radial_force(r) / r;
    let force_on_second = [scale * dx, scale * dy];
    [[-force_on_second[0], -force_on_second[1]], force_on_second]
}

pub fn total_energy(state: &State) -> f64 {
    let dx = state.positions[1][0] - state.positions[0][0];
    let dy = state.positions[1][1] - state.positions[0][1];
    let kinetic = state
        .velocities
        .iter()
        .flat_map(|velocity| velocity.iter())
        .map(|component| 0.5 * component * component)
        .sum::<f64>();
    kinetic + lennard_jones_energy(dx.hypot(dy))
}

pub trait Integrator {
    fn step(&self, state: &mut State, dt: f64);
}

pub struct Euler;

impl Integrator for Euler {
    fn step(&self, state: &mut State, dt: f64) {
        let forces = pair_forces(state);
        for (atom, force) in forces.iter().enumerate() {
            for (axis, component) in force.iter().enumerate() {
                state.positions[atom][axis] += dt * state.velocities[atom][axis];
                state.velocities[atom][axis] += dt * component;
            }
        }
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, state: &mut State, dt: f64) {
        let old_forces = pair_forces(state);
        for (atom, force) in old_forces.iter().enumerate() {
            for (axis, component) in force.iter().enumerate() {
                state.positions[atom][axis] +=
                    dt * state.velocities[atom][axis] + 0.5 * dt * dt * component;
            }
        }

        let new_forces = pair_forces(state);
        for atom in 0..2 {
            for axis in 0..2 {
                state.velocities[atom][axis] +=
                    0.5 * dt * (old_forces[atom][axis] + new_forces[atom][axis]);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub step: usize,
    pub time: f64,
    pub energy: f64,
    pub relative_error: f64,
}

/// Run one integrator from the fixed dimer initial condition.
pub fn run<I: Integrator>(integrator: &I, steps: usize, dt: f64) -> Vec<Observation> {
    let mut state = initial_state();
    let initial_energy = total_energy(&state);
    let mut observations = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        if step > 0 {
            integrator.step(&mut state, dt);
        }
        let energy = total_energy(&state);
        observations.push(Observation {
            step,
            time: step as f64 * dt,
            energy,
            relative_error: (energy - initial_energy) / initial_energy.abs(),
        });
    }
    observations
}

pub struct Experiments {
    pub euler_500: Vec<Observation>,
    pub verlet_500: Vec<Observation>,
    pub verlet_5000: Vec<Observation>,
}

pub fn run_experiments() -> Experiments {
    const DT: f64 = 0.01;
    Experiments {
        euler_500: run(&Euler, 500, DT),
        verlet_500: run(&VelocityVerlet, 500, DT),
        verlet_5000: run(&VelocityVerlet, 5000, DT),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Euler, Integrator, VelocityVerlet, initial_state, pair_forces, run, run_experiments,
        total_energy,
    };

    #[test]
    fn initial_pair_force_is_attractive_and_equal_opposite() {
        let forces = pair_forces(&initial_state());
        assert!((forces[0][0] - 2.2116933422230778).abs() < 1e-12);
        assert!((forces[1][0] + 2.2116933422230778).abs() < 1e-12);
        assert_eq!(forces[0][1], 0.0);
        assert_eq!(forces[1][1], 0.0);
    }

    #[test]
    fn total_energy_counts_two_dimensional_kinetic_energy_and_one_pair() {
        let mut state = initial_state();
        state.velocities = [[1.0, 2.0], [-1.0, 0.5]];
        assert!((total_energy(&state) - 2.234034712416924).abs() < 1e-12);
    }

    #[test]
    fn forward_euler_uses_old_velocity_and_force() {
        let mut state = initial_state();
        Euler.step(&mut state, 0.01);
        assert_eq!(state.positions, [[0.0, 0.0], [1.2, 0.0]]);
        assert!((state.velocities[0][0] - 0.02211693342223078).abs() < 1e-14);
        assert!((state.velocities[1][0] + 0.02211693342223078).abs() < 1e-14);
        assert_eq!(state.velocities[0][1], 0.0);
        assert_eq!(state.velocities[1][1], 0.0);
    }

    #[test]
    fn velocity_verlet_recomputes_force_after_moving_atoms() {
        let mut state = initial_state();
        VelocityVerlet.step(&mut state, 0.01);
        assert!((state.positions[0][0] - 0.0001105846671111539).abs() < 1e-14);
        assert!((state.positions[1][0] - 1.1998894153328888).abs() < 1e-14);
        assert!((state.velocities[0][0] - 0.022106357406835794).abs() < 1e-14);
        assert!((state.velocities[1][0] + 0.022106357406835794).abs() < 1e-14);
        assert_eq!(state.positions[0][1], 0.0);
        assert_eq!(state.positions[1][1], 0.0);
    }

    #[test]
    fn observations_record_signed_relative_error_from_initial_energy() {
        let observations = run(&VelocityVerlet, 1, 0.01);
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].step, 0);
        assert_eq!(observations[0].time, 0.0);
        assert!((observations[0].energy + 0.89096528758307625).abs() < 1e-14);
        assert_eq!(observations[0].relative_error, 0.0);
        assert_eq!(observations[1].step, 1);
        assert_eq!(observations[1].time, 0.01);
        assert!((observations[1].relative_error + 2.6271960680612238e-7).abs() < 1e-12);
    }

    #[test]
    fn three_runs_meet_500_step_energy_acceptance() {
        let experiments = run_experiments();
        assert_eq!(experiments.euler_500.len(), 501);
        assert_eq!(experiments.verlet_500.len(), 501);
        assert_eq!(experiments.verlet_5000.len(), 5001);
        assert_eq!(experiments.euler_500[0].relative_error, 0.0);
        assert_eq!(experiments.verlet_500[0].relative_error, 0.0);
        assert_eq!(experiments.verlet_5000[0].relative_error, 0.0);

        let euler_final = experiments.euler_500[500].relative_error;
        let verlet_max = experiments
            .verlet_500
            .iter()
            .skip(1)
            .map(|point| point.relative_error.abs())
            .fold(0.0, f64::max);
        assert!(euler_final > 0.5, "Euler final error: {euler_final}");
        assert!(verlet_max < 1e-3, "Verlet maximum error: {verlet_max}");

        assert!(
            experiments
                .verlet_5000
                .iter()
                .all(|point| point.relative_error.is_finite())
        );
        assert!(
            experiments
                .verlet_5000
                .windows(3)
                .filter(|window| {
                    let before = window[1].relative_error - window[0].relative_error;
                    let after = window[2].relative_error - window[1].relative_error;
                    before * after < 0.0
                })
                .count()
                > 10
        );
    }
}
