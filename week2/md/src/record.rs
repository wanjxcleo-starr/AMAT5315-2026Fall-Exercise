use std::io::{self, Write};

use crate::fluid::{
    Box2, FluidState, ForceMethod, forces_and_potential_with_method, kinetic_energy, wrapped,
};

pub struct RunMetadata {
    pub n: usize,
    pub rho: f64,
    pub box2: Box2,
    pub dt: f64,
    pub temperature: f64,
    pub ramp_to: Option<f64>,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

pub struct SavedFrame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub e_pot: f64,
    pub e_kin: f64,
}

fn nine_digits(value: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err("non-finite position or velocity in saved frame".into());
    }
    format!("{value:.8e}")
        .parse::<f64>()
        .map_err(|error| error.to_string())
}

impl SavedFrame {
    pub fn from_state(step: usize, dt: f64, state: &FluidState) -> Result<Self, String> {
        Self::from_state_with_method(step, dt, state, ForceMethod::Naive)
    }

    pub fn from_state_with_method(
        step: usize,
        dt: f64,
        state: &FluidState,
        method: ForceMethod,
    ) -> Result<Self, String> {
        let mut saved = state.clone();
        for point in &mut saved.pos {
            for (axis, length) in [state.box2.lx, state.box2.ly].into_iter().enumerate() {
                let rounded = nine_digits(wrapped(point[axis], length))?;
                point[axis] = if rounded >= length { 0.0 } else { rounded };
            }
        }
        for velocity in &mut saved.vel {
            for component in velocity {
                *component = nine_digits(*component)?;
            }
        }
        let e_pot = forces_and_potential_with_method(&saved, method)?.1;
        let e_kin = kinetic_energy(&saved);
        let t = step as f64 * dt;
        if !t.is_finite() || !e_kin.is_finite() {
            return Err("non-finite saved time or kinetic energy".into());
        }
        Ok(Self {
            step,
            t,
            pos: saved.pos,
            vel: saved.vel,
            e_pot,
            e_kin,
        })
    }
}

pub fn write_run_json(writer: &mut impl Write, metadata: &RunMetadata) -> io::Result<()> {
    write!(
        writer,
        "{{\"n\":{},\"rho\":{:.17e},\"box\":[{:.17e},{:.17e}],\"dt\":{:.17e},\"temperature\":{:.17e},\"eq_steps\":{},\"steps\":{},\"sample_every\":{},\"seed\":{},\"ramp_to\":",
        metadata.n,
        metadata.rho,
        metadata.box2.lx,
        metadata.box2.ly,
        metadata.dt,
        metadata.temperature,
        metadata.eq_steps,
        metadata.steps,
        metadata.sample_every,
        metadata.seed
    )?;
    match metadata.ramp_to {
        Some(target) => write!(writer, "{target:.17e}")?,
        None => write!(writer, "null")?,
    }
    writeln!(writer, ",\"integrator\":\"velocity-verlet\"}}")
}

pub fn write_frame_jsonl(writer: &mut impl Write, frame: &SavedFrame) -> io::Result<()> {
    write!(
        writer,
        "{{\"step\":{},\"t\":{:.17e},\"pos\":[",
        frame.step, frame.t
    )?;
    for (index, point) in frame.pos.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "[{:.8e},{:.8e}]", point[0], point[1])?;
    }
    write!(writer, "],\"vel\":[")?;
    for (index, velocity) in frame.vel.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "[{:.8e},{:.8e}]", velocity[0], velocity[1])?;
    }
    writeln!(
        writer,
        "],\"E_pot\":{:.17e},\"E_kin\":{:.17e}}}",
        frame.e_pot, frame.e_kin
    )
}

#[cfg(test)]
mod tests {
    use super::{RunMetadata, SavedFrame, write_frame_jsonl, write_run_json};
    use crate::fluid::{FluidState, ForceMethod, forces_and_potential, kinetic_energy};

    #[test]
    fn cell_method_records_energy_from_serialized_coordinates() {
        let state = FluidState::new(100, 0.8, 0.5, 2026).unwrap();
        let frame =
            SavedFrame::from_state_with_method(50, 0.01, &state, ForceMethod::Cells).unwrap();
        let saved = FluidState {
            pos: frame.pos,
            vel: frame.vel,
            box2: state.box2,
        };
        assert_eq!(frame.e_pot, forces_and_potential(&saved).unwrap().1);
        assert_eq!(frame.e_kin, kinetic_energy(&saved));
    }

    #[test]
    fn recorded_energy_uses_nine_digit_saved_coordinates_and_velocities() {
        let state = FluidState::new(100, 0.8, 0.5, 2026).unwrap();
        let frame = SavedFrame::from_state(50, 0.01, &state).unwrap();
        assert_eq!(frame.step, 50);
        assert_eq!(frame.t, 0.5);
        let saved = FluidState {
            pos: frame.pos.clone(),
            vel: frame.vel.clone(),
            box2: state.box2,
        };
        assert_eq!(frame.e_pot, forces_and_potential(&saved).unwrap().1);
        assert_eq!(frame.e_kin, kinetic_energy(&saved));
        assert_ne!(frame.pos[1][0], state.pos[1][0]);
        let mut output = Vec::new();
        write_frame_jsonl(&mut output, &frame).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.starts_with("{\"step\":50,\"t\":5.00000000000000000e-1,\"pos\":["));
        assert!(text.ends_with("}\n"));
        assert!(!text.contains(' '));
        assert!(text.contains("1.20140571e0"));
    }

    #[test]
    fn run_metadata_has_required_fixed_schema() {
        let metadata = RunMetadata {
            n: 100,
            rho: 0.8,
            box2: crate::fluid::lattice(100, 0.8).unwrap().box2,
            dt: 0.01,
            temperature: 0.5,
            ramp_to: None,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
        };
        let mut output = Vec::new();
        write_run_json(&mut output, &metadata).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("\"box\":["));
        assert!(text.contains("\"integrator\":\"velocity-verlet\""));
        assert!(text.contains("\"sample_every\":50"));
        assert!(text.ends_with("}\n"));
        assert!(!text.contains(' '));
    }
}
