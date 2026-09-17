use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use md::{
    fluid::{FluidState, ForceMethod, rescale_to_temperature, velocity_verlet_step_with_method},
    record::{SavedFrame, write_frame_jsonl},
};

fn output_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("md-{label}-{}-{nonce}", std::process::id()))
}

fn flags(out: &std::path::Path) -> Vec<String> {
    [
        "--n",
        "100",
        "--rho",
        "0.8",
        "--temperature",
        "0.5",
        "--dt",
        "0.01",
        "--eq-steps",
        "50",
        "--steps",
        "20",
        "--sample-every",
        "5",
        "--seed",
        "2026",
        "--out",
        out.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn set_flag(args: &mut [String], flag: &str, value: &str) {
    let index = args.iter().position(|arg| arg == flag).unwrap() + 1;
    args[index] = value.to_string();
}

fn recorded_temperature(frame: &str) -> f64 {
    let kinetic = frame
        .split_once("\"E_kin\":")
        .unwrap()
        .1
        .trim_end_matches('}')
        .parse::<f64>()
        .unwrap();
    2.0 * kinetic / 198.0
}

#[test]
fn required_options_and_runtime_failure_use_distinct_exit_codes() {
    let binary = env!("CARGO_BIN_EXE_md");
    for extra in [
        vec![],
        vec!["--unknown".to_string()],
        vec!["--n".to_string(), "100".to_string()],
    ] {
        let result = Command::new(binary).args(&extra).output().unwrap();
        assert_eq!(result.status.code(), Some(2), "{extra:?}");
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }
    let out = output_dir("usage");
    let mut duplicate = flags(&out);
    duplicate.extend(["--n".to_string(), "100".to_string()]);
    let result = Command::new(binary).args(&duplicate).output().unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());

    fs::write(&out, b"occupied").unwrap();
    let result = Command::new(binary).args(flags(&out)).output().unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    fs::remove_file(out).unwrap();
}

#[test]
fn huge_atom_count_is_a_usage_error_instead_of_an_overflow() {
    let out = output_dir("huge-n");
    let mut args = flags(&out);
    args[1] = usize::MAX.to_string();
    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    assert!(!out.exists());
}

#[test]
fn short_run_records_only_sampled_production_frames() {
    let out = output_dir("record");
    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(flags(&out))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stderr.is_empty());
    let stdout = String::from_utf8(result.stdout).unwrap();
    let rows: Vec<_> = stdout.lines().collect();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0], "t\tE_pot\tE_kin");
    assert_eq!(
        rows[1].split('\t').next().unwrap().parse::<f64>().unwrap(),
        0.05
    );
    assert_eq!(
        rows[4].split('\t').next().unwrap().parse::<f64>().unwrap(),
        0.2
    );
    let metadata = fs::read_to_string(out.join("run.json")).unwrap();
    assert!(metadata.contains("\"n\":100"));
    assert!(metadata.contains("\"eq_steps\":50"));
    assert!(metadata.contains("\"integrator\":\"velocity-verlet\""));
    let trajectory = fs::read_to_string(out.join("traj.jsonl")).unwrap();
    let frames: Vec<_> = trajectory.lines().collect();
    assert_eq!(frames.len(), 4);
    for (frame, step) in frames.iter().zip([5, 10, 15, 20]) {
        assert!(frame.starts_with(&format!("{{\"step\":{step},")));
        assert!(frame.contains("\"pos\":["));
        assert!(frame.contains("\"vel\":["));
        assert!(!frame.contains(' '));
    }
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn force_flags_run_the_same_short_trajectory_as_the_default() {
    let binary = env!("CARGO_BIN_EXE_md");
    let mut trajectories = Vec::new();
    for method in [None, Some("naive"), Some("cells")] {
        let out = output_dir(method.unwrap_or("default-force"));
        let mut args = flags(&out);
        if let Some(method) = method {
            args.extend(["--force".to_string(), method.to_string()]);
        }
        let result = Command::new(binary).args(args).output().unwrap();
        assert!(
            result.status.success(),
            "method={method:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(result.stderr.is_empty());
        trajectories.push(fs::read_to_string(out.join("traj.jsonl")).unwrap());
        fs::remove_dir_all(out).unwrap();
    }
    assert_eq!(trajectories[0], trajectories[1]);
    assert_eq!(trajectories[0], trajectories[2]);
}

#[test]
fn invalid_force_choice_is_a_usage_error() {
    let out = output_dir("invalid-force");
    let mut args = flags(&out);
    args.extend(["--force".to_string(), "octree".to_string()]);
    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("--force"));
    assert!(!out.exists());
}

#[test]
fn ramp_rescales_only_at_fifty_step_boundaries_to_linear_targets() {
    let binary = env!("CARGO_BIN_EXE_md");
    let control = output_dir("ramp-control");
    let heated = output_dir("ramp-heated");
    let mut control_args = flags(&control);
    set_flag(&mut control_args, "--temperature", "0.2");
    set_flag(&mut control_args, "--steps", "100");
    set_flag(&mut control_args, "--sample-every", "25");
    let control_result = Command::new(binary).args(control_args).output().unwrap();
    assert!(control_result.status.success());

    let mut heated_args = flags(&heated);
    set_flag(&mut heated_args, "--temperature", "0.2");
    set_flag(&mut heated_args, "--steps", "100");
    set_flag(&mut heated_args, "--sample-every", "25");
    heated_args.extend(["--ramp-to".to_string(), "1.2".to_string()]);
    let heated_result = Command::new(binary).args(heated_args).output().unwrap();
    assert!(
        heated_result.status.success(),
        "{}",
        String::from_utf8_lossy(&heated_result.stderr)
    );
    let control_traj = fs::read_to_string(control.join("traj.jsonl")).unwrap();
    let heated_traj = fs::read_to_string(heated.join("traj.jsonl")).unwrap();
    let control_frames: Vec<_> = control_traj.lines().collect();
    let heated_frames: Vec<_> = heated_traj.lines().collect();
    assert_eq!(heated_frames.len(), 4);
    assert_eq!(heated_frames[0], control_frames[0]);
    assert!((recorded_temperature(heated_frames[1]) - 0.7).abs() < 1e-8);
    assert!((recorded_temperature(heated_frames[3]) - 1.2).abs() < 1e-8);
    let metadata = fs::read_to_string(heated.join("run.json")).unwrap();
    let ramp_to = metadata
        .split_once("\"ramp_to\":")
        .unwrap()
        .1
        .split(',')
        .next()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert_eq!(ramp_to, 1.2);
    fs::remove_dir_all(control).unwrap();
    fs::remove_dir_all(heated).unwrap();
}

#[test]
fn unheated_run_keeps_production_thermostat_off_and_records_null_ramp() {
    let out = output_dir("unheated-regression");
    let mut args = flags(&out);
    set_flag(&mut args, "--temperature", "0.2");
    set_flag(&mut args, "--steps", "100");
    set_flag(&mut args, "--sample-every", "50");
    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(args)
        .output()
        .unwrap();
    assert!(result.status.success());
    let metadata = fs::read_to_string(out.join("run.json")).unwrap();
    assert!(metadata.contains("\"ramp_to\":null"));

    let mut state = FluidState::new(100, 0.8, 0.2, 2026).unwrap();
    for step in 1..=50 {
        velocity_verlet_step_with_method(&mut state, 0.01, ForceMethod::Cells).unwrap();
        if step % 50 == 0 {
            rescale_to_temperature(&mut state, 0.2).unwrap();
        }
    }
    for _ in 1..=100 {
        velocity_verlet_step_with_method(&mut state, 0.01, ForceMethod::Cells).unwrap();
    }
    let expected =
        SavedFrame::from_state_with_method(100, 0.01, &state, ForceMethod::Cells).unwrap();
    let mut expected_json = Vec::new();
    write_frame_jsonl(&mut expected_json, &expected).unwrap();
    let trajectory = fs::read_to_string(out.join("traj.jsonl")).unwrap();
    assert_eq!(
        trajectory.lines().last().unwrap(),
        String::from_utf8(expected_json).unwrap().trim_end()
    );
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn invalid_ramp_target_is_a_usage_error() {
    let out = output_dir("invalid-ramp");
    for target in ["0", "-1", "NaN"] {
        let mut args = flags(&out);
        args.extend(["--ramp-to".to_string(), target.to_string()]);
        let result = Command::new(env!("CARGO_BIN_EXE_md"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .contains("--ramp-to must be positive and finite")
        );
        assert!(!out.exists());
    }
}
