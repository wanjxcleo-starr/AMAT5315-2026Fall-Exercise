use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
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
