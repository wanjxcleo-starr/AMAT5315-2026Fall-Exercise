use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn field_taylor_green_writes_the_designed_json_object() {
    let output = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8", "--nu", "0.1", "--t", "1"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "field failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["case"], "taylor-green");
    assert_eq!(document["n"], 8);
    assert!(document["seed"].is_null());
    assert!(document["k_band"].is_null());
    assert_eq!(document["u"].as_array().unwrap().len(), 64);
    assert_eq!(document["v"].as_array().unwrap().len(), 64);

    let amplitude = (-0.2_f64).exp();
    let u_at_x0_ypi_over_2 = document["u"][2 * 8].as_f64().unwrap();
    let v_at_xpi_over_2_y0 = document["v"][2].as_f64().unwrap();
    assert!((u_at_x0_ypi_over_2 - amplitude).abs() < 1.0e-14);
    assert!((v_at_xpi_over_2_y0 + amplitude).abs() < 1.0e-14);
}

#[test]
fn random_field_has_half_energy_and_resolution_independent_phases() {
    let generate = |n: &str| {
        let output = Command::new(env!("CARGO_BIN_EXE_field"))
            .args([
                "random", "--n", n, "--seed", "2026", "--k-min", "2", "--k-max", "4",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "field failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<Value>(&output.stdout).unwrap()
    };
    let coarse = generate("16");
    let fine = generate("32");
    assert_eq!(coarse["seed"], 2026);
    assert_eq!(coarse["k_band"], serde_json::json!([2, 4]));

    let energy = |document: &Value| {
        let u = document["u"].as_array().unwrap();
        let v = document["v"].as_array().unwrap();
        u.iter()
            .zip(v)
            .map(|(u, v)| {
                let u = u.as_f64().unwrap();
                let v = v.as_f64().unwrap();
                0.5 * (u * u + v * v)
            })
            .sum::<f64>()
            / u.len() as f64
    };
    assert!((energy(&coarse) - 0.5).abs() < 1.0e-12);
    assert!((energy(&fine) - 0.5).abs() < 1.0e-12);

    for iy in 0..16 {
        for ix in 0..16 {
            let coarse_index = iy * 16 + ix;
            let fine_index = (2 * iy) * 32 + 2 * ix;
            for component in ["u", "v"] {
                let difference = (coarse[component][coarse_index].as_f64().unwrap()
                    - fine[component][fine_index].as_f64().unwrap())
                .abs();
                assert!(
                    difference < 1.0e-12,
                    "{component} differs by {difference:e}"
                );
            }
        }
    }
}

#[test]
fn fluid_writes_metadata_frames_and_energy_table() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8"])
        .output()
        .unwrap();
    assert!(field.status.success());
    let output_directory = temporary_directory("fluid-output");

    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "rk4",
            "--nu",
            "0.1",
            "--dt",
            "0.01",
            "--t-end",
            "0.02",
            "--every",
            "0.01",
            "--out",
            output_directory.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "fluid failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let table = String::from_utf8(output.stdout).unwrap();
    let table_lines: Vec<_> = table.lines().collect();
    assert_eq!(table_lines[0], "t\tE\tZ");
    assert_eq!(table_lines.len(), 4);

    let run: Value =
        serde_json::from_slice(&fs::read(output_directory.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["case"], "taylor-green");
    assert_eq!(run["n"], 8);
    assert_eq!(run["method"], "rk4");
    assert_eq!(run["nu"], 0.1);
    assert_eq!(run["dt"], 0.01);
    assert_eq!(run["t_end"], 0.02);
    assert_eq!(run["snapshot_every"], 0.01);

    let frames = fs::read_to_string(output_directory.join("fields.jsonl")).unwrap();
    let frame_lines: Vec<_> = frames.lines().collect();
    assert_eq!(frame_lines.len(), 3);
    for (step, line) in frame_lines.iter().enumerate() {
        let frame: Value = serde_json::from_str(line).unwrap();
        assert_eq!(frame["step"], step);
        for component in ["u", "v", "omega"] {
            assert_eq!(frame[component].as_array().unwrap().len(), 64);
            assert_array_has_six_decimal_places(line, component);
        }
    }

    fs::remove_dir_all(output_directory).unwrap();
}

#[test]
fn fluid_rejects_a_nonintegral_final_step_count() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8"])
        .output()
        .unwrap();
    let output_directory = temporary_directory("nonintegral-time");
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "rk4",
            "--nu",
            "0.1",
            "--dt",
            "0.01",
            "--t-end",
            "0.025",
            "--every",
            "0.01",
            "--out",
            output_directory.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("integer multiple of --dt"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output_directory.exists());
}

#[test]
fn fluid_accepts_a_final_step_ratio_with_roundoff_noise() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8"])
        .output()
        .unwrap();
    let output_directory = temporary_directory("rounded-time");
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "rk2",
            "--nu",
            "0.1",
            "--dt",
            "0.1",
            "--t-end",
            "0.30000000000000004",
            "--every",
            "0.1",
            "--out",
            output_directory.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "fluid failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 5);
    fs::remove_dir_all(output_directory).unwrap();
}

#[test]
fn fluid_flushes_prior_frames_before_exiting_on_nonfinite_energy() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8"])
        .output()
        .unwrap();
    let output_directory = temporary_directory("nonfinite-energy");
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "euler",
            "--nu",
            "1e308",
            "--dt",
            "1",
            "--t-end",
            "1",
            "--every",
            "1",
            "--out",
            output_directory.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert_eq!(output.status.code(), Some(1));
    let table = String::from_utf8(output.stdout).unwrap();
    assert_eq!(table.lines().count(), 3, "table was {table:?}");
    let energy: f64 = table
        .lines()
        .last()
        .unwrap()
        .split('\t')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert!(!energy.is_finite());
    let frames = fs::read_to_string(output_directory.join("fields.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 1, "frames were {frames:?}");
    fs::remove_dir_all(output_directory).unwrap();
}

#[test]
fn fluid_preserves_sub_microsecond_snapshot_times() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "8"])
        .output()
        .unwrap();
    let output_directory = temporary_directory("small-times");
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method",
            "rk4",
            "--nu",
            "0",
            "--dt",
            "0.0000001",
            "--t-end",
            "0.0000002",
            "--every",
            "0.0000001",
            "--out",
            output_directory.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&field.stdout)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());

    let table = String::from_utf8(output.stdout).unwrap();
    let table_times: Vec<f64> = table
        .lines()
        .skip(1)
        .map(|line| line.split('\t').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(table_times, [0.0, 1.0e-7, 2.0e-7]);

    let frames = fs::read_to_string(output_directory.join("fields.jsonl")).unwrap();
    let frame_times: Vec<f64> = frames
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["t"]
                .as_f64()
                .unwrap()
        })
        .collect();
    assert_eq!(frame_times, [0.0, 1.0e-7, 2.0e-7]);
    fs::remove_dir_all(output_directory).unwrap();
}

fn temporary_directory(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "spectral-fluid-{label}-{}-{unique}",
        std::process::id()
    ))
}

fn assert_array_has_six_decimal_places(line: &str, name: &str) {
    let marker = format!("\"{name}\":[");
    let values = line
        .split_once(&marker)
        .unwrap()
        .1
        .split_once(']')
        .unwrap()
        .0;
    for value in values.split(',') {
        let fractional = value
            .trim_start_matches('-')
            .split_once('.')
            .unwrap_or_else(|| panic!("{name} value {value:?} has no decimal point"))
            .1;
        assert_eq!(fractional.len(), 6, "{name} value {value:?}");
        assert!(fractional.bytes().all(|byte| byte.is_ascii_digit()));
    }
}
