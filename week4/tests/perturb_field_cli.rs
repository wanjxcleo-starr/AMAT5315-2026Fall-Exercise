use spectral_fluid::FieldDocument;
use std::f64::consts::TAU;
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn perturb_field_preserves_metadata_and_array_sizes() {
    let n = 16;
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for iy in 0..n {
        let y = TAU * iy as f64 / n as f64;
        for ix in 0..n {
            let x = TAU * ix as f64 / n as f64;
            u.push(x.cos() * y.sin());
            v.push(-x.sin() * y.cos());
        }
    }
    let input = FieldDocument {
        case: "taylor-green".into(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    };
    let mut child = Command::new(env!("CARGO_BIN_EXE_perturb-field"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(child.stdin.as_mut().unwrap(), &input).unwrap();
    child.stdin.as_mut().unwrap().flush().unwrap();
    drop(child.stdin.take());

    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let perturbed: FieldDocument = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(perturbed.case, input.case);
    assert_eq!(perturbed.n, n);
    assert_eq!(perturbed.seed, input.seed);
    assert_eq!(perturbed.k_band, input.k_band);
    assert_eq!(perturbed.u.len(), n * n);
    assert_eq!(perturbed.v.len(), n * n);
}
