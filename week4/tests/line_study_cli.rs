use serde_json::Value;
use std::process::Command;

#[test]
fn line_study_emits_the_learning_sheet_cases_and_accepted_slopes() {
    let output = Command::new(env!("CARGO_BIN_EXE_line-study"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "line-study failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();

    assert_eq!(document["stability"]["n"], 64);
    assert_eq!(document["stability"]["stable_dt"], 0.045);
    assert_eq!(document["stability"]["unstable_dt"], 0.056);
    assert_eq!(document["accuracy"]["profile_n"], 64);

    let slopes = &document["accuracy"]["slopes"];
    for (method, expected) in [
        ("euler", 1.0),
        ("midpoint", 2.0),
        ("rk4", 4.0),
        ("equal_weight_rk4", 2.0),
    ] {
        let actual = slopes[method].as_f64().unwrap();
        assert!(
            (actual - expected).abs() <= 0.15 * expected,
            "{method} slope {actual} was not within 15% of {expected}"
        );
    }
}
