use serde_json::Value;
use std::process::Command;

#[test]
fn part2_study_reports_roundoff_fourier_errors_and_second_order_ratios() {
    let output = Command::new(env!("CARGO_BIN_EXE_part2-study"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "part2-study failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    let rows = document["derivatives"].as_array().unwrap();
    assert_eq!(rows.len(), 4);
    for row in rows {
        assert!(row["fourier_n32"].as_f64().unwrap() < 1.0e-10);
        let ratio = row["finite_difference_n32"].as_f64().unwrap()
            / row["finite_difference_n64"].as_f64().unwrap();
        assert!((3.8..4.1).contains(&ratio));
    }
}
