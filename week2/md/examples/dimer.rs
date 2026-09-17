use std::env;
use std::error::Error;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use md::dimer::{Observation, run_experiments};

fn main() -> Result<(), Box<dyn Error>> {
    let week2 = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("md crate is inside week2");
    let mut arguments = env::args_os().skip(1);
    let output = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| week2.join("dimer.png"));
    if arguments.next().is_some() {
        return Err("usage: dimer [OUTPUT.png]".into());
    }

    let experiments = run_experiments();
    let euler_final = experiments.euler_500[500].relative_error;
    let verlet_max = experiments
        .verlet_500
        .iter()
        .skip(1)
        .map(|point| point.relative_error.abs())
        .fold(0.0, f64::max);
    if !(euler_final > 0.5 && verlet_max < 1e-3) {
        return Err("500-step energy acceptance failed".into());
    }

    let mut plotter = Command::new("python3")
        .arg(week2.join("plot_dimer.py"))
        .arg(&output)
        .stdin(Stdio::piped())
        .spawn()?;
    {
        let mut csv = BufWriter::new(plotter.stdin.take().expect("plotter stdin is piped"));
        writeln!(csv, "run,step,time,relative_error")?;
        for (name, series) in [
            ("euler_500", &experiments.euler_500),
            ("verlet_500", &experiments.verlet_500),
            ("verlet_5000", &experiments.verlet_5000),
        ] {
            write_series(&mut csv, name, series)?;
        }
    }
    if !plotter.wait()?.success() {
        return Err("dimer plotter failed".into());
    }

    let long_errors: Vec<f64> = experiments
        .verlet_5000
        .iter()
        .map(|point| point.relative_error)
        .collect();
    let long_min = long_errors.iter().copied().fold(f64::INFINITY, f64::min);
    let long_max = long_errors
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let early_mean = long_errors[1..=500].iter().sum::<f64>() / 500.0;
    let late_mean = long_errors[4501..=5000].iter().sum::<f64>() / 500.0;
    println!("Euler final signed error: {euler_final:.9e}");
    println!("Verlet 500-step max absolute error: {verlet_max:.9e}");
    println!("Verlet 5000-step signed range: [{long_min:.9e}, {long_max:.9e}]");
    println!("Verlet early/late 500-step means: {early_mean:.9e}, {late_mean:.9e}");
    Ok(())
}

fn write_series(
    csv: &mut impl Write,
    name: &str,
    series: &[Observation],
) -> Result<(), std::io::Error> {
    for point in series {
        writeln!(
            csv,
            "{name},{},{:.17e},{:.17e}",
            point.step, point.time, point.relative_error
        )?;
    }
    Ok(())
}
