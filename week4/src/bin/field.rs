use spectral_fluid::{FieldDocument, SpectralGrid, kinetic_energy};
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::fmt::Display;
use std::str::FromStr;

const USAGE: &str = "Usage:\n  field taylor-green --n N [--t T] [--nu NU]\n  field random --n N --seed SEED --k-min K --k-max K";

fn main() {
    if let Err(message) = run() {
        eprintln!("field: {message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        return Err(format!("missing subcommand\n{USAGE}"));
    };
    if command == "--help" || command == "-h" {
        println!("{USAGE}");
        return Ok(());
    }
    let values = option_map(arguments)?;
    let document = match command.as_str() {
        "taylor-green" => taylor_green(&values)?,
        "random" => random(&values)?,
        _ => return Err(format!("unknown subcommand {command:?}\n{USAGE}")),
    };
    serde_json::to_writer(std::io::stdout().lock(), &document)
        .map_err(|error| error.to_string())?;
    println!();
    Ok(())
}

fn option_map(arguments: impl Iterator<Item = String>) -> Result<HashMap<String, String>, String> {
    let mut arguments = arguments;
    let mut values = HashMap::new();
    while let Some(name) = arguments.next() {
        if !name.starts_with("--") {
            return Err(format!("unexpected argument {name:?}"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {name}"))?;
        if values.insert(name.clone(), value).is_some() {
            return Err(format!("duplicate option {name}"));
        }
    }
    Ok(values)
}

fn required<T>(values: &HashMap<String, String>, name: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    values
        .get(name)
        .ok_or_else(|| format!("missing {name}"))?
        .parse()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn optional<T>(values: &HashMap<String, String>, name: &str) -> Result<Option<T>, String>
where
    T: FromStr,
    T::Err: Display,
{
    values
        .get(name)
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid {name}: {error}"))
        })
        .transpose()
}

fn taylor_green(values: &HashMap<String, String>) -> Result<FieldDocument, String> {
    reject_unknown(values, &["--n", "--t", "--nu"])?;
    let n: usize = required(values, "--n")?;
    let grid = SpectralGrid::new(n)?;
    drop(grid);
    let time = optional(values, "--t")?.unwrap_or(0.0_f64);
    if !time.is_finite() || time < 0.0 {
        return Err("--t must be finite and non-negative".into());
    }
    let viscosity = optional::<f64>(values, "--nu")?;
    if viscosity.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err("--nu must be finite and non-negative".into());
    }
    if time > 0.0 && viscosity.is_none() {
        return Err("--nu is required when --t is greater than zero".into());
    }
    let amplitude = (-2.0 * viscosity.unwrap_or(0.0) * time).exp();
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for iy in 0..n {
        let y = TAU * iy as f64 / n as f64;
        for ix in 0..n {
            let x = TAU * ix as f64 / n as f64;
            u.push(x.cos() * y.sin() * amplitude);
            v.push(-x.sin() * y.cos() * amplitude);
        }
    }
    Ok(FieldDocument {
        case: "taylor-green".into(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    })
}

fn random(values: &HashMap<String, String>) -> Result<FieldDocument, String> {
    reject_unknown(values, &["--n", "--seed", "--k-min", "--k-max"])?;
    let n: usize = required(values, "--n")?;
    let grid = SpectralGrid::new(n)?;
    let seed: u64 = required(values, "--seed")?;
    let k_min: i32 = required(values, "--k-min")?;
    let k_max: i32 = required(values, "--k-max")?;
    if k_min < 1 || k_max < k_min {
        return Err("require 1 <= --k-min <= --k-max".into());
    }
    if k_max as usize > n / 3 {
        return Err("--k-max must not exceed floor(N/3)".into());
    }

    let mut omega = vec![0.0; n * n];
    let min_squared = k_min * k_min;
    let max_squared = k_max * k_max;
    let mut mode_count = 0_usize;
    for ky in -k_max..=k_max {
        for kx in -k_max..=k_max {
            let squared = kx * kx + ky * ky;
            let canonical = ky > 0 || (ky == 0 && kx > 0);
            if !canonical || squared < min_squared || squared > max_squared {
                continue;
            }
            mode_count += 1;
            let phase = mode_phase(seed, kx, ky);
            for iy in 0..n {
                let y = TAU * iy as f64 / n as f64;
                for ix in 0..n {
                    let x = TAU * ix as f64 / n as f64;
                    omega[iy * n + ix] += (kx as f64 * x + ky as f64 * y + phase).cos();
                }
            }
        }
    }
    if mode_count == 0 {
        return Err("the requested k band contains no integer wave vectors".into());
    }
    let (mut u, mut v) = grid.velocity_from_vorticity(&omega)?;
    let energy = kinetic_energy(&u, &v)?;
    if !energy.is_finite() || energy <= 0.0 {
        return Err("random field has no finite kinetic energy".into());
    }
    let scale = (0.5 / energy).sqrt();
    for value in u.iter_mut().chain(v.iter_mut()) {
        *value *= scale;
    }
    Ok(FieldDocument {
        case: "random".into(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    })
}

fn mode_phase(seed: u64, kx: i32, ky: i32) -> f64 {
    let mut value = seed
        ^ (kx as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (ky as i64 as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let unit = (value >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
    TAU * unit
}

fn reject_unknown(values: &HashMap<String, String>, allowed: &[&str]) -> Result<(), String> {
    if let Some(name) = values.keys().find(|name| !allowed.contains(&name.as_str())) {
        return Err(format!("unknown option {name}"));
    }
    Ok(())
}
