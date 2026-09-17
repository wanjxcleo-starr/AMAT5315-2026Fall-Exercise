use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    process::ExitCode,
};

use md::{
    fluid::{FluidState, lattice, rescale_to_temperature, velocity_verlet_step},
    record::{RunMetadata, SavedFrame, write_frame_jsonl, write_run_json},
};

struct Config {
    n: usize,
    rho: f64,
    temperature: f64,
    dt: f64,
    eq_steps: usize,
    steps: usize,
    sample_every: usize,
    seed: u64,
    out: PathBuf,
}

fn parse_args() -> Result<Config, String> {
    const KEYS: [&str; 9] = [
        "n",
        "rho",
        "temperature",
        "dt",
        "eq-steps",
        "steps",
        "sample-every",
        "seed",
        "out",
    ];
    let mut options = HashMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let key = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument: {flag}"))?;
        if !KEYS.contains(&key) {
            return Err(format!("unknown option: {flag}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        if value.starts_with("--") || value.is_empty() {
            return Err(format!("missing value for {flag}"));
        }
        if options.insert(key.to_string(), value).is_some() {
            return Err(format!("duplicate option: {flag}"));
        }
    }
    for key in KEYS {
        if !options.contains_key(key) {
            return Err(format!("missing required option: --{key}"));
        }
    }
    let number = |key: &str| -> Result<usize, String> {
        options[key]
            .parse()
            .map_err(|_| format!("invalid --{key}: {}", options[key]))
    };
    let real = |key: &str| -> Result<f64, String> {
        let value: f64 = options[key]
            .parse()
            .map_err(|_| format!("invalid --{key}: {}", options[key]))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("--{key} must be positive and finite"));
        }
        Ok(value)
    };
    let n = number("n")?;
    let rho = real("rho")?;
    lattice(n, rho)?;
    let temperature = real("temperature")?;
    let dt = real("dt")?;
    let eq_steps = number("eq-steps")?;
    let steps = number("steps")?;
    let sample_every = number("sample-every")?;
    if steps == 0 || sample_every == 0 {
        return Err("--steps and --sample-every must be positive".into());
    }
    let seed = options["seed"]
        .parse()
        .map_err(|_| format!("invalid --seed: {}", options["seed"]))?;
    Ok(Config {
        n,
        rho,
        temperature,
        dt,
        eq_steps,
        steps,
        sample_every,
        seed,
        out: PathBuf::from(&options["out"]),
    })
}

fn run(config: Config) -> Result<(), String> {
    let mut state = FluidState::new(config.n, config.rho, config.temperature, config.seed)?;
    for step in 1..=config.eq_steps {
        velocity_verlet_step(&mut state, config.dt)?;
        if step % 50 == 0 {
            rescale_to_temperature(&mut state, config.temperature)?;
        }
    }
    fs::create_dir_all(&config.out).map_err(|e| format!("cannot create output directory: {e}"))?;
    let metadata = RunMetadata {
        n: config.n,
        rho: config.rho,
        box2: state.box2,
        dt: config.dt,
        temperature: config.temperature,
        eq_steps: config.eq_steps,
        steps: config.steps,
        sample_every: config.sample_every,
        seed: config.seed,
    };
    let mut run_file = BufWriter::new(
        File::create(config.out.join("run.json"))
            .map_err(|e| format!("cannot create run.json: {e}"))?,
    );
    write_run_json(&mut run_file, &metadata).map_err(|e| format!("cannot write run.json: {e}"))?;
    run_file
        .flush()
        .map_err(|e| format!("cannot flush run.json: {e}"))?;
    let mut trajectory = BufWriter::new(
        File::create(config.out.join("traj.jsonl"))
            .map_err(|e| format!("cannot create traj.jsonl: {e}"))?,
    );
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    writeln!(stdout, "t\tE_pot\tE_kin").map_err(|e| e.to_string())?;
    for step in 1..=config.steps {
        velocity_verlet_step(&mut state, config.dt)?;
        if step % config.sample_every == 0 {
            let frame = SavedFrame::from_state(step, config.dt, &state)?;
            write_frame_jsonl(&mut trajectory, &frame)
                .map_err(|e| format!("cannot write traj.jsonl: {e}"))?;
            writeln!(
                stdout,
                "{:.17e}\t{:.17e}\t{:.17e}",
                frame.t, frame.e_pot, frame.e_kin
            )
            .map_err(|e| e.to_string())?;
        }
    }
    trajectory
        .flush()
        .map_err(|e| format!("cannot flush traj.jsonl: {e}"))?;
    stdout.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    if let Err(message) = run(config) {
        eprintln!("{message}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
