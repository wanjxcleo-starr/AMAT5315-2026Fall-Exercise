use ising::{Lattice, SplitMix64, temperature_grid};
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;

const USAGE: &str = "Usage: ising --update metropolis --l L --t-from T --t-to T --t-step DT \
    --discard N --measure N --seed SEED --out DIR [--every N]\n\
    One step is L*L Metropolis proposals sampled uniformly with replacement.\n\
    --every defaults to 0 (no spin frames); all other options are required.";

struct Args {
    side: usize,
    temperatures: Vec<f64>,
    discard: usize,
    measure: usize,
    seed: u64,
    every: usize,
    out: PathBuf,
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
        .map_err(|err| format!("invalid {name}: {err}"))
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut values = HashMap::new();
    let mut input = std::env::args().skip(1);
    while let Some(name) = input.next() {
        if name == "--help" || name == "-h" {
            return Ok(None);
        }
        if !matches!(
            name.as_str(),
            "--update"
                | "--l"
                | "--t-from"
                | "--t-to"
                | "--t-step"
                | "--discard"
                | "--measure"
                | "--seed"
                | "--every"
                | "--out"
        ) {
            return Err(format!("unknown option {name}\n{USAGE}"));
        }
        let value = input
            .next()
            .ok_or_else(|| format!("missing value for {name}"))?;
        if values.insert(name.clone(), value).is_some() {
            return Err(format!("duplicate option {name}"));
        }
    }

    let update: String = required(&values, "--update")?;
    if update != "metropolis" {
        return Err(format!("unsupported --update {update:?}; use metropolis"));
    }
    let side = required(&values, "--l")?;
    if side < 2 {
        return Err("--l must be at least 2".into());
    }
    let from: f64 = required(&values, "--t-from")?;
    let to: f64 = required(&values, "--t-to")?;
    let step: f64 = required(&values, "--t-step")?;
    if from <= 0.0 || to <= 0.0 {
        return Err("temperatures must be greater than 0".into());
    }
    let temperatures = temperature_grid(from, to, step)?;
    let discard: usize = required(&values, "--discard")?;
    let measure: usize = required(&values, "--measure")?;
    if measure == 0 {
        return Err("--measure must be at least 1".into());
    }
    let steps_per_temperature = discard
        .checked_add(measure)
        .ok_or("discard + measure is too large")?;
    if (steps_per_temperature as u128) * (temperatures.len() as u128) > u64::MAX as u128 {
        return Err("too many steps for a cumulative sweep count".into());
    }
    let seed = required(&values, "--seed")?;
    let every = match values.get("--every") {
        Some(value) => value
            .parse()
            .map_err(|err| format!("invalid --every: {err}"))?,
        None => 0,
    };
    let out: String = required(&values, "--out")?;
    if out.is_empty() {
        return Err("--out must name a folder".into());
    }
    Ok(Some(Args {
        side,
        temperatures,
        discard,
        measure,
        seed,
        every,
        out: PathBuf::from(out),
    }))
}

fn write_run_metadata(args: &Args) -> Result<(), String> {
    let path = args.out.join("run.json");
    let file = File::create(&path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut file = BufWriter::new(file);
    write!(
        file,
        "{{\"L\":{},\"update\":\"metropolis\",\"t_grid\":[",
        args.side
    )
    .map_err(|err| err.to_string())?;
    for (index, temperature) in args.temperatures.iter().enumerate() {
        if index > 0 {
            write!(file, ",").map_err(|err| err.to_string())?;
        }
        write!(file, "{temperature}").map_err(|err| err.to_string())?;
    }
    writeln!(
        file,
        "],\"discard\":{},\"measure\":{},\"seed\":{},\"sample_every\":1,\"time_unit\":\"sweep\"}}",
        args.discard, args.measure, args.seed
    )
    .map_err(|err| err.to_string())?;
    file.flush().map_err(|err| err.to_string())
}

fn write_frame(
    file: &mut BufWriter<File>,
    side: usize,
    temperature: f64,
    sweep: u64,
    lattice: &Lattice,
) -> std::io::Result<()> {
    write!(
        file,
        "{{\"L\":{side},\"T\":{temperature:.6},\"sweep\":{sweep},\"m\":{:.6},\"spins\":[",
        lattice.mean_spin()
    )?;
    for (index, spin) in lattice.spins().iter().enumerate() {
        if index > 0 {
            write!(file, ",")?;
        }
        write!(file, "{spin}")?;
    }
    writeln!(file, "]}}")
}

fn simulate(args: Args) -> Result<(), String> {
    let mut lattice = Lattice::all_up(args.side)?;
    fs::create_dir_all(&args.out).map_err(|err| format!("{}: {err}", args.out.display()))?;
    write_run_metadata(&args)?;
    let series_path = args.out.join("series.jsonl");
    let spins_path = args.out.join("spins.jsonl");
    let mut series = BufWriter::new(
        File::create(&series_path).map_err(|err| format!("{}: {err}", series_path.display()))?,
    );
    let mut spins = BufWriter::new(
        File::create(&spins_path).map_err(|err| format!("{}: {err}", spins_path.display()))?,
    );

    let mut rng = SplitMix64::new(args.seed);
    let mut total_sweep = 0_u64;
    let proposals_per_temperature =
        (args.side as u128) * (args.side as u128) * ((args.discard + args.measure) as u128);
    println!("T\tmean_abs_M\tacceptance_rate");

    for &temperature in &args.temperatures {
        let mut accepted = 0_u128;
        for _ in 0..args.discard {
            accepted += lattice.metropolis_sweep(temperature, &mut rng) as u128;
            total_sweep += 1;
        }
        let mut absolute_magnetization_sum = 0.0;
        for sweep in 1..=args.measure {
            accepted += lattice.metropolis_sweep(temperature, &mut rng) as u128;
            total_sweep += 1;
            let magnetization = lattice.mean_spin();
            absolute_magnetization_sum += magnetization.abs();
            writeln!(
                series,
                "{{\"L\":{},\"T\":{temperature:.6},\"sweep\":{sweep},\"M\":{magnetization:.6},\"E\":{:.6}}}",
                args.side,
                lattice.energy_per_site()
            )
            .map_err(|err| format!("{}: {err}", series_path.display()))?;
            if args.every > 0 && sweep % args.every == 0 {
                write_frame(&mut spins, args.side, temperature, total_sweep, &lattice)
                    .map_err(|err| format!("{}: {err}", spins_path.display()))?;
            }
        }
        println!(
            "{temperature:.6}\t{:.6}\t{:.6}",
            absolute_magnetization_sum / args.measure as f64,
            accepted as f64 / proposals_per_temperature as f64
        );
    }
    series
        .flush()
        .map_err(|err| format!("{}: {err}", series_path.display()))?;
    spins
        .flush()
        .map_err(|err| format!("{}: {err}", spins_path.display()))
}

fn main() {
    let result = match parse_args() {
        Ok(Some(args)) => simulate(args),
        Ok(None) => {
            println!("{USAGE}");
            Ok(())
        }
        Err(err) => Err(err),
    };
    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(2);
    }
}
