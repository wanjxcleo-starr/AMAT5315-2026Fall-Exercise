use serde::Serialize;
use spectral_fluid::{FieldDocument, FlowFields, FlowSolver, Method, enstrophy, kinetic_energy};
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;

const USAGE: &str = "Usage: fluid --method {euler|rk2|rk4} --nu NU --dt DT --t-end T \
    --every T --out DIR";

struct Args {
    method: Method,
    method_name: String,
    viscosity: f64,
    dt: f64,
    t_end: f64,
    steps: usize,
    every: f64,
    snapshot_stride: usize,
    out: PathBuf,
}

#[derive(Serialize)]
struct RunMetadata<'a> {
    case: &'a str,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[i32; 2]>,
    method: &'a str,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(message) => {
            eprintln!("fluid: {message}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let input: FieldDocument = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| format!("stdin: {error}"))?;
    let mut solver = FlowSolver::from_velocity(input.n, args.viscosity, &input.u, &input.v)?;

    fs::create_dir_all(&args.out).map_err(|error| format!("{}: {error}", args.out.display()))?;
    write_run_metadata(&args, &input)?;
    let fields_path = args.out.join("fields.jsonl");
    let mut frame_file = BufWriter::new(
        File::create(&fields_path)
            .map_err(|error| format!("{}: {error}", fields_path.display()))?,
    );

    println!("t\tE\tZ");
    for step in 0..=args.steps {
        let time = step as f64 * args.dt;
        let fields = solver.fields();
        let energy = kinetic_energy(&fields.u, &fields.v)?;
        let z = enstrophy(&fields.omega)?;
        if !energy.is_finite() {
            print_table_line(time, energy, z);
            frame_file.flush().map_err(|error| error.to_string())?;
            std::io::stdout()
                .flush()
                .map_err(|error| error.to_string())?;
            return Ok(false);
        }
        if step % args.snapshot_stride == 0 {
            print_table_line(time, energy, z);
            write_frame(&mut frame_file, time, step, &fields).map_err(|error| error.to_string())?;
        }
        if step < args.steps {
            solver.step(args.method, args.dt)?;
        }
    }
    frame_file.flush().map_err(|error| error.to_string())?;
    Ok(true)
}

fn parse_args() -> Result<Args, String> {
    let mut arguments = std::env::args().skip(1);
    let mut values = HashMap::new();
    while let Some(name) = arguments.next() {
        if name == "--help" || name == "-h" {
            println!("{USAGE}");
            std::process::exit(0);
        }
        if !matches!(
            name.as_str(),
            "--method" | "--nu" | "--dt" | "--t-end" | "--every" | "--out"
        ) {
            return Err(format!("unknown option {name}\n{USAGE}"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {name}"))?;
        if values.insert(name.clone(), value).is_some() {
            return Err(format!("duplicate option {name}"));
        }
    }
    let method_name: String = required(&values, "--method")?;
    let method = match method_name.as_str() {
        "euler" => Method::Euler,
        "rk2" => Method::MidpointRk2,
        "rk4" => Method::Rk4,
        _ => return Err("--method must be euler, rk2, or rk4".into()),
    };
    let viscosity: f64 = required(&values, "--nu")?;
    if !viscosity.is_finite() || viscosity < 0.0 {
        return Err("--nu must be finite and non-negative".into());
    }
    let dt: f64 = required(&values, "--dt")?;
    if !dt.is_finite() || dt <= 0.0 {
        return Err("--dt must be finite and greater than zero".into());
    }
    let t_end: f64 = required(&values, "--t-end")?;
    if !t_end.is_finite() || t_end < 0.0 {
        return Err("--t-end must be finite and non-negative".into());
    }
    let steps = integer_step_count(t_end, dt)?;
    let every: f64 = required(&values, "--every")?;
    if !every.is_finite() || every <= 0.0 {
        return Err("--every must be finite and greater than zero".into());
    }
    let stride = (every / dt).round();
    if stride < 1.0 || stride > usize::MAX as f64 {
        return Err("round(--every/--dt) must be at least one representable step".into());
    }
    let out: String = required(&values, "--out")?;
    if out.is_empty() {
        return Err("--out must name a folder".into());
    }
    Ok(Args {
        method,
        method_name,
        viscosity,
        dt,
        t_end,
        steps,
        every,
        snapshot_stride: stride as usize,
        out: PathBuf::from(out),
    })
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

fn integer_step_count(t_end: f64, dt: f64) -> Result<usize, String> {
    let ratio = t_end / dt;
    let nearest = ratio.round();
    let tolerance = 128.0 * f64::EPSILON * ratio.abs().max(1.0);
    if (ratio - nearest).abs() > tolerance || nearest > usize::MAX as f64 {
        return Err(
            "--t-end must be an integer multiple of --dt within floating-point tolerance".into(),
        );
    }
    Ok(nearest as usize)
}

fn write_run_metadata(args: &Args, input: &FieldDocument) -> Result<(), String> {
    let metadata = RunMetadata {
        case: &input.case,
        n: input.n,
        seed: input.seed,
        k_band: input.k_band,
        method: &args.method_name,
        nu: args.viscosity,
        dt: args.dt,
        t_end: args.t_end,
        snapshot_every: args.every,
    };
    let path = args.out.join("run.json");
    let mut file = BufWriter::new(
        File::create(&path).map_err(|error| format!("{}: {error}", path.display()))?,
    );
    serde_json::to_writer(&mut file, &metadata).map_err(|error| error.to_string())?;
    writeln!(file).map_err(|error| error.to_string())?;
    file.flush().map_err(|error| error.to_string())
}

fn print_table_line(time: f64, energy: f64, z: f64) {
    println!("{time}\t{energy:.12}\t{z:.12}");
}

fn write_frame(
    file: &mut BufWriter<File>,
    time: f64,
    step: usize,
    fields: &FlowFields,
) -> std::io::Result<()> {
    write!(file, "{{\"t\":{time},\"step\":{step},\"u\":[")?;
    write_array(file, &fields.u)?;
    write!(file, "],\"v\":[")?;
    write_array(file, &fields.v)?;
    write!(file, "],\"omega\":[")?;
    write_array(file, &fields.omega)?;
    writeln!(file, "]}}")
}

fn write_array(file: &mut impl Write, values: &[f64]) -> std::io::Result<()> {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            write!(file, ",")?;
        }
        write!(file, "{value:.6}")?;
    }
    Ok(())
}
