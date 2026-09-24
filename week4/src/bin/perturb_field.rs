use spectral_fluid::{FieldDocument, add_vorticity_ripple};

fn main() {
    if let Err(message) = run() {
        eprintln!("perturb-field: {message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    if std::env::args().len() != 1 {
        return Err("this Part 3 helper takes no command-line arguments".into());
    }
    let mut document: FieldDocument = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| format!("stdin: {error}"))?;
    (document.u, document.v) =
        add_vorticity_ripple(document.n, &document.u, &document.v, -7.0e-5, 3, 4)?;
    serde_json::to_writer(std::io::stdout().lock(), &document)
        .map_err(|error| error.to_string())?;
    println!();
    Ok(())
}
