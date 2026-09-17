use md::{lennard_jones_energy, lennard_jones_radial_force};

fn main() {
    const H: f64 = 1e-5;
    let equilibrium = 2.0_f64.powf(1.0 / 6.0);

    println!("Lennard-Jones radial force comparison (epsilon = sigma = 1; no cutoff)");
    println!("U(r) = 4(r^-12 - r^-6)");
    println!("Numerical F(r) = -[U(r + h) - U(r - h)] / (2h)");
    println!("h = {H:.0e}");
    println!("Equilibrium separation: 2^(1/6) = {equilibrium:.17e}");
    println!(
        "r                     analytic force        numerical force       absolute difference"
    );

    for r in [0.9, 1.0, equilibrium, 1.5, 2.0, 3.0] {
        let numerical_force =
            -(lennard_jones_energy(r + H) - lennard_jones_energy(r - H)) / (2.0 * H);
        let analytic_force = lennard_jones_radial_force(r);
        let absolute_difference = (analytic_force - numerical_force).abs();
        println!(
            "{r:.17e}  {analytic_force:+.17e}  {numerical_force:+.17e}  {absolute_difference:.17e}"
        );
    }

    println!(
        "Uncertainty: central-difference truncation is O(h^2); subtracting nearby energies also incurs floating-point cancellation."
    );
    println!(
        "At equilibrium the true force is zero, so use absolute rather than relative difference there."
    );
}
