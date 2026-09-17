use md::{lennard_jones_energy, lennard_jones_radial_force};

fn main() {
    const H: f64 = 1e-5;
    let equilibrium = 2.0_f64.powf(1.0 / 6.0);

    println!("Lennard-Jones radial force comparison (epsilon = sigma = 1; no cutoff)");
    println!("U(r) = 4(r^-12 - r^-6)");
    println!("Numerical F(r) = -[U(r + h) - U(r - h)] / (2h)");
    println!("h = {H:.0e}");
    println!("Required separations: 1.0, 1.1, 1.2, 1.5");
    println!("Equilibrium separation: 2^(1/6) = {equilibrium:.17e}");
    println!(
        "r                     analytic force        numerical force       absolute difference"
    );

    for r in [1.0, 1.1, equilibrium, 1.2, 1.5] {
        let numerical_force =
            -(lennard_jones_energy(r + H) - lennard_jones_energy(r - H)) / (2.0 * H);
        let analytic_force = lennard_jones_radial_force(r);
        let absolute_difference = (analytic_force - numerical_force).abs();
        let tolerance = 1e-6 * analytic_force.abs().max(1.0);
        assert!(
            absolute_difference < tolerance,
            "force mismatch at r={r}: difference {absolute_difference} >= {tolerance}"
        );
        println!(
            "{r:.17e}  {analytic_force:+.17e}  {numerical_force:+.17e}  {absolute_difference:.17e}"
        );
    }

    println!(
        "Verification: every row satisfies |F_analytic - F_numerical| < 1e-6 * max(1, |F_analytic|)."
    );
    println!(
        "Uncertainty: central-difference truncation is O(h^2); subtracting nearby energies also incurs floating-point cancellation."
    );
    println!(
        "At equilibrium the true force is zero, so use absolute rather than relative difference there."
    );
}
