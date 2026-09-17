pub mod dimer;
pub mod fluid;
pub mod record;

pub fn greeting() -> &'static str {
    "Hello, world!"
}

fn assert_valid_separation(r: f64) {
    assert!(
        r.is_finite() && r > 0.0,
        "separation must be positive and finite"
    );
}

/// Unshifted Lennard-Jones pair energy with epsilon = sigma = 1 and no cutoff.
///
/// Panics if `r` is not positive and finite.
pub fn lennard_jones_energy(r: f64) -> f64 {
    assert_valid_separation(r);
    let inverse_r6 = (1.0 / r).powi(6);
    4.0 * inverse_r6 * (inverse_r6 - 1.0)
}

/// Radial force `-dU/dr`, positive when it increases particle separation.
///
/// Panics if `r` is not positive and finite.
pub fn lennard_jones_radial_force(r: f64) -> f64 {
    assert_valid_separation(r);
    let inverse_r6 = (1.0 / r).powi(6);
    24.0 * inverse_r6 * (2.0 * inverse_r6 - 1.0) / r
}

#[cfg(test)]
mod tests {
    use super::{greeting, lennard_jones_energy, lennard_jones_radial_force};

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn lennard_jones_known_values_and_force_sign() {
        assert_eq!(lennard_jones_energy(1.0), 0.0);
        assert_eq!(lennard_jones_radial_force(1.0), 24.0);
        assert_eq!(lennard_jones_energy(2.0), -63.0 / 1024.0);
        assert_eq!(lennard_jones_radial_force(2.0), -93.0 / 512.0);

        let minimum = 2.0_f64.powf(1.0 / 6.0);
        assert!((lennard_jones_energy(minimum) + 1.0).abs() < 1e-12);
        assert!(lennard_jones_radial_force(minimum).abs() < 1e-12);
    }

    #[test]
    fn radial_force_matches_negative_energy_derivative() {
        let step = 1e-6;
        for r in [0.9, 1.5, 3.0] {
            let numerical_force =
                -(lennard_jones_energy(r + step) - lennard_jones_energy(r - step)) / (2.0 * step);
            let analytic_force = lennard_jones_radial_force(r);
            assert!(
                (analytic_force - numerical_force).abs() < 1e-6 * analytic_force.abs().max(1.0)
            );
        }
    }

    #[test]
    fn invalid_separations_panic() {
        for r in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(std::panic::catch_unwind(|| lennard_jones_energy(r)).is_err());
            assert!(std::panic::catch_unwind(|| lennard_jones_radial_force(r)).is_err());
        }
    }
}
