use spectral_fluid::{EqualWeightRk4, Euler, ExplicitMidpoint, Integrator, Rk4};

fn scalar_rate(state: &[f64]) -> Result<Vec<f64>, String> {
    Ok(state.iter().map(|value| -2.0 * value).collect())
}

#[test]
fn equal_weight_rk4_is_the_intentional_second_order_comparison() {
    let state = [1.0];
    let dt: f64 = 0.1;
    let z = -2.0 * dt;
    let expected = 1.0 + z + z.powi(2) / 2.0 + 3.0 * z.powi(3) / 16.0 + z.powi(4) / 16.0;
    let mut rate = scalar_rate;

    let actual = EqualWeightRk4.step(&state, dt, &mut rate).unwrap();

    assert!((actual[0] - expected).abs() < 1.0e-15);
}

#[test]
fn shared_integrators_use_the_expected_stability_polynomials() {
    let state = [1.0];
    let dt = 0.1;
    let z = -2.0 * dt;

    let cases: [(&dyn Integrator<f64>, f64); 3] = [
        (&Euler, 1.0 + z),
        (&ExplicitMidpoint, 1.0 + z + z.powi(2) / 2.0),
        (
            &Rk4,
            1.0 + z + z.powi(2) / 2.0 + z.powi(3) / 6.0 + z.powi(4) / 24.0,
        ),
    ];

    for (integrator, expected) in cases {
        let mut rate = scalar_rate;
        let actual = integrator.step(&state, dt, &mut rate).unwrap();
        assert!((actual[0] - expected).abs() < 1.0e-15);
    }
}
