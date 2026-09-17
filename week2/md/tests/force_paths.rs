use md::fluid::{Box2, FluidState, forces_and_potential, forces_and_potential_cells, wrapped};

fn compare_paths(label: &str, state: &FluidState) -> (f64, f64) {
    let (naive_forces, naive_energy) = forces_and_potential(state).unwrap();
    let (cell_forces, cell_energy) = forces_and_potential_cells(state).unwrap();
    let max_force_difference = naive_forces
        .iter()
        .zip(&cell_forces)
        .flat_map(|(a, b)| (0..2).map(move |axis| (a[axis] - b[axis]).abs()))
        .fold(0.0_f64, f64::max);
    let energy_difference = (naive_energy - cell_energy).abs();
    let max_abs_force = naive_forces
        .iter()
        .flatten()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let force_scale = max_abs_force.max(1.0);
    let force_tolerance = 1e-10 * force_scale;
    let energy_tolerance = 1e-10 * naive_energy.abs().max(1.0);
    println!(
        "\nCASE {label}: n={}, box=[{:.17e}, {:.17e}]",
        state.pos.len(),
        state.box2.lx,
        state.box2.ly
    );
    println!("E_pot naive = {naive_energy:+.17e}");
    println!("E_pot cells = {cell_energy:+.17e}");
    println!("max |ΔE_pot| = {energy_difference:.17e}; tolerance = {energy_tolerance:.17e}");
    println!(
        "max |F_naive component| = {max_abs_force:.17e}; tolerance scale = {force_scale:.17e}"
    );
    println!(
        "max |ΔF_component| = {max_force_difference:.17e}; tolerance = {force_tolerance:.17e}"
    );
    println!("atom  x  y  naive_Fx  naive_Fy  cells_Fx  cells_Fy  |ΔFx|  |ΔFy|");
    for (index, ((point, naive), cells)) in state
        .pos
        .iter()
        .zip(&naive_forces)
        .zip(&cell_forces)
        .enumerate()
    {
        println!(
            "{index:>4}  {:+.17e}  {:+.17e}  {:+.17e}  {:+.17e}  {:+.17e}  {:+.17e}  {:.17e}  {:.17e}",
            point[0],
            point[1],
            naive[0],
            naive[1],
            cells[0],
            cells[1],
            (naive[0] - cells[0]).abs(),
            (naive[1] - cells[1]).abs()
        );
    }
    assert!(max_force_difference <= force_tolerance);
    assert!(energy_difference <= energy_tolerance);
    (max_force_difference, energy_difference)
}

#[test]
fn perturbed_lattices_match_at_100_and_400_atoms() {
    for n in [100, 400] {
        let mut state = FluidState::new(n, 0.8, 0.5, 2026).unwrap();
        for (index, point) in state.pos.iter_mut().enumerate() {
            let dx = ((index * 37 % 23) as f64 - 11.0) * 0.006;
            let dy = ((index * 19 % 29) as f64 - 14.0) * 0.005;
            point[0] = wrapped(point[0] + dx, state.box2.lx);
            point[1] = wrapped(point[1] + dy, state.box2.ly);
        }
        compare_paths(&format!("perturbed n={n}"), &state);
    }
}

#[test]
fn pairs_across_both_periodic_boundaries_match() {
    let state = FluidState {
        pos: vec![[0.1, 1.0], [8.9, 1.0], [5.0, 0.2], [5.0, 9.0]],
        vel: vec![[0.0; 2]; 4],
        box2: Box2 { lx: 10.0, ly: 10.0 },
    };
    let (_, energy) = forces_and_potential(&state).unwrap();
    assert!(energy < -1.0);
    compare_paths("periodic x/y seams", &state);
}

#[test]
fn cutoff_below_at_and_above_match() {
    for (label, separation) in [
        ("below rc", 2.5 - 1e-9),
        ("at rc", 2.5),
        ("above rc", 2.5 + 1e-9),
    ] {
        let state = FluidState {
            pos: vec![[1.0, 1.0], [1.0 + separation, 1.0]],
            vel: vec![[0.0; 2]; 2],
            box2: Box2 { lx: 10.0, ly: 10.0 },
        };
        let (forces, energy) = forces_and_potential(&state).unwrap();
        if label == "below rc" {
            assert!(energy < 0.0);
            assert!(forces[0][0] != 0.0);
        } else {
            assert_eq!(energy, 0.0);
            assert_eq!(forces, vec![[0.0; 2]; 2]);
        }
        compare_paths(label, &state);
    }
}

#[test]
fn two_cells_per_axis_do_not_duplicate_pairs() {
    let state = FluidState {
        pos: vec![[0.2, 1.0], [4.0, 1.0], [2.4, 0.2], [2.4, 4.0]],
        vel: vec![[0.0; 2]; 4],
        box2: Box2 { lx: 5.1, ly: 5.4 },
    };
    assert_eq!((state.box2.lx / 2.5).floor() as usize, 2);
    assert_eq!((state.box2.ly / 2.5).floor() as usize, 2);
    let (_, energy) = forces_and_potential(&state).unwrap();
    assert!(energy != 0.0);
    compare_paths("two cells per axis", &state);
}

#[test]
fn dilute_box_matches_naive_without_allocating_empty_grid_cells() {
    let state = FluidState::new(100, 1e-8, 0.5, 2026).unwrap();
    compare_paths("dilute n=100", &state);
}
