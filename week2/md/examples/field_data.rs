use std::f64::consts::PI;

use md::{lennard_jones_energy, lennard_jones_radial_force};

const HALF_WIDTH: f64 = 2.25;
const GRID_SIZE: usize = 181;

fn main() {
    println!("kind,x,y,energy,fx,fy");

    for iy in 0..GRID_SIZE {
        let y = -HALF_WIDTH + 2.0 * HALF_WIDTH * iy as f64 / (GRID_SIZE - 1) as f64;
        for ix in 0..GRID_SIZE {
            let x = -HALF_WIDTH + 2.0 * HALF_WIDTH * ix as f64 / (GRID_SIZE - 1) as f64;
            let r = x.hypot(y);
            if r <= 1e-12 {
                println!("grid,{x:.17e},{y:.17e},NaN,,");
            } else {
                println!("grid,{x:.17e},{y:.17e},{:.17e},,", lennard_jones_energy(r));
            }
        }
    }

    for r in [0.72, 0.92, 1.34, 1.62, 1.95] {
        let radial_force = lennard_jones_radial_force(r);
        for index in 0..16 {
            let angle = 2.0 * PI * index as f64 / 16.0;
            let (sin, cos) = angle.sin_cos();
            let (x, y) = (r * cos, r * sin);
            println!(
                "arrow,{x:.17e},{y:.17e},,{:.17e},{:.17e}",
                radial_force * cos,
                radial_force * sin
            );
        }
    }
}
