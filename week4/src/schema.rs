use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct FieldDocument {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[i32; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}
