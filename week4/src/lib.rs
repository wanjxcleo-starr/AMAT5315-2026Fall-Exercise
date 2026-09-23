mod flow;
mod schema;
mod spectral;

pub use flow::{FlowFields, FlowSolver, Method, enstrophy, kinetic_energy};
pub use schema::FieldDocument;
pub use spectral::SpectralGrid;
