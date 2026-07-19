//! Supertrait [Float] with all we need from f32 and f64.

pub(crate) trait Float:
    num_traits::Float
    + std::fmt::Display // to use in format!, e.g. for Python exceptions
    // + num_traits::AsPrimitive<f32>
    + rand::distr::uniform::SampleUniform  // crate::tree::builder::SplitAlgorithm
    + numpy::Element  // numpy array element
    + Send + Sync  // for multithreading
    + 'static
{
    /// Euler gamma constant, used in [crate::utils::average_path_length].
    const EULER_GAMMA: Self;
}

impl Float for f32 {
    const EULER_GAMMA: Self = std::f32::consts::EULER_GAMMA;
}

impl Float for f64 {
    const EULER_GAMMA: Self = std::f64::consts::EULER_GAMMA;
}
