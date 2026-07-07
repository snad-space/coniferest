use num_traits::AsPrimitive;
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

/// Euler-Mascheroni constant, the same value as np.euler_gamma.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Average path length of an unsuccessful search in a binary search tree
/// with `n` elements, see Liu et al. 2008. It is used for both tree building
/// and score evaluation, the Python counterpart is
/// `coniferest.utils.average_path_length`.
///
/// The value is always computed in f64 and then cast to the output type.
pub(crate) fn average_path_length<N, R>(n: N) -> R
where
    N: AsPrimitive<f64>,
    f64: AsPrimitive<R>,
    R: Copy + 'static,
{
    let n: f64 = n.as_();
    let apl = if n <= 1.0 {
        0.0
    } else {
        2.0 * (n.ln() + EULER_GAMMA + 1.0 / (2.0 * n) - 1.0 / (12.0 * n * n) - 1.0)
    };
    apl.as_()
}

#[derive(FromPyObject)]
pub(crate) enum F64OrArray<'py> {
    // Try array first: a single-element array would extract to a scalar otherwise
    Array(Bound<'py, PyArrayDyn<f64>>),
    Scalar(f64),
}

/// Python wrapper for [average_path_length]: accepts either a float scalar
/// or a float64 array of any shape, and returns the same kind of object.
#[pyfunction(name = "average_path_length")]
pub(crate) fn average_path_length_py<'py>(
    py: Python<'py>,
    n: F64OrArray<'py>,
) -> PyResult<Py<PyAny>> {
    match n {
        F64OrArray::Scalar(n) => Ok(average_path_length::<f64, f64>(n)
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        F64OrArray::Array(n) => {
            let result = n
                .readonly()
                .as_array()
                .mapv(average_path_length::<f64, f64>);
            Ok(PyArrayDyn::from_owned_array(py, result).into_any().unbind())
        }
    }
}
