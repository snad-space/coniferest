use num_traits::{AsPrimitive, Float};
use numpy::prelude::*;
use numpy::{Element, PyArrayDyn, PyUntypedArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Per-type constants needed by [average_path_length], baked in at
/// monomorphization time so no runtime f64-to-`T` cast is needed.
pub(crate) trait AplConsts {
    const EULER_GAMMA: Self;
    const TWO: Self;
    const TWELVE: Self;
}

impl AplConsts for f32 {
    const EULER_GAMMA: Self = std::f32::consts::EULER_GAMMA;
    const TWO: Self = 2.0;
    const TWELVE: Self = 12.0;
}

impl AplConsts for f64 {
    const EULER_GAMMA: Self = std::f64::consts::EULER_GAMMA;
    const TWO: Self = 2.0;
    const TWELVE: Self = 12.0;
}

/// Average path length of an unsuccessful search in a binary search tree
/// with `n` elements, see Liu et al. 2008. It is used for both tree building
/// and score evaluation, the Python counterpart is
/// `coniferest.utils.average_path_length`.
pub(crate) fn average_path_length<N, T>(n: N) -> T
where
    N: AsPrimitive<T>,
    T: Float + AplConsts + 'static,
{
    let n: T = n.as_();
    if n <= T::one() {
        T::zero()
    } else {
        T::TWO
            * (n.ln() + T::EULER_GAMMA + T::one() / (T::TWO * n)
                - T::one() / (T::TWELVE * n * n)
                - T::one())
    }
}

/// Python wrapper for [average_path_length]: accepts a numpy array of any
/// numeric dtype and any shape. Float32/float64 input keeps its dtype;
/// every other numeric dtype is computed and returned as float64.
///
/// Dispatch is a single match on `dtype.char()`, mirroring the numpy
/// type-char dispatch used elsewhere (e.g. light-curve's band lookup):
/// `itemsize()` is only consulted to disambiguate the platform-dependent
/// `'l'`/`'L'` (C `long`/`unsigned long`) codes.
#[pyfunction(name = "average_path_length")]
pub(crate) fn average_path_length_py<'py>(
    py: Python<'py>,
    n: &Bound<'py, PyUntypedArray>,
) -> PyResult<Py<PyAny>> {
    let dtype = n.dtype();
    match dtype.char() as char {
        'f' => average_path_length_generic::<f32, f32>(py, n),
        'd' => average_path_length_generic::<f64, f64>(py, n),
        'b' => average_path_length_generic::<i8, f64>(py, n),
        'h' => average_path_length_generic::<i16, f64>(py, n),
        'i' => average_path_length_generic::<i32, f64>(py, n),
        'l' => match dtype.itemsize() {
            4 => average_path_length_generic::<i32, f64>(py, n),
            8 => average_path_length_generic::<i64, f64>(py, n),
            _ => Err(unsupported_dtype_err(&dtype)),
        },
        'q' => average_path_length_generic::<i64, f64>(py, n),
        'B' => average_path_length_generic::<u8, f64>(py, n),
        'H' => average_path_length_generic::<u16, f64>(py, n),
        'I' => average_path_length_generic::<u32, f64>(py, n),
        'L' => match dtype.itemsize() {
            4 => average_path_length_generic::<u32, f64>(py, n),
            8 => average_path_length_generic::<u64, f64>(py, n),
            _ => Err(unsupported_dtype_err(&dtype)),
        },
        'Q' => average_path_length_generic::<u64, f64>(py, n),
        _ => Err(unsupported_dtype_err(&dtype)),
    }
}

fn unsupported_dtype_err(dtype: &Bound<'_, numpy::PyArrayDescr>) -> PyErr {
    PyTypeError::new_err(format!("average_path_length: unsupported dtype '{dtype}'"))
}

fn average_path_length_generic<'py, N, T>(
    py: Python<'py>,
    n: &Bound<'py, PyUntypedArray>,
) -> PyResult<Py<PyAny>>
where
    N: Element + AsPrimitive<T>,
    T: Element + Float + AplConsts + 'static,
{
    let typed = n.cast::<PyArrayDyn<N>>()?;
    let result = typed
        .readonly()
        .as_array()
        .mapv(average_path_length::<N, T>);
    Ok(PyArrayDyn::from_owned_array(py, result).into_any().unbind())
}
