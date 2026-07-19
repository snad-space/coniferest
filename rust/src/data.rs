use numpy::PyReadonlyArray2;
use pyo3::FromPyObject;

/// Input data: 2-D numpy array of features, one sample per row.
#[derive(FromPyObject)]
pub(crate) enum Data<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}
