use boxcars_frames::Collector;
use numpy::pyo3::IntoPy;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::{exceptions, wrap_pyfunction};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[pyfunction]
fn parse_replay<'p>(py: Python<'p>, data: &[u8]) -> PyResult<PyObject> {
    let replay = serde_json::to_value(replay_from_data(data)?).map_err(to_py_error)?;
    Ok(convert_to_py(py, &replay))
}

fn replay_from_data(data: &[u8]) -> PyResult<boxcars::Replay> {
    boxcars::ParserBuilder::new(data)
        .must_parse_network_data()
        .on_error_check_crc()
        .parse()
        .map_err(to_py_error)
}

#[pymodule]
fn boxcars_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(parse_replay))?;
    m.add_wrapped(wrap_pyfunction!(get_replay_meta_and_numpy_ndarray))?;
    Ok(())
}

fn to_py_error<E: std::error::Error>(e: E) -> PyErr {
    PyErr::new::<exceptions::PyException, _>(format!("{}", e))
}

fn convert_to_py(py: Python, value: &Value) -> PyObject {
    match value {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => match n {
            n if n.is_u64() => n.as_u64().unwrap().into_py(py),
            n if n.is_i64() => n.as_i64().unwrap().into_py(py),
            n if n.is_f64() => n.as_f64().unwrap().into_py(py),
            _ => py.None(),
        },
        Value::String(s) => s.into_py(py),
        Value::Array(list) => {
            let list: Vec<PyObject> = list.iter().map(|e| convert_to_py(py, e)).collect();
            list.into_py(py)
        }
        Value::Object(m) => {
            let mut map = BTreeMap::new();
            m.iter().for_each(|(k, v)| {
                map.insert(k, convert_to_py(py, v));
            });
            map.into_py(py)
        }
    }
}

#[pyfunction]
fn get_replay_meta_and_numpy_ndarray<'p>(py: Python<'p>, filepath: PathBuf) -> PyResult<PyObject> {
    let data = std::fs::read(filepath.as_path()).map_err(to_py_error)?;
    let replay = replay_from_data(&data)?;

    let handle_frames_exception = PyErr::new::<exceptions::PyException, _>;
    let collector = boxcars_frames::NDArrayCollector::<f32>::with_jump_availabilities()
        .process_replay(&replay)
        .map_err(handle_frames_exception)?;
    let (replay_meta, rust_nd_array) = collector
        .get_meta_and_ndarray()
        .map_err(handle_frames_exception)?;
    let python_replay_meta = convert_to_py(
        py,
        &serde_json::to_value(&replay_meta).map_err(to_py_error)?,
    );
    let python_nd_array = rust_nd_array.into_pyarray(py);
    Ok((python_replay_meta, python_nd_array).into_py(py))
}

fn get_replay_meta<'p>(py: Python<'p>, filepath: PathBuf) -> PyResult<PyObject> {
    let data = std::fs::read(filepath.as_path()).map_err(to_py_error)?;
    let replay = replay_from_data(&data)?;

    let processor = boxcars_frames::ReplayProcessor::new(&replay);

    let replay_meta = processor
        .get_replay_meta()
        .map_err(PyErr::new::<exceptions::PyException, _>)?;
    Ok(convert_to_py(
        py,
        &serde_json::to_value(&replay_meta).map_err(to_py_error)?,
    ))
}
