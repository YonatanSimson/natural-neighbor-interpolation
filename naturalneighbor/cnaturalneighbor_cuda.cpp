#include <cstddef>
#include <vector>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "cuda_kernels.h"

static char module_docstring[] =
        "Discrete natural neighbor interpolation in 3D (CUDA backend).";

static char griddata_docstring[] =
        "Calculate the natural neighbor interpolation of a dataset.";

static PyObject* ccudanaturalneighbor_griddata(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {
        {"griddata", ccudanaturalneighbor_griddata, METH_VARARGS, griddata_docstring},
        {NULL, NULL, 0, NULL}};

/*
The macros below are for backward compatibility with Python 2.
Here, the functions used to creating and initializing modules
differ in both their names and their signatures.

Module initialization
    2: init<name>, void function (does not return a value)
    3: PyInit_<name>, returns either PyObject/NULL for success/failure
Module creation
    2: Py_InitModule3, takes three input arguments
    3: PyModule_Create, takes input args in a single struct

For more, see http://python3porting.com/cextensions.html
*/
#if PY_MAJOR_VERSION >= 3
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, doc, methods)               \
    static struct PyModuleDef module = {              \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, \
    };                                                \
    ob = PyModule_Create(&module);
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
#define MOD_DEF(ob, name, doc, methods) ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(ccudanaturalneighbor) {
    PyObject* m;

    MOD_DEF(m, "ccudanaturalneighbor", module_docstring, module_methods);

    if (m == NULL) {
        return MOD_ERROR_VAL;
    }

    import_array();

    return MOD_SUCCESS_VAL(m);
}

static PyObject* ccudanaturalneighbor_griddata(PyObject* self, PyObject* args) {
    PyArrayObject *known_points_ijk, *known_values, *interp_values;

    if (!PyArg_ParseTuple(
                args,
                "O!O!O!",
                &PyArray_Type,
                &known_points_ijk,
                &PyArray_Type,
                &known_values,
                &PyArray_Type,
                &interp_values)) {
        return NULL;
    }

    npy_intp* known_points_ijk_dims = PyArray_DIMS(known_points_ijk);
    int num_known = static_cast<int>(known_points_ijk_dims[0]);

    npy_intp* interp_values_shape = PyArray_DIMS(interp_values);
    std::size_t ni = static_cast<std::size_t>(interp_values_shape[0]);
    std::size_t nj = static_cast<std::size_t>(interp_values_shape[1]);
    std::size_t nk = static_cast<std::size_t>(interp_values_shape[2]);

    double* interp_values_ptr = static_cast<double*>(PyArray_GETPTR1(interp_values, 0));

    if (num_known == 0) {
        Py_RETURN_NONE;
    }

    /* Unpack the (N, 3) array into separate x/y/z vectors for the kernel. */
    std::vector<double> kx(static_cast<std::size_t>(num_known));
    std::vector<double> ky(static_cast<std::size_t>(num_known));
    std::vector<double> kz(static_cast<std::size_t>(num_known));
    std::vector<double> kv(static_cast<std::size_t>(num_known));

    for (int i = 0; i < num_known; i++) {
        kx[static_cast<std::size_t>(i)] = *(double*)PyArray_GETPTR2(known_points_ijk, i, 0);
        ky[static_cast<std::size_t>(i)] = *(double*)PyArray_GETPTR2(known_points_ijk, i, 1);
        kz[static_cast<std::size_t>(i)] = *(double*)PyArray_GETPTR2(known_points_ijk, i, 2);
        kv[static_cast<std::size_t>(i)] = *(double*)PyArray_GETPTR1(known_values, i);
    }

    int err = cuda_griddata(
            kx.data(),
            ky.data(),
            kz.data(),
            kv.data(),
            num_known,
            interp_values_ptr,
            ni,
            nj,
            nk);

    if (err != 0) {
        PyErr_SetString(PyExc_RuntimeError, "CUDA griddata failed (see stderr for details)");
        return NULL;
    }

    Py_RETURN_NONE;
}
