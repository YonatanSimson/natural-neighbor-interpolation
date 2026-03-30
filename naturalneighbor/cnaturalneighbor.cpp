#include <cmath>

#include <vector>
#include <algorithm>
#include <omp.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "kdtree.h"
#include "geometry.h"


typedef geometry::Point<double, 3> Point;

static char module_docstring[] = "Discrete natural neighbor interpolation in 3D.";

static char griddata_docstring[] = "Calculate the natural neighbor interpolation of a dataset.";

static PyObject* cnaturalneighbor_griddata(PyObject* self, PyObject* args);

static PyMethodDef module_methods[] = {
    {"griddata", cnaturalneighbor_griddata, METH_VARARGS, griddata_docstring},
    {NULL, NULL, 0, NULL}
};

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
    #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef module = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&module);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(cnaturalneighbor) {
    PyObject* m;

    MOD_DEF(m, "cnaturalneighbor", module_docstring, module_methods);

    if (m == NULL) {
        return MOD_ERROR_VAL;
    }

    import_array();

    return MOD_SUCCESS_VAL(m);
}

inline long clamp(long val, long min, long max) {
    return std::min(max, std::max(min, val));
}


static PyObject* cnaturalneighbor_griddata(PyObject* self, PyObject* args) {
    PyArrayObject *known_points_ijk, *known_values, *interp_values;

    if (!PyArg_ParseTuple(args, "O!O!O!",
                &PyArray_Type, &known_points_ijk,
                &PyArray_Type, &known_values,
                &PyArray_Type, &interp_values)) {
        return NULL;
    }

    npy_intp* known_points_ijk_dims = PyArray_DIMS(known_points_ijk);
    std::size_t num_known_points = known_points_ijk_dims[0];

    npy_intp* interp_values_shape = PyArray_DIMS(interp_values);
    std::size_t ni = interp_values_shape[0];
    std::size_t nj = interp_values_shape[1];
    std::size_t nk = interp_values_shape[2];

    double* interp_values_ptr = (double*)PyArray_GETPTR1(interp_values, 0);

    kdtree::kdtree<double> *tree = new kdtree::kdtree<double>();
    for (std::size_t i = 0; i < num_known_points; i++) {
        Point p {
            *(double*)PyArray_GETPTR2(known_points_ijk, i, 0),
            *(double*)PyArray_GETPTR2(known_points_ijk, i, 1),
            *(double*)PyArray_GETPTR2(known_points_ijk, i, 2)
        };
        double v {*(double*)PyArray_GETPTR1(known_values, i)};
        tree->add(p, v);
    }
    tree->build();

    std::size_t grid_size = ni * nj * nk;
    auto contribution_counter = new unsigned long[grid_size]();

    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    // Per-thread private accumulators avoid atomic contention entirely.
    // Each thread writes to its own slice; we sum them in a final reduction.
    std::vector<double> thread_interp(nthreads * grid_size, 0.0);
    std::vector<unsigned long> thread_counter(nthreads * grid_size, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* my_interp = thread_interp.data() + tid * grid_size;
        unsigned long* my_counter = thread_counter.data() + tid * grid_size;

        #pragma omp for schedule(dynamic)
        for (long i = 0; i < (long)ni; i++) {
            for (long j = 0; j < (long)nj; j++) {
                for (long k = 0; k < (long)nk; k++) {
                    auto query_point = Point(i, j, k);
                    auto nearest_known_point = tree->nearest_iterative(query_point);

                    double distance_sq_query_to_known = nearest_known_point.distance;
                    int roi_radius = (int)ceil(sqrt(distance_sq_query_to_known));

                    long i_roi_min = clamp(i - roi_radius, 0L, (long)ni - 1);
                    long i_roi_max = clamp(i + roi_radius, 0L, (long)ni - 1);
                    long j_roi_min = clamp(j - roi_radius, 0L, (long)nj - 1);
                    long j_roi_max = clamp(j + roi_radius, 0L, (long)nj - 1);
                    long k_roi_min = clamp(k - roi_radius, 0L, (long)nk - 1);
                    long k_roi_max = clamp(k + roi_radius, 0L, (long)nk - 1);

                    for (long i_roi = i_roi_min; i_roi <= i_roi_max; i_roi++) {
                        double deltai_2 = (double)(i - i_roi)*(i - i_roi);
                        std::size_t indice_i_component = nj*nk*i_roi;
                        for (long j_roi = j_roi_min; j_roi <= j_roi_max; j_roi++) {
                            double deltaj_2 = (double)(j - j_roi)*(j - j_roi);
                            std::size_t indice_j_component = nk*j_roi;
                            for (long k_roi = k_roi_min; k_roi <= k_roi_max; k_roi++) {
                                double deltak_2 = (double)(k - k_roi)*(k - k_roi);
                                double distance_sq_roi_to_known = deltai_2 + deltaj_2 + deltak_2;

                                if (distance_sq_roi_to_known == 0 || distance_sq_roi_to_known < distance_sq_query_to_known) {
                                    std::size_t indice = indice_i_component + indice_j_component + k_roi;
                                    my_interp[indice] += nearest_known_point.value;
                                    my_counter[indice] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce per-thread results into the output arrays
    for (int t = 0; t < nthreads; t++) {
        double* my_interp = thread_interp.data() + t * grid_size;
        unsigned long* my_counter = thread_counter.data() + t * grid_size;
        for (std::size_t idx = 0; idx < grid_size; idx++) {
            interp_values_ptr[idx] += my_interp[idx];
            contribution_counter[idx] += my_counter[idx];
        }
    }

    for (std::size_t i = 0; i < ni; i++) {
        for (std::size_t j = 0; j < nj; j++) {
            for (std::size_t k = 0; k < nk; k++) {
                auto indice = nj*nk*i + nk*j + k;
                if (contribution_counter[indice] != 0) {
                    interp_values_ptr[indice] /= contribution_counter[indice];
                }
            }
        }
    }

    delete[] contribution_counter;
    delete tree;

    Py_RETURN_NONE;
}
