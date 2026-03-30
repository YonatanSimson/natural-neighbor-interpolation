#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run discrete natural-neighbor interpolation on the GPU.
 *
 * known_x/y/z: coordinates of the num_known input points in grid-index space.
 * known_values: values at those points.
 * interp_values: ni*nj*nk output array (modified in place, must be contiguous).
 *
 * Returns 0 on success, non-zero on CUDA error.
 */
int cuda_griddata(
        const double* known_x,
        const double* known_y,
        const double* known_z,
        const double* known_values,
        int num_known,
        double* interp_values,
        size_t ni,
        size_t nj,
        size_t nk);

#ifdef __cplusplus
}
#endif

#endif  /* CUDA_KERNELS_H_ */
