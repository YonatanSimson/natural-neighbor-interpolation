#include "cuda_kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

/*
 * One GPU thread per query voxel Q=(qi,qj,qk).
 *
 * The discrete Sibson algorithm:
 *   For each query voxel Q, find its nearest known point P (brute-force NN).
 *   Every output cell C within Q's ROI sphere (dist²(Q,C) < dist²(Q,P),
 *   or Q==C) atomically accumulates value(P) and a count of 1.
 *   Final value at C = interp_sum[C] / count[C].
 *
 * Atomics handle the scatter from multiple Q threads writing to the same C.
 * No octant partitioning is needed — that was a CPU thread-safety trick only.
 */

/*
 * Double-precision atomicAdd requires SM 6.0+.
 * Provide a CAS-based fallback for older architectures.
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ inline double atomicAddDouble(double* addr, double val) {
    unsigned long long int* a = (unsigned long long int*)addr;
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline double atomicAddDouble(double* addr, double val) {
    return atomicAdd(addr, val);
}
#endif

namespace {

__device__ inline long clamp_long(long val, long lo, long hi) {
    long t = (val > lo) ? val : lo;
    return (hi < t) ? hi : t;
}

/*
 * interpolation_kernel — parallel-over-Q
 *
 * Each thread handles one query voxel Q.  It finds Q's nearest known point
 * via brute-force search (fast for N_known < ~10 k), then scatters Q's
 * contribution to every output cell C inside Q's ROI sphere.
 */
__global__ void interpolation_kernel(
        const double* __restrict__ known_x,
        const double* __restrict__ known_y,
        const double* __restrict__ known_z,
        const double* __restrict__ known_values,
        int num_known,
        size_t ni,
        size_t nj,
        size_t nk,
        double* __restrict__ interp_sum,
        unsigned long long* __restrict__ d_count) {

    size_t total = ni * nj * nk;
    size_t q_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q_idx >= total) {
        return;
    }

    /* Query voxel coordinates */
    size_t qk = q_idx % nk;
    size_t qj = (q_idx / nk) % nj;
    size_t qi = q_idx / (nk * nj);

    double dqi = static_cast<double>(qi);
    double dqj = static_cast<double>(qj);
    double dqk = static_cast<double>(qk);

    /* Brute-force nearest-neighbour search over all known points */
    double best_dist_sq = 1.79769313486231570815e+308;  /* DBL_MAX */
    double best_value = 0.0;
    double num_contributions = 1.0;

    for (int p = 0; p < num_known; p++) {
        double di = dqi - known_x[p];
        double dj = dqj - known_y[p];
        double dk = dqk - known_z[p];
        double d2 = di * di + dj * dj + dk * dk;
        if (d2 < best_dist_sq) {
            best_dist_sq = d2;
            best_value = known_values[p];
            num_contributions = 1.0;
        } else if (d2 == best_dist_sq) {
            num_contributions += 1.0;
            best_value = best_value * (num_contributions - 1.0) / num_contributions
                       + known_values[p] / num_contributions;
        }
    }

    /* ROI bounding box, clamped to grid bounds */
    int roi_radius = static_cast<int>(ceil(sqrt(best_dist_sq)));
    long roi_r = static_cast<long>(roi_radius);
    long qi_l = static_cast<long>(qi);
    long qj_l = static_cast<long>(qj);
    long qk_l = static_cast<long>(qk);
    long ni_l = static_cast<long>(ni);
    long nj_l = static_cast<long>(nj);
    long nk_l = static_cast<long>(nk);

    long ci_min = clamp_long(qi_l - roi_r, 0L, ni_l - 1);
    long ci_max = clamp_long(qi_l + roi_r, 0L, ni_l - 1);
    long cj_min = clamp_long(qj_l - roi_r, 0L, nj_l - 1);
    long cj_max = clamp_long(qj_l + roi_r, 0L, nj_l - 1);
    long ck_min = clamp_long(qk_l - roi_r, 0L, nk_l - 1);
    long ck_max = clamp_long(qk_l + roi_r, 0L, nk_l - 1);

    /* Scatter contributions to all output cells C inside Q's ROI sphere */
    for (long ci = ci_min; ci <= ci_max; ci++) {
        double deltai = static_cast<double>(qi_l - ci);
        double deltai_sq = deltai * deltai;
        for (long cj = cj_min; cj <= cj_max; cj++) {
            double deltaj = static_cast<double>(qj_l - cj);
            double deltaj_sq = deltaj * deltaj;
            for (long ck = ck_min; ck <= ck_max; ck++) {
                double deltak = static_cast<double>(qk_l - ck);
                double dist_sq_qc = deltai_sq + deltaj_sq + deltak * deltak;
                if (dist_sq_qc == 0.0 || dist_sq_qc < best_dist_sq) {
                    size_t c_idx = static_cast<size_t>(ci) * nj * nk
                                 + static_cast<size_t>(cj) * nk
                                 + static_cast<size_t>(ck);
                    atomicAddDouble(&interp_sum[c_idx], best_value);
                    atomicAdd(&d_count[c_idx], 1ULL);
                }
            }
        }
    }
}

/*
 * normalize_kernel — one thread per output cell.
 * Divides the accumulated sum by the contribution count.
 */
__global__ void normalize_kernel(
        double* __restrict__ interp_values,
        const double* __restrict__ interp_sum,
        const unsigned long long* __restrict__ d_count,
        size_t total) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    if (d_count[i] > 0ULL) {
        interp_values[i] = interp_sum[i] / static_cast<double>(d_count[i]);
    }
}

inline int check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cuda_griddata: %s: %s\n", what, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}  // namespace

extern "C" int cuda_griddata(
        const double* known_x,
        const double* known_y,
        const double* known_z,
        const double* known_values,
        int num_known,
        double* interp_values,
        size_t ni,
        size_t nj,
        size_t nk) {
    if (num_known <= 0) {
        return 0;
    }

    const size_t grid_cells = ni * nj * nk;
    if (grid_cells == 0) {
        return 0;
    }

    const unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((grid_cells + threads - 1) / threads);

    size_t kn = static_cast<size_t>(num_known);

    double*             d_kx         = nullptr;
    double*             d_ky         = nullptr;
    double*             d_kz         = nullptr;
    double*             d_kv         = nullptr;
    double*             d_interp_sum = nullptr;
    unsigned long long* d_count      = nullptr;
    double*             d_interp_out = nullptr;

    if (check_cuda(cudaMalloc(&d_kx, kn * sizeof(double)), "cudaMalloc d_kx") != 0) {
        return 1;
    }
    if (check_cuda(cudaMalloc(&d_ky, kn * sizeof(double)), "cudaMalloc d_ky") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaMalloc(&d_kz, kn * sizeof(double)), "cudaMalloc d_kz") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaMalloc(&d_kv, kn * sizeof(double)), "cudaMalloc d_kv") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaMalloc(&d_interp_sum, grid_cells * sizeof(double)), "cudaMalloc d_interp_sum") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaMalloc(&d_count, grid_cells * sizeof(unsigned long long)), "cudaMalloc d_count") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaMalloc(&d_interp_out, grid_cells * sizeof(double)), "cudaMalloc d_interp_out") != 0) {
        goto cleanup;
    }

    if (check_cuda(cudaMemset(d_interp_sum, 0, grid_cells * sizeof(double)), "cudaMemset d_interp_sum") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaMemset(d_count, 0, grid_cells * sizeof(unsigned long long)), "cudaMemset d_count") != 0) {
        goto cleanup;
    }

    if (check_cuda(
                cudaMemcpy(d_kx, known_x, kn * sizeof(double), cudaMemcpyHostToDevice),
                "cudaMemcpy d_kx")
            != 0) {
        goto cleanup;
    }
    if (check_cuda(
                cudaMemcpy(d_ky, known_y, kn * sizeof(double), cudaMemcpyHostToDevice),
                "cudaMemcpy d_ky")
            != 0) {
        goto cleanup;
    }
    if (check_cuda(
                cudaMemcpy(d_kz, known_z, kn * sizeof(double), cudaMemcpyHostToDevice),
                "cudaMemcpy d_kz")
            != 0) {
        goto cleanup;
    }
    if (check_cuda(
                cudaMemcpy(d_kv, known_values, kn * sizeof(double), cudaMemcpyHostToDevice),
                "cudaMemcpy d_kv")
            != 0) {
        goto cleanup;
    }

    interpolation_kernel<<<blocks, threads>>>(
            d_kx,
            d_ky,
            d_kz,
            d_kv,
            num_known,
            ni,
            nj,
            nk,
            d_interp_sum,
            d_count);

    if (check_cuda(cudaGetLastError(), "interpolation_kernel launch") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaDeviceSynchronize(), "interpolation_kernel sync") != 0) {
        goto cleanup;
    }

    normalize_kernel<<<blocks, threads>>>(
            d_interp_out,
            d_interp_sum,
            d_count,
            grid_cells);

    if (check_cuda(cudaGetLastError(), "normalize_kernel launch") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaDeviceSynchronize(), "normalize_kernel sync") != 0) {
        goto cleanup;
    }

    if (check_cuda(
                cudaMemcpy(interp_values, d_interp_out, grid_cells * sizeof(double), cudaMemcpyDeviceToHost),
                "cudaMemcpy d_interp_out D2H")
            != 0) {
        goto cleanup;
    }

    cudaFree(d_kx);
    cudaFree(d_ky);
    cudaFree(d_kz);
    cudaFree(d_kv);
    cudaFree(d_interp_sum);
    cudaFree(d_count);
    cudaFree(d_interp_out);
    return 0;

cleanup:
    cudaFree(d_kx);
    cudaFree(d_ky);
    cudaFree(d_kz);
    cudaFree(d_kv);
    cudaFree(d_interp_sum);
    cudaFree(d_count);
    cudaFree(d_interp_out);
    return 1;
}
