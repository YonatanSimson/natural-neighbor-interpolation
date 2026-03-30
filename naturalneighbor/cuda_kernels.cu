#include "cuda_kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

/*
 * One GPU thread per output cell (ir, jr, kr).
 *
 * The discrete Sibson algorithm works as follows:
 *   For each query voxel Q=(i,j,k), find its nearest known point P.
 *   Every cell C in the ROI of Q (within radius dist(Q,P) of Q) receives
 *   a contribution of value(P).
 *
 * Equivalently, from the perspective of output cell C=(ir,jr,kr):
 *   C receives a contribution from query voxel Q=(i,j,k) if and only if:
 *     dist²(Q, C) < dist²(Q, nearest_known(Q))   OR   Q == C
 *   AND C is in the octant of Q determined by the octant partitioning.
 *
 * This thread iterates over all query voxels Q and checks whether C falls
 * in Q's ROI.  No atomics needed — each thread writes only its own cell.
 *
 * The brute-force NN search per query voxel (O(N) over known points) is
 * fast because N is small (hundreds to thousands) and all threads run in
 * parallel.  This matches the CPU result exactly.
 */

namespace {

__device__ inline long clamp_long(long val, long lo, long hi) {
    long t = (val > lo) ? val : lo;
    return (hi < t) ? hi : t;
}

__global__ void interpolation_kernel(
        const double* __restrict__ known_x,
        const double* __restrict__ known_y,
        const double* __restrict__ known_z,
        const double* __restrict__ known_values,
        int num_known,
        size_t ni,
        size_t nj,
        size_t nk,
        double* __restrict__ interp_values) {
    size_t total = ni * nj * nk;
    size_t out_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total) {
        return;
    }

    /* Output cell coordinates */
    size_t kr = out_idx % nk;
    size_t jr = (out_idx / nk) % nj;
    size_t ir = out_idx / (nk * nj);

    long i_middle = static_cast<long>(floor(static_cast<double>(ni) / 2.0));
    long j_middle = static_cast<long>(floor(static_cast<double>(nj) / 2.0));
    long k_middle = static_cast<long>(floor(static_cast<double>(nk) / 2.0));

    double accum = 0.0;
    unsigned long long count = 0ULL;

    /*
     * Iterate over all query voxels Q=(i,j,k).
     * For each Q, check if output cell C=(ir,jr,kr) is in Q's ROI.
     *
     * Q's ROI for octant `thread_number` is the intersection of:
     *   - the sphere of radius ceil(sqrt(dist²(Q, NN(Q)))) centred on Q
     *   - the octant half-space determined by thread_number
     *
     * C is in Q's ROI if:
     *   dist²(Q,C) == 0  OR  dist²(Q,C) < dist²(Q, NN(Q))
     *   AND C is in the octant of Q for some thread_number.
     *
     * The octant check: for thread_number t,
     *   bit0 of t: i-half  (0 → upper half [i_middle+1, ni-1],
     *                        1 → lower half [0, i_middle])
     *   bit1 of t: j-half
     *   bit2 of t: k-half
     *
     * C=(ir,jr,kr) is in the i-lower-half of Q=(i,j,k) iff
     *   ir is in [clamp(i-R,0,i_middle), clamp(i+R,0,i_middle)]
     * and in the i-upper-half iff
     *   ir is in [clamp(i-R,i_middle+1,ni-1), clamp(i+R,i_middle+1,ni-1)]
     *
     * Rather than iterating over all 8 octants per Q, we just check
     * whether C is in the full ROI sphere of Q (octant-independent check):
     * the union of all 8 octant ROIs covers the full sphere, so C is
     * contributed by Q iff dist²(Q,C) == 0 OR dist²(Q,C) < dist²(Q,NN(Q)).
     * The octant split only matters for thread-safety in the CPU version;
     * here each thread writes its own cell so we don't need it.
     */
    for (size_t i = 0; i < ni; i++) {
        for (size_t j = 0; j < nj; j++) {
            for (size_t k = 0; k < nk; k++) {
                /* Brute-force NN for query voxel Q=(i,j,k) */
                double qi = static_cast<double>(i);
                double qj = static_cast<double>(j);
                double qk = static_cast<double>(k);

                double best_dist_sq = 1.79769313486231570815e+308;
                double best_value = 0.0;
                double num_contributions = 1.0;

                for (int p = 0; p < num_known; p++) {
                    double di = qi - known_x[p];
                    double dj = qj - known_y[p];
                    double dk = qk - known_z[p];
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

                /* Distance² from Q to output cell C */
                double dci = static_cast<double>(i) - static_cast<double>(ir);
                double dcj = static_cast<double>(j) - static_cast<double>(jr);
                double dck = static_cast<double>(k) - static_cast<double>(kr);
                double dist_q_to_c_sq = dci * dci + dcj * dcj + dck * dck;

                if (dist_q_to_c_sq == 0.0 || dist_q_to_c_sq < best_dist_sq) {
                    /*
                     * C is in Q's ROI.  Now check the octant constraint:
                     * C must fall in the octant of Q that covers C.
                     * We replicate the CPU octant logic to find which
                     * octant C belongs to for query Q.
                     */
                    long i_l = static_cast<long>(i);
                    long j_l = static_cast<long>(j);
                    long k_l = static_cast<long>(k);
                    int roi_radius = static_cast<int>(ceil(sqrt(best_dist_sq)));
                    long roi_r = static_cast<long>(roi_radius);
                    long ni_l = static_cast<long>(ni);
                    long nj_l = static_cast<long>(nj);
                    long nk_l = static_cast<long>(nk);

                    bool in_any_octant = false;
                    for (int thread_number = 0; thread_number < 8 && !in_any_octant; thread_number++) {
                        size_t i_roi_min, i_roi_max;
                        if ((thread_number >> 0) % 2) {
                            i_roi_min = static_cast<size_t>(clamp_long(i_l - roi_r, 0, i_middle));
                            i_roi_max = static_cast<size_t>(clamp_long(i_l + roi_r, 0, i_middle));
                        } else {
                            i_roi_min = static_cast<size_t>(clamp_long(i_l - roi_r, i_middle + 1, ni_l - 1));
                            i_roi_max = static_cast<size_t>(clamp_long(i_l + roi_r, i_middle + 1, ni_l - 1));
                        }
                        size_t j_roi_min, j_roi_max;
                        if ((thread_number >> 1) % 2) {
                            j_roi_min = static_cast<size_t>(clamp_long(j_l - roi_r, 0, j_middle));
                            j_roi_max = static_cast<size_t>(clamp_long(j_l + roi_r, 0, j_middle));
                        } else {
                            j_roi_min = static_cast<size_t>(clamp_long(j_l - roi_r, j_middle + 1, nj_l - 1));
                            j_roi_max = static_cast<size_t>(clamp_long(j_l + roi_r, j_middle + 1, nj_l - 1));
                        }
                        size_t k_roi_min, k_roi_max;
                        if ((thread_number >> 2) % 2) {
                            k_roi_min = static_cast<size_t>(clamp_long(k_l - roi_r, 0, k_middle));
                            k_roi_max = static_cast<size_t>(clamp_long(k_l + roi_r, 0, k_middle));
                        } else {
                            k_roi_min = static_cast<size_t>(clamp_long(k_l - roi_r, k_middle + 1, nk_l - 1));
                            k_roi_max = static_cast<size_t>(clamp_long(k_l + roi_r, k_middle + 1, nk_l - 1));
                        }

                        if (i_roi_min > i_roi_max || j_roi_min > j_roi_max || k_roi_min > k_roi_max) {
                            continue;
                        }

                        if (ir >= i_roi_min && ir <= i_roi_max
                                && jr >= j_roi_min && jr <= j_roi_max
                                && kr >= k_roi_min && kr <= k_roi_max) {
                            in_any_octant = true;
                        }
                    }

                    if (in_any_octant) {
                        accum += best_value;
                        count += 1ULL;
                    }
                }
            }
        }
    }

    if (count > 0ULL) {
        interp_values[out_idx] = accum / static_cast<double>(count);
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

    double* d_kx = nullptr;
    double* d_ky = nullptr;
    double* d_kz = nullptr;
    double* d_kv = nullptr;
    double* d_interp = nullptr;

    size_t kn = static_cast<size_t>(num_known);

    if (check_cuda(cudaMalloc(&d_kx, kn * sizeof(double)), "cudaMalloc d_kx") != 0) {
        return 1;
    }
    if (check_cuda(cudaMalloc(&d_ky, kn * sizeof(double)), "cudaMalloc d_ky") != 0) {
        cudaFree(d_kx);
        return 1;
    }
    if (check_cuda(cudaMalloc(&d_kz, kn * sizeof(double)), "cudaMalloc d_kz") != 0) {
        cudaFree(d_kx);
        cudaFree(d_ky);
        return 1;
    }
    if (check_cuda(cudaMalloc(&d_kv, kn * sizeof(double)), "cudaMalloc d_kv") != 0) {
        cudaFree(d_kx);
        cudaFree(d_ky);
        cudaFree(d_kz);
        return 1;
    }
    if (check_cuda(cudaMalloc(&d_interp, grid_cells * sizeof(double)), "cudaMalloc d_interp") != 0) {
        cudaFree(d_kx);
        cudaFree(d_ky);
        cudaFree(d_kz);
        cudaFree(d_kv);
        return 1;
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
    if (check_cuda(
                cudaMemcpy(d_interp, interp_values, grid_cells * sizeof(double), cudaMemcpyHostToDevice),
                "cudaMemcpy d_interp H2D")
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
            d_interp);

    if (check_cuda(cudaGetLastError(), "interpolation_kernel launch") != 0) {
        goto cleanup;
    }
    if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize") != 0) {
        goto cleanup;
    }
    if (check_cuda(
                cudaMemcpy(interp_values, d_interp, grid_cells * sizeof(double), cudaMemcpyDeviceToHost),
                "cudaMemcpy d_interp D2H")
            != 0) {
        goto cleanup;
    }

    cudaFree(d_kx);
    cudaFree(d_ky);
    cudaFree(d_kz);
    cudaFree(d_kv);
    cudaFree(d_interp);
    return 0;

cleanup:
    cudaFree(d_kx);
    cudaFree(d_ky);
    cudaFree(d_kz);
    cudaFree(d_kv);
    cudaFree(d_interp);
    return 1;
}
