/**
 * @file dimm_manager.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny functions for the DPUs in C++.
 *
 */

#include "dimm_manager.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include <pybind11/numpy.h>

extern "C" {
#include <dpu.h>
}

/**
 * @brief Fills the DPUs with their assigned points.
 */
void populate_dpus(kmeans_params *p,             /**< Algorithm parameters */
                 const py::array_t<int_feature> &py_features) /**< array: [npoints][nfeatures] */
{
  auto features = py_features.unchecked<2>();

  /* Iteration variables for the DPUs. */
  dpu_set_t dpu{};
  uint32_t each_dpu = 0;

  std::vector<int> nreal_points(p->ndpu); /* number of real data points on each dpu */
  int64_t padding_points =
      p->npadded - p->npoints; /* number of padding points */

  const auto tic = std::chrono::steady_clock::now();

  int64_t next = 0;
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    int64_t current = next;
    DPU_ASSERT(dpu_prepare_xfer(dpu, const_cast<int_feature *>(features.data(next, 0))));
    padding_points -= p->npointperdpu;
    next = std::max(0L, -padding_points);
    nreal_points[each_dpu] = next - current;
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "t_features", 0,
                           p->npointperdpu * p->nfeatures * sizeof(int_feature),
                           DPU_XFER_DEFAULT));

  /* DEBUG : print the number of non-padding points assigned to each DPU */
  // printf("nreal_points :\n");
  // for(int idpu = 0; idpu < ndpu; idpu++)
  // {
  //     printf("%d ", nreal_points[idpu]);
  // }
  // printf("\n");

  DPU_FOREACH(p->allset, dpu, each_dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &nreal_points[each_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(p->allset, DPU_XFER_TO_DPU, "npoints", 0,
                           sizeof(int), DPU_XFER_DEFAULT));

  const auto toc = std::chrono::steady_clock::now();
  p->cpu_pim_time = std::chrono::duration<double>{toc - tic}.count();
}
