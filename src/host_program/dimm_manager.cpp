/**
 * @file dimm_manager.c
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Nanny functions for the DPUs in C++.
 *
 */

#include "dimm_manager.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

extern "C" {
#include <dpu.h>
}

/**
 * @brief Fills the DPUs with their assigned points.
 */
void populate_dpus(kmeans_params *p, /**< Algorithm parameters */
                   const py::array_t<int_feature>
                       &py_features) /**< array: [npoints][nfeatures] */
{
  auto features = py_features.unchecked<2>();

  /* Iteration variables for the DPUs. */
  dpu_set_t dpu{};
  uint32_t each_dpu = 0;

  std::vector<int> nreal_points(
      p->ndpu); /* number of real data points on each dpu */
  int64_t padding_points =
      p->npadded - p->npoints; /* number of padding points */

  const auto tic = std::chrono::steady_clock::now();

  int64_t next = 0;
  DPU_FOREACH(p->allset, dpu, each_dpu) {
    int64_t current = next;
    DPU_ASSERT(dpu_prepare_xfer(
        dpu, const_cast<int_feature *>(features.data(next, 0))));
    padding_points -= p->npointperdpu;
    next = std::max(0L, -padding_points);
    if (next - current > std::numeric_limits<int>::max()) {
      throw std::runtime_error("too many points for one DPU");
    }
    nreal_points[each_dpu] = static_cast<int>(next - current);
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

/**
 * @brief Computes the appropriate task size for DPU tasklets.
 *
 * @param p Algorithm parameters.
 * @return The task size in bytes.
 */
static constexpr auto get_task_size(kmeans_params *p) -> int {
  /* how many points we can fit in w_features */
  int max_task_size =
      (WRAM_FEATURES_SIZE / static_cast<int>(sizeof(int_feature))) /
      p->nfeatures;

  /* number of tasks as the smallest multiple of NR_TASKLETS higher than
   * npointperdu / max_task_size */
  int ntasks =
      static_cast<int>((p->npointperdpu + max_task_size - 1) / max_task_size);
  ntasks = ((ntasks + NR_TASKLETS - 1) / NR_TASKLETS) * NR_TASKLETS;

  int task_size_in_points = std::min(
      static_cast<int>((p->npointperdpu + ntasks - 1) / ntasks), max_task_size);
  /* task size has to be at least 1 */
  task_size_in_points = std::max(task_size_in_points, 1);

  int task_size_in_features = task_size_in_points * p->nfeatures;
  int task_size_in_bytes =
      task_size_in_features * static_cast<int>(sizeof(int_feature));

  /* task size in bytes must be a multiple of 8 for DMA alignment and also a
   * multiple of number of features x byte size of integers */
  int lcm = std::lcm(static_cast<int>(sizeof(int_feature)) * p->nfeatures, 8);
  task_size_in_bytes = task_size_in_bytes / lcm * lcm;
  if (task_size_in_bytes > WRAM_FEATURES_SIZE) {
    throw std::runtime_error("tasks will not fit in WRAM");
  }
  /* minimal size */
  task_size_in_bytes = std::max(task_size_in_bytes, lcm);

  return task_size_in_bytes;
}

/**
 * @brief Broadcasts iteration parameters to the DPUs.
 *
 * @param p Algorithm parameters.
 */
void broadcastParameters(kmeans_params *p) {
  /* parameters to calculate once here and send to the DPUs. */

  /* compute the iteration variables for the DPUs */
  int task_size_in_bytes = get_task_size(p);

  /* realign task size in features and points */
  int task_size_in_features =
      task_size_in_bytes / static_cast<int>(sizeof(int_feature));
  int task_size_in_points = task_size_in_features / p->nfeatures;

  /* send computation parameters to the DPUs */
  DPU_ASSERT(dpu_broadcast_to(p->allset, "nfeatures_host", 0, &p->nfeatures,
                              sizeof(p->nfeatures), DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_points_host", 0,
                              &task_size_in_points, sizeof(task_size_in_points),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_bytes_host", 0,
                              &task_size_in_bytes, sizeof(task_size_in_bytes),
                              DPU_XFER_DEFAULT));
  DPU_ASSERT(dpu_broadcast_to(p->allset, "task_size_in_features_host", 0,
                              &task_size_in_features,
                              sizeof(task_size_in_features), DPU_XFER_DEFAULT));

  if (p->isOutput) {
    std::cout << "points per DPU : " << p->npointperdpu << '\n';
    std::cout << "task size in points : " << task_size_in_points << '\n';
    std::cout << "task size in bytes : " << task_size_in_bytes << '\n';
    std::cout << "tasks per DPU: " << p->npointperdpu / task_size_in_points
              << '\n';
  }
}
