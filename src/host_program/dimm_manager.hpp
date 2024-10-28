/**
 * @file dimm_manager.hpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for dimm_manager.cpp.
 *
 */

#include <pybind11/numpy.h>

#include <filesystem>

extern "C" {
#include "../kmeans.h"
}

namespace py = pybind11;

/** @name dimmm_manager.cpp */
/**@{*/
void allocate_dpus(kmeans_params &p);
void free_dpus(const kmeans_params &p);
void load_kernel_internal(const kmeans_params &p,
                          const std::filesystem::path &binary_path);
void broadcast_number_of_clusters(const kmeans_params &p, int nclusters);
void populate_dpus(kmeans_params &p,
                   const py::array_t<int_feature> &py_features);
void broadcast_parameters(const kmeans_params &p);
/**@}*/
