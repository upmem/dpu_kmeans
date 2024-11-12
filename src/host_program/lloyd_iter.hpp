#pragma once

/**
 * @file lloyd_iter.hpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for lloyd_iter.cpp.
 *
 */

#include <pybind11/numpy.h>

extern "C" {
#include "../kmeans.h"
}

namespace py = pybind11;

/** @name lloyd_iter.cpp */
/**@{*/
void lloyd_iter(kmeans_params &p, const py::array_t<int_feature> &old_centers,
                py::array_t<int64_t> &centers_psum,
                py::array_t<int> &centers_pcount);
auto lloyd_iter_with_inertia(kmeans_params &p,
                             const py::array_t<int_feature> &old_centers,
                             std::vector<int64_t> &inertia_psum) -> int64_t;
/**@}*/
