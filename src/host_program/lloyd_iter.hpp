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
void lloydIter(kmeans_params &p, const py::array_t<int_feature> &old_centers,
               py::array_t<int64_t> &new_centers,
               py::array_t<int> &new_centers_len,
               std::vector<int> &centers_pcount,
               std::vector<int64_t> &centers_psum);
uint64_t lloydIterWithInertia(kmeans_params *p, int_feature *old_centers,
                              int64_t *inertia_psum);
/**@}*/
