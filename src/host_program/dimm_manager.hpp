/**
 * @file dimm_manager.hpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Header file for dimm_manager.cpp.
 *
 */

#include <pybind11/numpy.h>

extern "C" {
#include "../kmeans.h"
}

namespace py = pybind11;

/** @name dimmm_manager.cpp */
/**@{*/
void populate_dpus(kmeans_params *p,
                   const py::array_t<int_feature> &py_features);
void broadcastParameters(kmeans_params *p);
/**@}*/
