/**
 * @file main.cpp
 * @author Sylvan Brocard (sbrocard@upmem.com)
 * @brief Binding file for the KMeans project
 * @copyright 2024 UPMEM
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "kmeans.hpp"

extern "C" {
#include <dpu.h>

#include "common.h"
}

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

[[nodiscard]] constexpr auto add(int i, int j) -> int { return i + j; }

extern "C" auto checksum(char *) -> int;

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        DPU kmeans plugin
        -----------------

        .. currentmodule:: dpu_kmeans._core

        .. autosummary::
           :toctree: _generate

           Container
    )pbdoc";

  py::class_<Container>(m, "Container", R"pbdoc(
        Container object to interface with the DPUs
    )pbdoc")
      .def(py::init<>())
      .def("allocate", &Container::allocate)
      .def_property_readonly("nr_dpus", &Container::get_ndpu)
      .def_property_readonly("allocated", &Container::allocated)
      .def_property_readonly("hash", &Container::hash)
      .def_property_readonly("binary_path", &Container::binary_path)
      .def_property_readonly("data_size", &Container::data_size)
      .def_property_readonly("inertia", &Container::get_inertia)
      .def("load_kernel", &Container::load_kernel)
      .def("load_array_data", &Container::load_array_data)
      .def("load_n_clusters", &Container::load_nclusters)
      .def("free_dpus", &Container::free_dpus)
      .def("lloyd_iter", &Container::lloyd_iter)
      .def("reset_timer", &Container::reset_timer)
      .def_property_readonly("dpu_run_time", &Container::get_dpu_run_time)
      .def_property_readonly("cpu_pim_time", &Container::get_cpu_pim_time)
      .def_property_readonly("pim_cpu_time", &Container::get_pim_cpu_time);

  m.attr("FEATURE_TYPE") = py::int_(FEATURE_TYPE);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
