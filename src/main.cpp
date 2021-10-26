#include <pybind11/pybind11.h>
// #include <wordexp.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

int array_sum(){
    return 0;
}

extern "C" char* call_home(char*);
extern "C" int dpu_test(char*);
extern "C" int checksum(char*);
extern "C" int main(int, char**);

// namespace py = pybind11;

void kmeans(char *commandLine){
    int argc = 0;
    char *argv[64];
    char *p2 = strtok(commandLine, " ");
    while (p2 && argc < 63)
    {
        argv[argc++] = p2;
        p2 = strtok(0, " ");
    }
    argv[argc] = 0;

    main(argc, argv);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        DPU kmeans plugin
        -----------------

        .. currentmodule:: dpu_kmeans

        .. autosummary::
           :toctree: _generate

           add
           subtract
           call_home
           dpu_test
           checksum
           main
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("call_home", &call_home, R"pbdoc(
        Get a number from the c file
    )pbdoc");

    m.def("dpu_test", &dpu_test, R"pbdoc(
        Call hello world on dpu
    )pbdoc");

    m.def("checksum", &checksum, R"pbdoc(
        Checksum test on dpus
    )pbdoc");

    m.def("kmeans", &kmeans, R"pbdoc(
        Main kmeans function
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
