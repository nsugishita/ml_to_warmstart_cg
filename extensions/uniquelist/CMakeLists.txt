cmake_minimum_required(VERSION 3.4...3.18)
project(uniquelist)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(uniquelist src/main.cpp)
target_include_directories(uniquelist PUBLIC include)
target_compile_features(uniquelist PUBLIC cxx_std_20)

install(TARGETS uniquelist DESTINATION pythonmodules)
