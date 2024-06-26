cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80;90")

project(crs4_cassandra_plugin LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories("${DALI_LIB_DIR}")
link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
link_directories("$ENV{CONDA_DALI_LIB}")

add_library(crs4cassandra SHARED cassandra_dali_interactive.cc cassandra_dali_selffeed.cc cassandra_dali_decoupled.cc batch_loader.cc numpy_decoder.cc)
target_link_libraries(crs4cassandra dali cudart cassandra)
