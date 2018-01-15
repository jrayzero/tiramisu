cmake_minimum_required(VERSION 2.9)
project(TIRAMISU)

# setup
set(CMAKE_CXX_COMPILER g++)#/usr/local/bin/mpicxx)
set(HALIDE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Halide)
set(ISL_SOURCE_DIR 3rdParty/isl) #/usr/local/)
set(ENV{DYLD_LIBRARY_PATH} ${CMAKE_SOURCE_DIR}/Halide/lib /usr/local/Cellar/jpeg/8d/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
add_definitions(-DBIN_DIR=\"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\")

# compiler flags
set(CMAKE_CXX_FLAGS "-DNODES=1 -DPROCS=1 -g -std=c++11 -O0 -Wall -Wno-sign-compare -fno-rtti -fvisibility=hidden")

# includes
include_directories(${ISL_SOURCE_DIR}/include ${ISL_SOURCE_DIR} /usr/lib/openmpi/include/ /home/jray/include ${CMAKE_SOURCE_DIR}/include ${HALIDE_SOURCE_DIR}/include ${HALIDE_SOURCE_DIR}/tools ${CMAKE_SOURCE_DIR}/build  /Users/je23693/anaconda2/include/python2.7 /Developer/NVIDIA/CUDA-7.5/include/)

# libraries
link_directories(${ISL_SOURCE_DIR}/.libs /home/jray/lib ${HALIDE_SOURCE_DIR}/lib /usr/lib/openmpi/lib /opt/X11/lib/)
set(LIBS "-lisl -lgmp -lHalide -ldl -lpthread -lz -lpng16 -ljpeg -lncurses -lmpi")

# build the main tiramisu library
file(GLOB TIRAMISU_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp)
#file(GLOB TIRAMISU_SRC_CU ${CMAKE_SOURCE_DIR}/src/*.cu)
file(GLOB TIRAMISU_INCLUDE ${CMAKE_SOURCE_DIR}/include/tiramisu/*.h)
add_library(tiramisu STATIC ${TIRAMISU_SRC}  ${TIRAMISU_INCLUDE})
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/tiramisu_cuda.o
#        COMMAND bash ${CMAKE_SOURCE_DIR}/utils/compile_cuda_files.sh
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running nvcc on tiramisu_cuda.cu"
#        SOURCES ${CMAKE_SOURCE_DIR}/src/tiramisu_cuda.cu)

add_executable(generated_blur distributed/blur.cpp distributed/blur_params.h)
add_executable(blur distributed/wrapper_blur.cpp ${CMAKE_SOURCE_DIR}/build/generated_blur_dist.o)
add_dependencies(blur generated_blur)
target_link_libraries(generated_blur tiramisu ${LIBS})
target_link_libraries(blur tiramisu ${LIBS})

add_executable(generated_test_74 tests/test_74.cpp)
add_executable(test_74 tests/wrapper_test_74.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_test_74.o)
add_dependencies(test_74 generated_test_74)
target_link_libraries(generated_test_74 tiramisu ${LIBS})
target_link_libraries(test_74 tiramisu ${LIBS})

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_blur_dist.o
        COMMAND generated_blur
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_blur"
        SOURCES distributed/blur.cpp)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_test_74.o
        COMMAND generated_test_74
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_test1"
        SOURCES tests/test_74.cpp)