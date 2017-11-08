cmake_minimum_required(VERSION 2.9)
project(TIRAMISU)

# setup
set(CMAKE_CXX_COMPILER /usr/local/bin/mpicxx)
set(HALIDE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Halide)
set(ISL_SOURCE_DIR /usr/local/)
set(ENV{DYLD_LIBRARY_PATH} ${CMAKE_SOURCE_DIR}/Halide/lib /usr/local/Cellar/jpeg/8d/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
add_definitions(-DBIN_DIR=\"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/\")

# compiler flags
set(CMAKE_CXX_FLAGS "-g -std=c++11 -O3 -Wall -Wno-sign-compare -fno-rtti -fvisibility=hidden")

# includes
include_directories(${ISL_SOURCE_DIR}/include /usr/lib/openmpi/include/ /home/jray/include ${CMAKE_SOURCE_DIR}/include ${HALIDE_SOURCE_DIR}/include ${HALIDE_SOURCE_DIR}/tools ${CMAKE_SOURCE_DIR}/build  /Users/je23693/anaconda2/include/python2.7)

# libraries
link_directories(${ISL_SOURCE_DIR}/lib /home/jray/lib ${HALIDE_SOURCE_DIR}/lib /usr/lib/openmpi/lib /opt/X11/lib/)
set(LIBS "-lisl -lgmp -lHalide -ldl -lpthread -lz -lpng16 -ljpeg -lncurses -lmpi")

# build the main tiramisu library
file(GLOB TIRAMISU_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB TIRAMISU_INCLUDE ${CMAKE_SOURCE_DIR}/include/tiramisu/*.h)
add_library(tiramisu STATIC ${TIRAMISU_SRC} ${TIRAMISU_INCLUDE})

add_executable(generated_dblurxy distributed_tests/dblurxy.cpp)
add_executable(dblurxy distributed_tests/wrapper_dblurxy.cpp ${CMAKE_SOURCE_DIR}/build/generated_dblurxy.o)
add_dependencies(dblurxy generated_dblurxy)
target_link_libraries(generated_dblurxy tiramisu ${LIBS})
target_link_libraries(dblurxy tiramisu ${LIBS})

add_executable(generated_dblurxy_dist_data distributed_tests/dblurxy_dist_data.cpp)
add_executable(dblurxy_dist_data distributed_tests/wrapper_dblurxy_dist_data.cpp ${CMAKE_SOURCE_DIR}/build/generated_dblurxy_dist_data.o)
add_dependencies(dblurxy_dist_data generated_dblurxy_dist_data)
target_link_libraries(generated_dblurxy_dist_data tiramisu ${LIBS})
target_link_libraries(dblurxy_dist_data tiramisu ${LIBS})

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_dblurxy.o
        COMMAND generated_dblurxy
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_dblurxy"
        SOURCES distributed_tests/dblurxy.cpp)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_dblurxy_dist_data.o
        COMMAND generated_dblurxy_dist_data
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_dblurxy_dist_data"
        SOURCES distributed_tests/dblurxy_dist_data.cpp)