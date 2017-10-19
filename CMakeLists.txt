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

add_executable(generated_dtest_01 distributed_tests/dtest_01.cpp)
add_executable(dtest_01 distributed_tests/wrapper_dtest_01.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_01_dtest.o)
add_dependencies(dtest_01 generated_dtest_01)

add_executable(generated_dtest_02 distributed_tests/dtest_02.cpp)
add_executable(dtest_02 distributed_tests/wrapper_dtest_02.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_02_dtest.o)
add_dependencies(dtest_02 generated_dtest_02)

add_executable(generated_dtest_03 distributed_tests/dtest_03.cpp)
add_executable(dtest_03 distributed_tests/wrapper_dtest_03.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_03_dtest.o)
add_dependencies(dtest_03 generated_dtest_03)

add_executable(generated_test_14 tests/test_14.cpp)
add_executable(test_14 tests/wrapper_test_14.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_test_14.o)
add_dependencies(test_14 generated_test_14)

target_link_libraries(generated_dtest_01 tiramisu ${LIBS})
target_link_libraries(dtest_01 tiramisu ${LIBS})

target_link_libraries(generated_dtest_02 tiramisu ${LIBS})
target_link_libraries(dtest_02 tiramisu ${LIBS})

target_link_libraries(generated_dtest_03 tiramisu ${LIBS})
target_link_libraries(dtest_03 tiramisu ${LIBS})

target_link_libraries(generated_test_14 tiramisu ${LIBS})
target_link_libraries(test_14 tiramisu ${LIBS})

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_01_dtest.o
        COMMAND generated_dtest_01
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_dtest_01"
        SOURCES distributed_tests/dtest_01.cpp)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_02_dtest.o
        COMMAND generated_dtest_02
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_dtest_02"
        SOURCES distributed_tests/dtest_02.cpp)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_03_dtest.o
        COMMAND generated_dtest_03
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_dtest_03"
        SOURCES distributed_tests/dtest_03.cpp)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_test_14.o
        COMMAND generated_test_14
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running generated_test_14"
        SOURCES tests/test_14.cpp)