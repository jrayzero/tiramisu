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
#
#add_executable(generated_dist_01 distributed_tutorials/tutorial_01_dist.cpp)
#add_executable(dist_01 distributed_tutorials/wrapper_tutorial_01_dist.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_01_dist.o)
#add_dependencies(dist_01 generated_dist_01)
#
#add_executable(theano /Users/je23693/PycharmProjects/Theano/theano/jess/theano_2_tiramisu_add.cpp)
#
#add_executable(generated_dist_02 distributed_tutorials/tutorial_02_dist.cpp)
#add_executable(dist_02 distributed_tutorials/wrapper_tutorial_02_dist.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_02_dist.o)
#add_dependencies(dist_02 generated_dist_02)
#
#add_executable(generated_dist_03 distributed_tutorials/tutorial_03_dist.cpp)
#add_executable(dist_03 distributed_tutorials/wrapper_tutorial_03_dist.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_03_dist.o)
#add_dependencies(dist_03 generated_dist_03)
#
#add_executable(generated_serial_fwdpass distributed_tutorials/serial_fwdpass.cpp)
#add_executable(serial_fwdpass distributed_tutorials/wrapper_serial_fwdpass.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_serial_fwdpass.o)
#add_dependencies(serial_fwdpass generated_serial_fwdpass)
#
#add_executable(generated_distributed_fwdpass distributed_tutorials/distributed_fwdpass.cpp)
#add_executable(distributed_fwdpass distributed_tutorials/wrapper_distributed_fwdpass.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_distributed_fwdpass.o)
#add_dependencies(distributed_fwdpass generated_distributed_fwdpass)
#
#add_executable(generated_test_01 tests/test_01.cpp)
#add_executable(test_01 tests/wrapper_test_01.cpp ${CMAKE_SOURCE_DIR}/build/generated_fct_test_01.o)
#add_dependencies(test_01 generated_test_01)
#
#target_link_libraries(generated_dist_01 tiramisu ${LIBS})
#target_link_libraries(dist_01 tiramisu ${LIBS})
#
#target_link_libraries(theano tiramisu ${LIBS})
#
#target_link_libraries(generated_dist_02 tiramisu ${LIBS})
#target_link_libraries(dist_02 tiramisu ${LIBS})
#
#target_link_libraries(generated_dist_03 tiramisu ${LIBS})
#target_link_libraries(dist_03 tiramisu ${LIBS})
#
#target_link_libraries(generated_serial_fwdpass tiramisu ${LIBS})
#target_link_libraries(serial_fwdpass tiramisu ${LIBS})
#
#target_link_libraries(generated_distributed_fwdpass tiramisu ${LIBS})
#target_link_libraries(distributed_fwdpass tiramisu ${LIBS})
#
#target_link_libraries(generated_test_01 tiramisu ${LIBS})
#target_link_libraries(test_01 tiramisu ${LIBS})
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_01_dist.o
#        COMMAND generated_dist_01
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running generated_dist_01"
#        SOURCES distributed_tutorials/tutorial_01_dist.cpp)
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_02_dist.o
#        COMMAND generated_dist_02
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running generated_dist_02"
#        SOURCES distributed_tutorials/tutorial_02_dist.cpp)
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_tutorial_03_dist.o
#        COMMAND generated_dist_03
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running generated_dist_03"
#        SOURCES distributed_tutorials/tutorial_03_dist.cpp)
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_serial_fwdpass.o
#        COMMAND generated_serial_fwdpass
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running generated_serial_fwdpass"
#        SOURCES distributed_tutorials/serial_fwdpass.cpp)
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_distributed_fwdpass.o
#        COMMAND generated_distributed_fwdpass
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running generated_distributed_fwdpass"
#        SOURCES distributed_tutorials/distributed_fwdpass.cpp)
#
#add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/generated_fct_test_01.o
#        COMMAND generated_test_01
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        COMMENT "Running generated_test_01"
#        SOURCES tests/test_01.cpp)