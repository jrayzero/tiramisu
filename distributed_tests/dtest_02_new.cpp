//
// Created by Jessica Ray on 10/6/17.
//

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

// 2D distributed blur. Adapted from tutorials/tutorial_02.cpp

#define SIZE0 1280
#define SIZE1 768
#define NUM_NODES 2

using namespace tiramisu;

typedef std::pair<std::string, std::string> ppair;
typedef std::vector<std::vector<computation *>> pred_group;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    /*
     * Declare a function blurxy.
     * Declare two arguments (tiramisu buffers) for the function: b_input0 and b_blury0
     * Declare an invariant for the function.
     */
    function blurxy("blurxy");

    constant p0("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &blurxy);
    constant p1("M", expr((int32_t) SIZE1), p_int32, true, NULL, 0, &blurxy);

    // Declare the computations c_blurx and c_blury.
    computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &blurxy);

    var i("i"), j("j");

    expr e1 = (c_input(i,j) + (uint8_t)10);

    computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<M}", e1, true, p_uint8, &blurxy);

    // Layer II

    S0.separate(0, expr((int32_t)SIZE0 / 4), 4, -1);

    S0.get_update(0).rename_computation("S0_0");
    S0.get_update(1).rename_computation("S0_1");

    // For ranks 1,2,3
    var p("p"), i_inner("i_inner");

    S0.get_update(0).distributed_split(i, 320, p, i_inner);

    S0.get_update(1).distributed_split(i, 320, p, i_inner);
//    S0.get_update(0).tag_distribute_level(p);
    S0.get_update(1).tag_distribute_level(p);

    // Layer III
    buffer b_input("b_input", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_input, &blurxy);
    buffer b_output("b_output", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_output, &blurxy);
    buffer b_temp("b_temp", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_dist, &blurxy);

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    S0.get_update(0).set_access("{S0_0[i,j]->b_output[i,j]}");
    S0.get_update(1).set_access("{S0_1[i,j]->b_temp[i,j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments to blurxy
    blurxy.set_arguments({&b_input, &b_output});
    // Generate code
    blurxy.gen_time_space_domain();
    blurxy.lift_ops_to_library_calls();
    blurxy.gen_isl_ast();
    blurxy.gen_halide_stmt();
    blurxy.gen_halide_obj("build/generated_fct_dtest_02.o");

    // Some debugging
    blurxy.dump_iteration_domain();
    blurxy.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    blurxy.dump(true);

    return 0;
}



////
//// Created by Jessica Ray on 10/6/17.
////
//
//#include <tiramisu/debug.h>
//#include <tiramisu/core.h>
//
//// 2D distributed blur. Adapted from tutorials/tutorial_02.cpp
//
//#define SIZE0 1280
//#define SIZE1 768
//#define NUM_NODES 2
//
//using namespace tiramisu;
//
//typedef std::pair<std::string, std::string> ppair;
//typedef std::vector<std::vector<computation *>> pred_group;
//
//int main(int argc, char **argv)
//{
//    // Set default tiramisu options.
//    global::set_default_tiramisu_options();
//
//    // -------------------------------------------------------
//    // Layer I
//    // -------------------------------------------------------
//
//    /*
//     * Declare a function blurxy.
//     * Declare two arguments (tiramisu buffers) for the function: b_input0 and b_blury0
//     * Declare an invariant for the function.
//     */
//    function blurxy("blurxy");
//
//    constant p0("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &blurxy);
//    constant p1("M", expr((int32_t) SIZE1), p_int32, true, NULL, 0, &blurxy);
//
//    // Declare the computations c_blurx and c_blury.
//    computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &blurxy);
//
//    var i("i"), j("j");
//
//    expr e1 = (c_input(i - 1, j) +
//               c_input(i    , j) +
//               c_input(i + 1, j)) / (uint8_t)3;
//
//    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N and 0<j<M}", e1, true, p_uint8, &blurxy);
//
//    expr e2 = (c_blurx(i, j - 1) +
//               c_blurx(i, j) +
//               c_blurx(i, j + 1)) / (uint8_t)3;
//
//    computation c_blury("[N,M]->{c_blury[i,j]: 1<i<N-1 and 1<j<M-1}", e2, true, p_uint8, &blurxy);
//
//    // -------------------------------------------------------
//    // Layer II
//    // -------------------------------------------------------
//
//    // TODO do this with separate if want some specific splits
//    // TODO can split take an expression instead, or a constant.
//
//    c_blurx.split(var("i"), SIZE0/2, var("p"), var("i_inner")); // The outer loop will have two iterations, so two nodes
//    // Need to make sure that the data computed on each node stays there for both the blurx and the blurx
//    c_blurx.tag_distribute_level(var("p"));
//
//    // Make the communication primitives
//
//
//    // -------------------------------------------------------
//    // Layer III
//    // -------------------------------------------------------
//
//    // TODO how to indicate that the buffer on the home node is the one that should be returned?
//
//
//    buffer b_input("b_input", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_input, &blurxy);
//    // Create an "exemplar" buffer. Under the hood, this actually represents many buffers, one for each distributed node
//    dist_buffer b_blurx("b_blurx", {tiramisu::expr(SIZE0)/2, tiramisu::expr(SIZE1)}, p_uint8, &blurxy);
//    buffer b_blury("b_blury", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_output, &blurxy);
//
//    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
//    // we are saying that we need to assign this access per buffer p in this distributed version
//    c_blurx.set_access("[N]->{c_blurx[p,i,j]->b_blurx[i-p*N/2,j]}"); // recognize that b_blurx here is a distributed buffer
//    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");
//
//
//    // -------------------------------------------------------
//    // Code Generation
//    // -------------------------------------------------------
//
//    // Set the arguments to blurxy
//    blurxy.set_arguments({&b_input, &b_blury});
//    // Generate code
//    blurxy.gen_time_space_domain();
//    blurxy.lift_ops_to_library_calls();
//    blurxy.gen_isl_ast();
//    blurxy.gen_halide_stmt();
//    blurxy.gen_halide_obj("build/generated_fct_dtest_02.o");
//
//    // Some debugging
//    blurxy.dump_iteration_domain();
//    blurxy.dump_halide_stmt();
//
//    // Dump all the fields of the blurxy class.
//    blurxy.dump(true);
//
//    return 0;
//}
