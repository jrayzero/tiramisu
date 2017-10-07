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

    expr e1 = (c_input(i - 1, j) +
               c_input(i    , j) +
               c_input(i + 1, j)) / (uint8_t)3;

    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N and 0<j<M}", e1, true, p_uint8, &blurxy);

    expr e2 = (c_blurx(i, j - 1) +
               c_blurx(i, j) +
               c_blurx(i, j + 1)) / (uint8_t)3;

    computation c_blury("[N,M]->{c_blury[i,j]: 1<i<N-1 and 1<j<M-1}", e2, true, p_uint8, &blurxy);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Do just 2 nodes for now
    std::vector<ppair> blurx_domain_splits = {ppair("c_blurx0", "[N,M]->{c_blurx[i,j]: 0<i<N/2 and 0<j<M}"),
                                              ppair("c_blurx1", "[N,M]->{c_blurx[i,j]: N/2<=i<N-1 and 0<j<M}")};
    std::vector<ppair> blury_domain_splits = {ppair("c_blury0", "[N,M]->{c_blury[i,j]: 1<i<N/2 and 1<j<M-1}"),
                                              ppair("c_blury1", "[N,M]->{c_blury[i,j]: N/2<=i<N-1 and 1<j<M-1}")};

    std::vector<computation *> blurx_parts = c_blurx.partition<computation>(blurx_domain_splits);
    std::vector<computation *> blury_parts = c_blury.partition<computation>(blury_domain_splits);

    channel chan_sync_block("chan_sync_block", p_uint8, {FIFO, SYNC, BLOCK, MPI});

    send_recv n0n1 = computation::create_transfer("[N,M]->{[i,j]: N/2-1<=i<N and 0<=j<M}", "send_0_1", "recv_0_1",
                                                  &chan_sync_block, &chan_sync_block, c_input(var("i"), var("j")),
                                                  {blurx_parts[1]}, &blurxy);
    send_recv n1n0 = computation::create_transfer("[N,M]->{[i,j]: N/2<=i<N and 0<=j<M}", "send_1_0", "recv_1_0",
                                                  &chan_sync_block, &chan_sync_block,
                                                  (*blury_parts[1])(var("i"), var("j")), {}, &blurxy);

    // node 0 scheduling
    n0n1.s->set_low_level_schedule("{send_0_1[i,j]->send_0_1[0,0,i,0,j,0]}");
    blurx_parts[0]->set_low_level_schedule("{c_blurx0[i,j]->c_blurx0[0,1,i,0,j,0]}");
    blury_parts[0]->set_low_level_schedule("{c_blury0[i,j]->c_blury0[0,2,i,0,j,0]}");
    n1n0.r->set_low_level_schedule("{recv_1_0[i,j]->recv_1_0[0,3,i,0,j,0]}");

    // node 1 scheduling
    n0n1.r->set_low_level_schedule("{recv_0_1[i,j]->recv_0_1[0,0,i,0,j,0]}");
    blurx_parts[1]->set_low_level_schedule("{c_blurx1[i,j]->c_blurx1[0,1,i,0,j,0]}");
    blury_parts[1]->set_low_level_schedule("{c_blury1[i,j]->c_blury1[0,2,i,0,j,0]}");
    n1n0.s->set_low_level_schedule("{send_1_0[i,j]->send_1_0[0,3,i,0,j,0]}");

    pred_group groups = {{&c_input, n0n1.s, blurx_parts[0], blury_parts[0], n1n0.r}, {n0n1.r, blurx_parts[1],
                                            blury_parts[1], n1n0.s}};
    computation::distribute(groups, {0, 1});

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_input0("b_input0", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_input, &blurxy);
    buffer b_input1("b_input1", {tiramisu::expr(SIZE0)/2 + 1, tiramisu::expr(SIZE1)}, p_uint8, a_temporary, &blurxy);

    buffer b_blury0("b_blury0", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_output, &blurxy);
    buffer b_blury("b_blury1", {tiramisu::expr(SIZE0)/2, tiramisu::expr(SIZE1)}, p_uint8, a_temporary, &blurxy);

    buffer b_blurx0("b_blurx0", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_temporary, &blurxy);
    buffer b_blurx1("b_blurx1", {tiramisu::expr(SIZE0)/2, tiramisu::expr(SIZE1)}, p_uint8, a_temporary, &blurxy);

    c_input.set_access("{c_input[i,j]->b_input0[i,j]}");
    blurx_parts[0]->set_access("{c_blurx0[i,j]->b_blurx0[i,j]}");
    blurx_parts[1]->set_access("{c_blurx1[i,j]->b_blurx1[i,j]}");
    blury_parts[0]->set_access("{c_blury0[i,j]->b_blury0[i,j]}");
    blury_parts[1]->set_access("{c_blury1[i,j]->b_blury1[i,j]}");
    n0n1.r->set_access("{recv_0_1[i,j]->b_input1[i,j]}");
    n1n0.r->set_access("{recv_1_0[i,j]->b_blury0[i,j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments to blurxy
    blurxy.set_arguments({&b_input0, &b_blury0});
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
