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
    computation c_input("[N,M]->{c_input[i,j]: 0<=i<N and 0<=j<M}", expr(), false, p_uint32, &blurxy);

    var i("i"), j("j");

    expr e1 = (c_input(i,j) + (uint32_t)10);

    computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<M}", e1, true, p_uint32, &blurxy);

    // Layer II

    S0.separate(0, expr((int32_t)SIZE0 / 4), 4, -3);

    S0.get_update(0).rename_computation("S0_0");
    S0.get_update(1).rename_computation("S0_1");

    var p("p"), i_inner("i_inner");
    S0.get_update(0).distributed_split(i, 320, p, i_inner);
    S0.get_update(1).distributed_split(i, 320, p, i_inner);
    S0.get_update(0).tag_distribute_level(p);
    S0.get_update(1).tag_distribute_level(p);


    // communication
    channel chan_sync_block("chan_sync_block", p_uint32, {FIFO, SYNC, BLOCK, MPI});
    // The requirements for the var naming here is a little wonky.
    send_recv fan_out = computation::create_transfer(
            "{[q,p,i,j]: 0<=q<1 and 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}",
            "{[p,i,j]: 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}", "send_0_1", "recv_0_1", 0, var("p"),
            &chan_sync_block, &chan_sync_block, c_input(var("i"), var("j")),
            {&(S0.get_update(1))}, &blurxy);
    send_recv fan_in = computation::create_transfer(
            "{[p,i,j]: 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}",
            "{[q,p,i,j]: 0<=q<1 and 1<=p<4 and q*320<=i<(q+1)*320 and 0<=j<768}", "send_1_0", "recv_1_0",
            var("p"), 0, &chan_sync_block, &chan_sync_block, S0.get_update(1)(var("i")-var("p")*320, var("j")),
            {}, &blurxy);
    fan_out.s->tag_distribute_level(var("q"));
    fan_out.r->tag_distribute_level(var("p"));
    fan_in.s->tag_distribute_level(var("p"));

    fan_in.r->tag_distribute_level(var("q"));

    fan_out.s->before(S0.get_update(0), computation::root);
    S0.get_update(0).before(*fan_out.r, computation::root);
    fan_out.r->before(S0.get_update(1), computation::root);
    fan_in.s->after(S0.get_update(1), computation::root);
    fan_in.r->after(S0.get_update(0), computation::root);

    // Layer III
    // Distribute this input buffer
    buffer b_input("b_input", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint32, a_input, &blurxy);
    b_input.distribute({tiramisu::expr(SIZE0 / 4), tiramisu::expr(SIZE1)}, "b_input_temp");
    buffer b_output("b_output", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint32, a_output, &blurxy);
    buffer b_temp("b_temp", {tiramisu::expr(SIZE0/4), tiramisu::expr(SIZE1)}, p_uint32, a_temporary, &blurxy);

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    S0.get_update(0).set_access("{S0_0[i,j]->b_output[i,j]}");
    S0.get_update(1).set_access("{S0_1[i,j]->b_temp[i,j]}");
    fan_out.r->set_access("{recv_0_1[p,i,j]->b_input_temp[i-p*320,j]}");
    fan_in.r->set_access("{recv_1_0[q,k,i,j]->b_output[k*320+i,j]}");

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
