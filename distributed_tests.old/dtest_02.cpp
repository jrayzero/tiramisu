//
// Created by Jessica Ray on 10/6/17.
//

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

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
     * Declare a function dtest_02.
     * Declare two arguments (tiramisu buffers) for the function: b_input0 and b_blury0
     * Declare an invariant for the function.
     */
    function dtest_02("dtest_02");

    constant p0("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &dtest_02);
    constant p1("M", expr((int32_t) SIZE1), p_int32, true, NULL, 0, &dtest_02);

    // Declare the computations c_blurx and c_blury.
    computation c_input("[N,M]->{c_input[i,j]: 0<=i<N and 0<=j<M}", expr(), false, p_uint32, &dtest_02);

    var i("i"), j("j");

    expr e1 = (c_input(i,j) + (uint32_t)10);

    computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<M}", e1, true, p_uint32, &dtest_02);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S0.separate_at(0, SIZE0, SIZE0/4, -3);

    S0.get_update(0).rename_computation("S0_0");
    S0.get_update(1).rename_computation("S0_1");

    var p("p"), i_inner("i_inner");
    S0.get_update(0).split(i, 320, p, i_inner);
    S0.get_update(1).split(i, 320, p, i_inner);
    S0.get_update(0).tag_distribute_level(p);
    S0.get_update(1).tag_distribute_level(p);

    // communication
    channel chan_sync_block("chan_sync_block", p_uint32, {FIFO, SYNC, BLOCK, MPI});
    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &dtest_02);
    send_recv fan_out = computation::create_transfer(
            "[one]->{send_0_1[q,p,i,j]: 0<=q<one and 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}",
            "{recv_0_1[p,i,j]: 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}", 0, var("p"), &chan_sync_block,
            &chan_sync_block, c_input(var("i"), var("j")),
            {&(S0.get_update(1))}, &dtest_02);

    send_recv fan_in =
            computation::create_transfer("{send_1_0[p,i,j]: 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}",
                                         "[one]->{recv_1_0[q,p,i,j]: 0<=q<one and 1<=p<4 and q*320<=i<(q+1)*320 and 0<=j<768}",
                                         var("p"), 0, &chan_sync_block, &chan_sync_block,
                                         S0.get_update(1)(var("i") - var("p") * 320, var("j")), {}, &dtest_02);

    fan_out.s->tag_distribute_level(var("q"));
    fan_out.r->tag_distribute_level(var("p"));
    fan_in.s->tag_distribute_level(var("p"));
    fan_in.r->tag_distribute_level(var("q"));
    S0.get_update(0).drop_rank_iter();
    S0.get_update(1).drop_rank_iter();

    fan_out.s->before(S0.get_update(0), computation::root);
    S0.get_update(0).before(*fan_out.r, computation::root);
    fan_out.r->before(S0.get_update(1), computation::root);
    fan_in.s->after(S0.get_update(1), computation::root);
    fan_in.r->after(S0.get_update(0), computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Distribute this input buffer
    buffer b_input("b_input", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint32, a_input, &dtest_02);
    b_input.distribute({tiramisu::expr(SIZE0/4), tiramisu::expr(SIZE1)}, "b_input_temp");
    buffer b_output("b_output", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint32, a_output, &dtest_02);
    buffer b_temp("b_temp", {tiramisu::expr(SIZE0/4), tiramisu::expr(SIZE1)}, p_uint32, a_temporary, &dtest_02);

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    S0.get_update(0).set_access("{S0_0[i,j]->b_output[i,j]}");
    S0.get_update(1).set_access("{S0_1[i,j]->b_temp[i,j]}");
    fan_out.r->set_access("{recv_0_1[p,i,j]->b_input_temp[i-p*320,j]}");
    fan_in.r->set_access("{recv_1_0[q,k,i,j]->b_output[k*320+i,j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments to dtest_02
    dtest_02.set_arguments({&b_input, &b_output});
    // Generate code
    dtest_02.gen_time_space_domain();
    dtest_02.lift_ops_to_library_calls();
    dtest_02.gen_isl_ast();
    dtest_02.gen_halide_stmt();
    dtest_02.gen_halide_obj("build/generated_fct_dtest_02.o");

    // Some debugging
    dtest_02.dump_iteration_domain();
    dtest_02.dump_halide_stmt();

    // Dump all the fields of the dtest_02 class.
    dtest_02.dump(true);

    return 0;
}
