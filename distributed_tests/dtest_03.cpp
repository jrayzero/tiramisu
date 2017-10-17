//
// Created by Jessica Ray on 10/6/17.
//

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

// 2D distributed blur. Adapted from tutorials/tutorial_02.cpp

#define SIZE0 1280
#define SIZE1 768

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
     * Declare a function dtest_03.
     * Declare two arguments (tiramisu buffers) for the function: b_input0 and b_blury0
     * Declare an invariant for the function.
     */
    function dtest_03("dtest_03");

    constant p0("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &dtest_03);
    constant p1("M", expr((int32_t) SIZE1), p_int32, true, NULL, 0, &dtest_03);

    // Declare the computations c_blurx and c_blury.
    computation c_input("[N,M]->{c_input[i,j]: 0<=i<N and 0<=j<M}", expr(), false, p_uint8, &dtest_03);

    var i("i"), j("j");

    expr e1 = (c_input(i - 1, j) +
               c_input(i    , j) +
               c_input(i + 1, j)) / (uint8_t)3;

    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N and 0<j<M}", e1, true, p_uint8, &dtest_03);

    expr e2 = (c_blurx(i, j - 1) +
               c_blurx(i, j) +
               c_blurx(i, j + 1)) / (uint8_t)3;

    computation c_blury("[N,M]->{c_blury[i,j]: 1<i<N-1 and 1<j<M-1}", e2, true, p_uint8, &dtest_03);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    c_blurx.separate(0, expr((int32_t)SIZE0 / 4), 4, -3);
    c_blury.separate(0, expr((int32_t)SIZE0 / 4), 4, -3);
    c_blurx.get_update(0).rename_computation("c_blurx0");
    c_blurx.get_update(1).rename_computation("c_blurx1");
    c_blury.get_update(0).rename_computation("c_blury0");
    c_blury.get_update(1).rename_computation("c_blury1");
    // TODO this should be handled automatically somehow, or at least wrap it in a better function
    tiramisu::generator::replace_expr_name(c_blury.get_update(0).expression, "c_blurx", "c_blurx0");
    tiramisu::generator::replace_expr_name(c_blury.get_update(1).expression, "c_blurx", "c_blurx1");

    var p("p"), ii("ii"), q("q");
    // Associate ranks with the split computations
    c_blurx.get_update(0).distributed_split(i, (int32_t)SIZE0/4, p, ii);
    c_blurx.get_update(1).distributed_split(i, (int32_t)SIZE0/4, p, ii);
    c_blurx.get_update(0).tag_distribute_level(p);
    c_blurx.get_update(1).tag_distribute_level(p);
    c_blury.get_update(0).distributed_split(i, (int32_t)SIZE0/4, p, ii);
    c_blury.get_update(1).distributed_split(i, (int32_t)SIZE0/4, p, ii);
    c_blury.get_update(0).tag_distribute_level(p);
    c_blury.get_update(1).tag_distribute_level(p);

    // communication
    channel chan_sync_block("chan_sync_block", p_uint8, {FIFO, SYNC, BLOCK, MPI});

    send_recv fan_out = computation::create_transfer(
            "{send_fan_out[q,p,i,j]: 0<=q<1 and 1<=p<4 and p*320-1<=i<(p+1)*320+1 and 0<=j<768}",
            "{recv_fan_out[p,i,j]: 1<=p<4 and p*320-1<=i<(p+1)*320+1 and 0<=j<768}", 0, var("p"), &chan_sync_block,
            &chan_sync_block, c_input(var("i"), var("j")), {&(c_blurx.get_update(1))}, &dtest_03);
    send_recv fan_in = computation::create_transfer(
            "{send_fan_in[p,i,j]: 1<=p<4 and p*320<=i<(p+1)*320 and 0<=j<768}",
            "{recv_fan_in[q,p,i,j]: 0<=q<1 and 1<=p<4 and q*320<=i<(q+1)*320 and 0<=j<768}",
            var("p"), 0, &chan_sync_block, &chan_sync_block, c_blury.get_update(1)(var("i") - var("p") * 320, var("j")), {},
            &dtest_03);

    fan_out.s->tag_distribute_level(q);
    fan_out.r->tag_distribute_level(p);
    fan_in.s->tag_distribute_level(p);
    fan_in.r->tag_distribute_level(q);

    // Order them
    fan_out.s->before(c_blurx.get_update(0), computation::root);
    c_blurx.get_update(0).before(c_blury.get_update(0), computation::root);
    c_blury.get_update(0).before(*fan_out.r, computation::root);
    fan_out.r->before(c_blurx.get_update(1), computation::root);
    c_blurx.get_update(1).before(c_blury.get_update(1), computation::root);
    c_blury.get_update(1).before(*fan_in.s, computation::root);
    fan_in.r->after(*fan_in.s, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_input("b_input", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_input, &dtest_03);
    b_input.distribute({tiramisu::expr(SIZE0/4)+2, tiramisu::expr(SIZE1)}, "b_input_temp");
    buffer b_blurx("b_blurx", {tiramisu::expr(SIZE0)/4, tiramisu::expr(SIZE1)}, p_uint8, a_temporary, &dtest_03);
    buffer b_blury("b_blury", {tiramisu::expr(SIZE0), tiramisu::expr(SIZE1)}, p_uint8, a_output, &dtest_03);
    buffer b_blury_temp("b_blury_temp", {tiramisu::expr(SIZE0)/4, tiramisu::expr(SIZE1)}, p_uint8, a_temporary, &dtest_03);

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.get_update(0).set_access("{c_blurx0[i,j]->b_blurx[i,j]}");
    c_blurx.get_update(1).set_access("{c_blurx1[i,j]->b_blurx[i,j]}");
    c_blury.get_update(0).set_access("{c_blury0[i,j]->b_blury[i,j]}");
    c_blury.get_update(1).set_access("{c_blury1[i,j]->b_blury_temp[i,j]}");
    fan_out.r->set_access("{recv_fan_out[p,i,j]->b_input_temp[i-p*320+1,j]}");
    fan_in.r->set_access("{recv_fan_in[q,p,i,j]->b_blury[p*320+i,j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments to dtest_03
    dtest_03.set_arguments({&b_input, &b_blury});
    // Generate code
    dtest_03.gen_time_space_domain();
    dtest_03.lift_ops_to_library_calls();
    dtest_03.gen_isl_ast();
    dtest_03.gen_halide_stmt();
    dtest_03.gen_halide_obj("build/generated_fct_dtest_03.o");

    // Some debugging
    dtest_03.dump_iteration_domain();
    dtest_03.dump_halide_stmt();

    // Dump all the fields of the dtest_03 class.
    dtest_03.dump(true);

    return 0;
}
