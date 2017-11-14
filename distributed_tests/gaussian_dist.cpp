#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"
#include "sizes.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    tiramisu::function gaussian_tiramisu("gaussian_dist");

    int COLS = _COLS;
    int ROWS = _ROWS;
    int CHANNELS = _CHANNELS;
    int rows_per_node = _ROWS / NODES;
    constant rows("rows", expr(ROWS), p_int32, true, NULL, 0, &gaussian_tiramisu);
    constant cols("cols", expr(COLS), p_int32, true, NULL, 0, &gaussian_tiramisu);
    constant channels("channels", expr(CHANNELS), p_int32, true, NULL, 0, &gaussian_tiramisu);

    tiramisu::computation input("[rows, cols, channels]->{input[i2, i1, i0]: (0 <= i2 <= (channels + -1)) and (0 <= i1 <= (rows + -1)) and (0 <= i0 < (cols))}", expr(), false, tiramisu::p_uint64, &gaussian_tiramisu);

    int kernelx_extent_0 = 5;
    tiramisu::computation kernelx("[kernelx_extent_0]->{kernelx[i0]: (0 <= i0 <= (kernelx_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &gaussian_tiramisu);

    int kernely_extent_0 = 5;

    tiramisu::computation kernely("[kernely_extent_0]->{kernely[i0]: (0 <= i0 <= (kernely_extent_0 + -1))}", expr(), false, tiramisu::p_float32, &gaussian_tiramisu);


    // Define loop bounds for dimension "gaussian_x_s0_c".
    tiramisu::constant gaussian_x_s0_c_loop_min("gaussian_x_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_x_s0_c_loop_extent("gaussian_x_s0_c_loop_extent", tiramisu::expr(CHANNELS), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_x_s0_y".
    tiramisu::constant gaussian_x_s0_y_loop_min("gaussian_x_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_x_s0_y_loop_extent("gaussian_x_s0_y_loop_extent", (tiramisu::expr(ROWS) + tiramisu::expr((int32_t)4)), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_x_s0_x".
    tiramisu::constant gaussian_x_s0_x_loop_min("gaussian_x_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_x_s0_x_loop_extent("gaussian_x_s0_x_loop_extent", tiramisu::expr(COLS), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::computation gaussian_x_s0("[gaussian_x_s0_c_loop_min, gaussian_x_s0_c_loop_extent, gaussian_x_s0_y_loop_min, gaussian_x_s0_y_loop_extent, gaussian_x_s0_x_loop_min, gaussian_x_s0_x_loop_extent]->{gaussian_x_s0[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]: "
                        "(gaussian_x_s0_c_loop_min <= gaussian_x_s0_c <= ((gaussian_x_s0_c_loop_min + gaussian_x_s0_c_loop_extent) + -1)) and (gaussian_x_s0_y_loop_min <= gaussian_x_s0_y < ((gaussian_x_s0_y_loop_min + gaussian_x_s0_y_loop_extent))) and (gaussian_x_s0_x_loop_min <= gaussian_x_s0_x < ((gaussian_x_s0_x_loop_min + gaussian_x_s0_x_loop_extent) + -4))}",
                        (((((tiramisu::expr((float)0) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("gaussian_x_s0_c"), tiramisu::var("gaussian_x_s0_y"), (tiramisu::var("gaussian_x_s0_x") + tiramisu::expr((int32_t)0))))*kernelx(tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("gaussian_x_s0_c"), tiramisu::var("gaussian_x_s0_y"), (tiramisu::var("gaussian_x_s0_x") + tiramisu::expr((int32_t)1))))*kernelx(tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("gaussian_x_s0_c"), tiramisu::var("gaussian_x_s0_y"), (tiramisu::var("gaussian_x_s0_x") + tiramisu::expr((int32_t)2))))*kernelx(tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("gaussian_x_s0_c"), tiramisu::var("gaussian_x_s0_y"), (tiramisu::var("gaussian_x_s0_x") + tiramisu::expr((int32_t)3))))*kernelx(tiramisu::expr((int32_t)3)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("gaussian_x_s0_c"), tiramisu::var("gaussian_x_s0_y"), (tiramisu::var("gaussian_x_s0_x") + tiramisu::expr((int32_t)4))))*kernelx(tiramisu::expr((int32_t)4)))), true, tiramisu::p_float32, &gaussian_tiramisu);


    // Define loop bounds for dimension "gaussian_s0_c".
    tiramisu::constant gaussian_s0_c_loop_min("gaussian_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_s0_c_loop_extent("gaussian_s0_c_loop_extent", tiramisu::expr(CHANNELS), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_s0_y".
    tiramisu::constant gaussian_s0_y_loop_min("gaussian_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_s0_y_loop_extent("gaussian_s0_y_loop_extent", tiramisu::expr(ROWS), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);

    // Define loop bounds for dimension "gaussian_s0_x".
    tiramisu::constant gaussian_s0_x_loop_min("gaussian_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant gaussian_s0_x_loop_extent("gaussian_s0_x_loop_extent", tiramisu::expr(COLS), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::computation gaussian_s0("[gaussian_s0_c_loop_min, gaussian_s0_c_loop_extent, gaussian_s0_y_loop_min, gaussian_s0_y_loop_extent, gaussian_s0_x_loop_min, gaussian_s0_x_loop_extent]->{gaussian_s0[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]: "
                        "(gaussian_s0_c_loop_min <= gaussian_s0_c <= ((gaussian_s0_c_loop_min + gaussian_s0_c_loop_extent) + -1)) and (gaussian_s0_y_loop_min <= gaussian_s0_y < ((gaussian_s0_y_loop_min + gaussian_s0_y_loop_extent - 4))) and (gaussian_s0_x_loop_min <= gaussian_s0_x < ((gaussian_s0_x_loop_min + gaussian_s0_x_loop_extent) + -4))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, (((((tiramisu::expr((float)0) + (gaussian_x_s0(tiramisu::var("gaussian_s0_c"), (tiramisu::var("gaussian_s0_y") + tiramisu::expr((int32_t)0)), tiramisu::var("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)0)))) + (gaussian_x_s0(tiramisu::var("gaussian_s0_c"), (tiramisu::var("gaussian_s0_y") + tiramisu::expr((int32_t)1)), tiramisu::var("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)1)))) + (gaussian_x_s0(tiramisu::var("gaussian_s0_c"), (tiramisu::var("gaussian_s0_y") + tiramisu::expr((int32_t)2)), tiramisu::var("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)2)))) + (gaussian_x_s0(tiramisu::var("gaussian_s0_c"), (tiramisu::var("gaussian_s0_y") + tiramisu::expr((int32_t)3)), tiramisu::var("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)3)))) + (gaussian_x_s0(tiramisu::var("gaussian_s0_c"), (tiramisu::var("gaussian_s0_y") + tiramisu::expr((int32_t)4)), tiramisu::var("gaussian_s0_x"))*kernely(tiramisu::expr((int32_t)4))))), true, tiramisu::p_uint64, &gaussian_tiramisu);

    var y1("y1"), y2("y2"), q("q"), c("c"), y("y"), x("x");
    
    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    tiramisu::constant three("three", tiramisu::expr(3), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    constant nodes("nodes", expr(NODES), p_int32, true, NULL, 0, &gaussian_tiramisu);
    constant nodes_minus_one("nodes_minus_one", expr(NODES-1), p_int32, true, NULL, 0, &gaussian_tiramisu);
    constant nodes_minus_two("nodes_minus_two", expr(NODES-2), p_int32, true, NULL, 0, &gaussian_tiramisu);
    
    gaussian_x_s0.interchange(var("gaussian_x_s0_c"), var("gaussian_x_s0_y"));
    gaussian_x_s0.split(var("gaussian_x_s0_y"), rows_per_node, y1, y2);
    gaussian_x_s0.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    gaussian_x_s0.get_update(0).rename_computation("gaussian_x_s0_0");
    gaussian_x_s0.get_update(1).rename_computation("gaussian_x_s0_1");
    gaussian_x_s0.get_update(2).rename_computation("gaussian_x_s0_2");

    gaussian_s0.interchange(var("gaussian_s0_c"), var("gaussian_s0_y"));
    gaussian_s0.split(var("gaussian_s0_y"), rows_per_node, y1, y2);
    gaussian_s0.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    gaussian_s0.get_update(0).rename_computation("gaussian_s0_0");
    gaussian_s0.get_update(1).rename_computation("gaussian_s0_1");
    gaussian_s0.get_update(2).rename_computation("gaussian_s0_2");
    
    generator::replace_expr_name(gaussian_s0.get_update(0).expression, "gaussian_x_s0", "gaussian_x_s0_0");
    generator::replace_expr_name(gaussian_s0.get_update(1).expression, "gaussian_x_s0", "gaussian_x_s0_1");
    generator::replace_expr_name(gaussian_s0.get_update(2).expression, "gaussian_x_s0", "gaussian_x_s0_2");

    communication_prop sync_block("sync_block", p_float32, {FIFO, SYNC, BLOCK, MPI});
    communication_prop async_nonblock("async_nonblock", p_float32, {FIFO, ASYNC, NONBLOCK, MPI});
    communication_prop async_block("async_block", p_float32, {FIFO, ASYNC, BLOCK, MPI});

    // transfer the computed rows from gaussian_x    
    send_recv gaussian_x_exchange = computation::create_transfer("[one, nodes_minus_one, cols, channels]->{gaussian_exchange_s[q,c,y,x]: one<=q<nodes_minus_one and 0<=y<4 and 0<=x<cols-4 and 0<=c<channels}",
							 "[nodes_minus_two, cols, channels]->{gaussian_exchange_r[q,c,y,x]: 0<=c<channels and 0<=q<nodes_minus_two and 0<=y<4 and 0<=x<cols-4}",
								 q, q-1, q+1, q, async_block, sync_block,
								 gaussian_x_s0.get_update(1)(c,y,x), &gaussian_tiramisu);
    
    send_recv gaussian_x_exchange_last_node = computation::create_transfer("[channels, one, nodes_minus_one, nodes, cols]->{gaussian_exchange_last_node_s[q,c,y,x]: nodes_minus_one<=q<nodes and 0<=y<4 and 0<=x<cols-4 and 0<=c<channels}",
									  "[nodes_minus_one, nodes_minus_two, cols, channels]->{gaussian_exchange_last_node_r[q,c,y,x]: nodes_minus_two<=q<nodes_minus_one and 0<=y<4 and 0<=x<cols-4 and 0<=c<channels}",
									  q, q-1, q+1, q, async_block, sync_block,
									   gaussian_x_s0.get_update(2)(c,y,x), &gaussian_tiramisu);

    gaussian_x_exchange.s->tag_distribute_level(q);
    gaussian_x_exchange.r->tag_distribute_level(q);

    gaussian_x_exchange_last_node.s->tag_distribute_level(q);
    gaussian_x_exchange_last_node.r->tag_distribute_level(q);

    gaussian_x_s0.get_update(0).tag_distribute_level(y1);
    gaussian_x_s0.get_update(1).tag_distribute_level(y1);
    gaussian_x_s0.get_update(2).tag_distribute_level(y1);
    gaussian_x_s0.get_update(0).drop_rank_iter();
    gaussian_x_s0.get_update(1).drop_rank_iter();
    gaussian_x_s0.get_update(2).drop_rank_iter();

    gaussian_s0.get_update(0).tag_distribute_level(y1);
    gaussian_s0.get_update(1).tag_distribute_level(y1);
    gaussian_s0.get_update(2).tag_distribute_level(y1);
    gaussian_s0.get_update(0).drop_rank_iter();
    gaussian_s0.get_update(1).drop_rank_iter();
    gaussian_s0.get_update(2).drop_rank_iter();

    gaussian_x_s0.get_update(0).before(gaussian_x_s0.get_update(1), computation::root);
    gaussian_x_s0.get_update(1).before(gaussian_x_s0.get_update(2), computation::root);
    gaussian_x_s0.get_update(2).before(*gaussian_x_exchange.s, computation::root);
    gaussian_x_exchange.s->before(*gaussian_x_exchange_last_node.s, computation::root);
    gaussian_x_exchange_last_node.s->before(*gaussian_x_exchange.r, computation::root);
    gaussian_x_exchange.r->before(*gaussian_x_exchange_last_node.r, computation::root);
    gaussian_x_exchange_last_node.r->before(gaussian_s0.get_update(0), computation::root);
    gaussian_s0.get_update(0).before(gaussian_s0.get_update(1), computation::root);
    gaussian_s0.get_update(1).before(gaussian_s0.get_update(2), computation::root);

    gaussian_x_exchange.s->collapse_many({collapser(3, 0, COLS-4), collapser(2, 0, 4)});//, collapser(1, 0, CHANNELS)});
    gaussian_x_exchange.r->collapse_many({collapser(3, 0, COLS-4), collapser(2, 0, 4)});//, collapser(1, 0, CHANNELS)});
    gaussian_x_exchange_last_node.s->collapse_many({collapser(3, 0, COLS-4), collapser(2, 0, 4)});//, collapser(1, 0, CHANNELS)});
    gaussian_x_exchange_last_node.r->collapse_many({collapser(3, 0, COLS-4), collapser(2, 0, 4)});//, collapser(1, 0, CHANNELS)});

    //    gaussian_s0.get_update(0).set_schedule_this_comp(false);
    //    gaussian_s0.get_update(1).set_schedule_this_comp(false);
    //    gaussian_s0.get_update(2).set_schedule_this_comp(false);
    //    gaussian_x_exchange.s->set_schedule_this_comp(false);
    //    gaussian_x_exchange.r->set_schedule_this_comp(false);
    //    gaussian_x_exchange_last_node.s->set_schedule_this_comp(false);
    //    gaussian_x_exchange_last_node.r->set_schedule_this_comp(false);
    // Buffers
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node), tiramisu::expr(COLS)}, tiramisu::p_uint64, tiramisu::a_input, &gaussian_tiramisu);
    tiramisu::buffer buff_kernelx("buff_kernelx", {tiramisu::expr(kernelx_extent_0)}, tiramisu::p_float32, tiramisu::a_input, &gaussian_tiramisu);
    tiramisu::buffer buff_kernely("buff_kernely", {tiramisu::expr(kernely_extent_0)}, tiramisu::p_float32, tiramisu::a_input, &gaussian_tiramisu);

    tiramisu::buffer buff_gaussian_x("buff_gaussian_x", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node + 4), tiramisu::expr(COLS-4)}, tiramisu::p_float32, tiramisu::a_temporary, &gaussian_tiramisu);
    tiramisu::buffer buff_gaussian_x_last("buff_gaussian_x_last", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node), tiramisu::expr(COLS-4)}, tiramisu::p_float32, tiramisu::a_temporary, &gaussian_tiramisu);

    tiramisu::buffer buff_gaussian("buff_gaussian", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node), tiramisu::expr(COLS-4)}, tiramisu::p_uint64, tiramisu::a_output, &gaussian_tiramisu);
    tiramisu::buffer buff_gaussian_last("buff_gaussian_last", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node - 4), tiramisu::expr(COLS-4)}, tiramisu::p_uint64, tiramisu::a_output, &gaussian_tiramisu);

    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    kernelx.set_access("{kernelx[i0]->buff_kernelx[i0]}");
    kernely.set_access("{kernely[i0]->buff_kernely[i0]}");
    gaussian_x_s0.get_update(0).set_access("{gaussian_x_s0_0[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]->buff_gaussian_x[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]}");
    gaussian_x_s0.get_update(1).set_access("{gaussian_x_s0_1[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]->buff_gaussian_x[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]}");
    gaussian_x_s0.get_update(2).set_access("{gaussian_x_s0_2[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]->buff_gaussian_x_last[gaussian_x_s0_c, gaussian_x_s0_y, gaussian_x_s0_x]}");
    gaussian_s0.get_update(0).set_access("{gaussian_s0_0[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]->buff_gaussian[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]}");
    gaussian_s0.get_update(1).set_access("{gaussian_s0_1[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]->buff_gaussian[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]}");
    gaussian_s0.get_update(2).set_access("{gaussian_s0_2[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]->buff_gaussian_last[gaussian_s0_c, gaussian_s0_y, gaussian_s0_x]}");

    gaussian_x_exchange.r->set_access("{gaussian_exchange_r[q,c,y,x]->buff_gaussian_x[c, " + std::to_string(rows_per_node) + " + y, x]}");
    gaussian_x_exchange_last_node.r->set_access("{gaussian_exchange_last_node_r[q,c,y,x]->buff_gaussian_x[c, " + std::to_string(rows_per_node) + " + y, x]}");

    // Add schedules.

    gaussian_tiramisu.set_arguments({&buff_input, &buff_kernelx, &buff_kernely, /*&buff_gaussian_x, &buff_gaussian_x_last,*/ &buff_gaussian, &buff_gaussian_last});
    gaussian_tiramisu.gen_time_space_domain();
    gaussian_tiramisu.lift_ops_to_library_calls();
    gaussian_tiramisu.gen_isl_ast();
    gaussian_tiramisu.gen_halide_stmt();
    gaussian_tiramisu.gen_halide_obj("build/generated_gaussian_dist.o");
    gaussian_tiramisu.dump_halide_stmt();

    return 0;
}
