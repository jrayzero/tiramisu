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

    tiramisu::function filter2D_dist("filter2D_dist");
    int COLS = _COLS;
    int ROWS = _ROWS;
    int CHANNELS = _CHANNELS;

    int rows_per_node = _ROWS / NODES;
    constant rows("rows", expr(ROWS), p_int32, true, NULL, 0, &filter2D_dist);
    constant cols("cols", expr(COLS), p_int32, true, NULL, 0, &filter2D_dist);
    constant channels("channels", expr(CHANNELS), p_int32, true, NULL, 0, &filter2D_dist);

    int kernel_extent_1 = 3;
    int kernel_extent_0 = 3;
    tiramisu::computation kernel("[kernel_extent_1, kernel_extent_0]->{kernel[i1, i0]: (0 <= i1 < (kernel_extent_1)) and (0 <= i0 < (kernel_extent_0))}", expr(), false, tiramisu::p_float32, &filter2D_dist);
    tiramisu::computation input("[channels, rows, cols]->{input[i2, i1, i0]: (0 <= i2 < (channels)) and (0 <= i1 < (rows)) and (0 <= i0 < (cols))}", expr(), false, tiramisu::p_uint64, &filter2D_dist);

    // Define loop bounds for dimension "filter2D_s0_c".
    tiramisu::constant filter2D_s0_c_loop_min("filter2D_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &filter2D_dist);
    tiramisu::constant filter2D_s0_c_loop_extent("filter2D_s0_c_loop_extent", tiramisu::expr(CHANNELS), tiramisu::p_int32, true, NULL, 0, &filter2D_dist);

    // Define loop bounds for dimension "filter2D_s0_y".
    tiramisu::constant filter2D_s0_y_loop_min("filter2D_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &filter2D_dist);
    tiramisu::constant filter2D_s0_y_loop_extent("filter2D_s0_y_loop_extent", tiramisu::expr(rows) - 2, tiramisu::p_int32, true, NULL, 0, &filter2D_dist);

    // Define loop bounds for dimension "filter2D_s0_x".
    tiramisu::constant filter2D_s0_x_loop_min("filter2D_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &filter2D_dist);
    tiramisu::constant filter2D_s0_x_loop_extent("filter2D_s0_x_loop_extent", tiramisu::expr(COLS) - 2, tiramisu::p_int32, true, NULL, 0, &filter2D_dist);
    tiramisu::computation filter2D_s0("[filter2D_s0_c_loop_min, filter2D_s0_c_loop_extent, filter2D_s0_y_loop_min, filter2D_s0_y_loop_extent, filter2D_s0_x_loop_min, filter2D_s0_x_loop_extent]->{filter2D_s0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]: "
                        "(filter2D_s0_c_loop_min <= filter2D_s0_c < ((filter2D_s0_c_loop_min + filter2D_s0_c_loop_extent))) and (filter2D_s0_y_loop_min <= filter2D_s0_y < ((filter2D_s0_y_loop_min + filter2D_s0_y_loop_extent))) and (filter2D_s0_x_loop_min <= filter2D_s0_x < ((filter2D_s0_x_loop_min + filter2D_s0_x_loop_extent)))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, (((((((((tiramisu::expr((float)0) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)0)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)0), tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)1)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)1), tiramisu::expr((int32_t)2)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)0))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)0)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)1))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)1)))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(tiramisu::var("filter2D_s0_c"), (tiramisu::var("filter2D_s0_y") + tiramisu::expr((int32_t)2)), (tiramisu::var("filter2D_s0_x") + tiramisu::expr((int32_t)2))))*kernel(tiramisu::expr((int32_t)2), tiramisu::expr((int32_t)2))))), true, tiramisu::p_uint64, &filter2D_dist);

    constant one("one", expr(1), p_int32, true, NULL, 0, &filter2D_dist);
    constant two("two", expr(2), p_int32, true, NULL, 0, &filter2D_dist);
    constant nodes("nodes", expr(NODES), p_int32, true, NULL, 0, &filter2D_dist);
    constant nodes_minus_one("nodes_minus_one", expr(NODES-1), p_int32, true, NULL, 0, &filter2D_dist);
    constant nodes_minus_two("nodes_minus_two", expr(NODES-2), p_int32, true, NULL, 0, &filter2D_dist);
    constant _rows_per_node("_rows_per_node", expr(rows_per_node), p_int32, true, NULL, 0, &filter2D_dist);

    var y1("y1"), y2("y2"), d("d"), q("q"), c("c"), _c("filter2D_s0_c"), y("filter2D_s0_y"), x("x");
    filter2D_s0.interchange(_c, y);
    filter2D_s0.split(y, rows_per_node, y1, y2);
    filter2D_s0.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    filter2D_s0.get_update(0).rename_computation("filter2D_0");
    filter2D_s0.get_update(1).rename_computation("filter2D_1");
    filter2D_s0.get_update(2).rename_computation("filter2D_2");

    /*
     * Create communication
     */

    channel sync_block("sync_block", p_uint64, {FIFO, SYNC, BLOCK, MPI});
    channel async_nonblock("async_nonblock", p_uint64, {FIFO, ASYNC, NONBLOCK, MPI});
    channel async_block("async_block", p_uint64, {FIFO, ASYNC, BLOCK, MPI});
    send_recv filter2D_exchange = computation::create_transfer("[one, nodes_minus_one, cols, channels]->{filter2D_exchange_s[q,c,y,x]: one<=q<nodes_minus_one and 0<=y<2 and 0<=x<cols and 0<=c<channels}",
							       "[nodes_minus_two, cols, channels]->{filter2D_exchange_r[q,c,y,x]: 0<=q<nodes_minus_two and 0<=y<2 and 0<=x<cols and 0<=c<channels}",
							       q, q-1, q+1, q, async_nonblock, sync_block,
							       input(c,var("y"),x), &filter2D_dist);
    send_recv filter2D_exchange_last_node = computation::create_transfer("[one, nodes_minus_one, nodes_minus_two, nodes, cols, channels]->{filter2D_exchange_last_node_s[q,c,y,x]: nodes_minus_one<=q<nodes and 0<=y<2 and 0<=x<cols and 0<=c<channels}",
									 "[nodes_minus_one, nodes_minus_two, cols, channels]->{filter2D_exchange_last_node_r[q,c,y,x]: nodes_minus_two<=q<nodes_minus_one and 0<=y<2 and 0<=x<cols and 0<=c<channels}",
									 q, q-1, q+1, q, async_nonblock, sync_block,
									 input(c,var("y"),x), &filter2D_dist);

    filter2D_s0.get_update(0).tag_distribute_level(y1);
    filter2D_s0.get_update(0).drop_rank_iter();
    filter2D_s0.get_update(1).tag_distribute_level(y1);
    filter2D_s0.get_update(1).drop_rank_iter();
    filter2D_s0.get_update(2).tag_distribute_level(y1);
    filter2D_s0.get_update(2).drop_rank_iter();
    filter2D_exchange.s->tag_distribute_level(q);
    filter2D_exchange.r->tag_distribute_level(q);
    filter2D_exchange_last_node.s->tag_distribute_level(q);
    filter2D_exchange_last_node.r->tag_distribute_level(q);

    filter2D_exchange.s->before(*filter2D_exchange.r, computation::root);
    filter2D_exchange.r->before(*filter2D_exchange_last_node.s, computation::root);
    filter2D_exchange_last_node.s->before(*filter2D_exchange_last_node.r, computation::root);
    filter2D_exchange_last_node.r->before(filter2D_s0.get_update(0), computation::root);
    filter2D_s0.get_update(0).before(filter2D_s0.get_update(1), computation::root);
    filter2D_s0.get_update(1).before(filter2D_s0.get_update(2), computation::root);    

    filter2D_exchange.s->collapse_many({collapser(3, 0, COLS), collapser(2, 0, 2), collapser(1, 0, 3)}); 
    filter2D_exchange.r->collapse_many({collapser(3, 0, COLS), collapser(2, 0, 2), collapser(1, 0, 3)}); 
    filter2D_exchange_last_node.s->collapse_many({collapser(3, 0, COLS), collapser(2, 0, 2), collapser(1, 0, 3)}); 
    filter2D_exchange_last_node.r->collapse_many({collapser(3, 0, COLS), collapser(2, 0, 2), collapser(1, 0, 3)}); 

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node) + 2, tiramisu::expr(COLS)}, tiramisu::p_uint64, tiramisu::a_input, &filter2D_dist);
    //    tiramisu::buffer buff_input_last_node("buff_input_last_node", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node), tiramisu::expr(COLS)}, tiramisu::p_uint64, tiramisu::a_input, &filter2D_dist);
    tiramisu::buffer buff_filter2D("buff_filter2D", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node), tiramisu::expr(COLS) - 2}, tiramisu::p_uint64, tiramisu::a_output, &filter2D_dist);
    tiramisu::buffer buff_filter2D_last_node("buff_filter2D_last_node", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node) - 2, tiramisu::expr(COLS) - 2}, tiramisu::p_uint64, tiramisu::a_output, &filter2D_dist);
    tiramisu::buffer buff_kernel("buff_kernel", {tiramisu::expr(kernel_extent_1), tiramisu::expr(kernel_extent_0)}, tiramisu::p_float32, tiramisu::a_input, &filter2D_dist);

    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    kernel.set_access("{kernel[i1, i0]->buff_kernel[i1, i0]}");
    filter2D_s0.get_update(0).set_access("{filter2D_0[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->buff_filter2D[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]}");
    filter2D_s0.get_update(1).set_access("{filter2D_1[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->buff_filter2D[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]}");
    filter2D_s0.get_update(2).set_access("{filter2D_2[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]->buff_filter2D_last_node[filter2D_s0_c, filter2D_s0_y, filter2D_s0_x]}");
    filter2D_exchange.r->set_access("{filter2D_exchange_r[q,c,y,x]->buff_input[c, " + std::to_string(rows_per_node) + " + y, x]}");
    filter2D_exchange_last_node.r->set_access("{filter2D_exchange_last_node_r[q,c,y,x]->buff_input[c, " + std::to_string(rows_per_node) + " + y, x]}");

    filter2D_dist.set_arguments({&buff_input, &buff_kernel, &buff_filter2D, &buff_filter2D_last_node});
    filter2D_dist.gen_time_space_domain();
    filter2D_dist.lift_ops_to_library_calls();
    filter2D_dist.gen_isl_ast();
    filter2D_dist.gen_halide_stmt();
    filter2D_dist.dump_halide_stmt();
    filter2D_dist.gen_halide_obj("build/generated_filter2D_dist.o");

    return 0;
}

