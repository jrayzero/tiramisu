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

#define NODES 5

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

    tiramisu::computation kernel("{kernel[i, j]: 0<=i<3 and 0<=j<3}", expr(), false, tiramisu::p_float32, &filter2D_dist);
    tiramisu::computation input("[channels, rows, cols]->{input[c, i, j]: 0<=c<channels and 0<=i<rows and 0<=j<cols}", expr(), false, tiramisu::p_float32, &filter2D_dist);

    var i("i"), j("j"), c("c");
    tiramisu::expr filter2D_expr(input(c, i, j) * kernel(i,j) + input(c, i+1, j) * kernel(i,j) + input(c, i+2, j) * kernel(i,j)
                                 + input(c, i, j+1) * kernel(i,j) + input(c, i+1, j+1) * kernel(i,j) + input(c, i+2, j+1) * kernel(i,j)
                                 + input(c, i, j+2) * kernel(i,j) + input(c, i+1, j+2) * kernel(i,j) + input(c, i+2, j+2) * kernel(i,j));
    tiramisu::computation filter2D("[channels, rows, cols]->{filter2D[c, i, j]: 0<=c<channels and 0<=i<rows-2 and 0<=j<cols-2}",
                                   filter2D_expr,
                                   true, tiramisu::p_float32, &filter2D_dist);

    constant one("one", expr(1), p_int32, true, NULL, 0, &filter2D_dist);
    constant two("two", expr(2), p_int32, true, NULL, 0, &filter2D_dist);
    constant nodes("nodes", expr(NODES), p_int32, true, NULL, 0, &filter2D_dist);
    constant nodes_minus_one("nodes_minus_one", expr(NODES-1), p_int32, true, NULL, 0, &filter2D_dist);
    constant nodes_minus_two("nodes_minus_two", expr(NODES-2), p_int32, true, NULL, 0, &filter2D_dist);
    constant _rows_per_node("_rows_per_node", expr(rows_per_node), p_int32, true, NULL, 0, &filter2D_dist);

    var i1("i1"), i2("i2"), q("q");
    filter2D.interchange(c, i);
    filter2D.split(i, rows_per_node, i1, i2);
    filter2D.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    filter2D.get_update(0).rename_computation("filter2D_0");
    filter2D.get_update(1).rename_computation("filter2D_1");
    filter2D.get_update(2).rename_computation("filter2D_2");

    /*
     * Create communication
     */

    channel sync_block("sync_block", p_float32, {FIFO, SYNC, BLOCK, MPI});
    channel async_block("async_block", p_float32, {FIFO, ASYNC, BLOCK, MPI});
    send_recv filter2D_exchange = computation::create_transfer("[one, nodes_minus_one, cols, channels]->{filter2D_exchange_s[q,c,i,j]: one<=q<nodes_minus_one and 0<=i<2 and 0<=j<cols and 0<=c<channels}",
                                                               "[nodes_minus_two, cols, channels]->{filter2D_exchange_r[q,c,i,j]: 0<=q<nodes_minus_two and 0<=i<2 and 0<=j<cols and 0<=c<channels}",
                                                               q, q-1, q+1, q, async_block, sync_block,
                                                               input(c,i,j), &filter2D_dist);
    send_recv filter2D_exchange_last_node = computation::create_transfer("[one, nodes_minus_one, nodes_minus_two, nodes, cols, channels]->{filter2D_exchange_last_node_s[q,c,i,j]: nodes_minus_one<=q<nodes and 0<=i<2 and 0<=j<cols and 0<=c<channels}",
                                                                         "[nodes_minus_one, nodes_minus_two, cols, channels]->{filter2D_exchange_last_node_r[q,c,i,j]: nodes_minus_two<=q<nodes_minus_one and 0<=i<2 and 0<=j<cols and 0<=c<channels}",
                                                                         q, q-1, q+1, q, async_block, sync_block,
                                                                         input(c,i,j), &filter2D_dist);

    filter2D.get_update(0).tag_distribute_level(i1);
    filter2D.get_update(0).drop_rank_iter();
    filter2D.get_update(1).tag_distribute_level(i1);
    filter2D.get_update(1).drop_rank_iter();
    filter2D.get_update(2).tag_distribute_level(i1);
    filter2D.get_update(2).drop_rank_iter();
    filter2D_exchange.s->tag_distribute_level(q);
    filter2D_exchange.r->tag_distribute_level(q);
    filter2D_exchange_last_node.s->tag_distribute_level(q);
    filter2D_exchange_last_node.r->tag_distribute_level(q);

    filter2D_exchange.s->before(*filter2D_exchange.r, computation::root);
    filter2D_exchange.r->before(*filter2D_exchange_last_node.s, computation::root);
    filter2D_exchange_last_node.s->before(*filter2D_exchange_last_node.r, computation::root);
    filter2D_exchange_last_node.r->before(filter2D.get_update(0), computation::root);
    filter2D.get_update(0).before(filter2D.get_update(1), computation::root);
    filter2D.get_update(1).before(filter2D.get_update(2), computation::root);

    filter2D_exchange.s->collapse_many({collapser(3, 0, COLS)});
    filter2D_exchange.r->collapse_many({collapser(3, 0, COLS)});
    filter2D_exchange_last_node.s->collapse_many({collapser(3, 0, COLS)});
    filter2D_exchange_last_node.r->collapse_many({collapser(3, 0, COLS)});

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node) + 2, tiramisu::expr(COLS)}, tiramisu::p_float32, tiramisu::a_input, &filter2D_dist);
    tiramisu::buffer buff_kernel("buff_kernel", {tiramisu::expr(3), tiramisu::expr(3)}, tiramisu::p_float32, tiramisu::a_input, &filter2D_dist);
    tiramisu::buffer buff_filter2D("buff_filter2D", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node), tiramisu::expr(COLS) - 2}, tiramisu::p_float32, tiramisu::a_output, &filter2D_dist);
    tiramisu::buffer buff_filter2D_last_node("buff_filter2D_last_node", {tiramisu::expr(CHANNELS), tiramisu::expr(rows_per_node) - 2, tiramisu::expr(COLS) - 2}, tiramisu::p_float32, tiramisu::a_output, &filter2D_dist);

    input.set_access("{input[c, i, j]->buff_input[c, i, j]}");
    kernel.set_access("{kernel[i, j]->buff_kernel[i, j]}");
    filter2D.get_update(0).set_access("{filter2D_0[c, i, j]->buff_filter2D[c, i, j]}");
    filter2D.get_update(1).set_access("{filter2D_1[c, i, j]->buff_filter2D[c, i, j]}");
    filter2D.get_update(2).set_access("{filter2D_2[c, i, j]->buff_filter2D_last_node[c, i, j]}");
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

