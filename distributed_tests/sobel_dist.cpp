//
// Created by Jessica Ray on 11/12/17.
//

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

    tiramisu::function sobel_dist("sobel_dist");

    int COLS = _COLS;
    int ROWS = _ROWS;
    int ROWS_PER_NODE = _ROWS / NODES;

    var i("i"), j("j");
    computation input("[rows, cols]->{input[i,j]: 0<=i<rows and 0<=j<cols}", expr(), false, tiramisu::p_float32, &sobel_dist);
    computation sobel_x("[rows, cols]->{sobel_x[i,j]: 0<=i<rows-2 and 0<=j<cols-2}",
                        expr(input(i, j) + -1.0f*input(i, j+2) + 2.0f*input(i+1, j) + -2.0f*input(i+1, j+2) + input(i+2, j) + -1.0f*input(i+2, j+2)),
                        true, tiramisu::p_float32, &sobel_dist);
    computation sobel_y("[rows, cols]->{sobel_y[i,j]: 0<=i<rows-2 and 0<=j<cols-2}",
                        expr(input(i, j) + 2.0f*input(i, j+1) + 1.0f*input(i, j+2) + -1.0f*input(i+2, j) + -2.0f*input(i+2, j+1) + -1.0f*input(i+2, j+2)),
                        true, tiramisu::p_float32, &sobel_dist);
    computation sobel("[rows, cols]->{sobel[i,j]: 0<=i<rows-2 and 0<=j<cols-2}",
                      expr(tiramisu::o_sqrt, sobel_x(i,j) * sobel_x(i,j) + sobel_y(i, j) * sobel_y(i,j)), true, tiramisu::p_float32, &sobel_dist);

    constant rows("rows", expr(ROWS), p_int32, true, NULL, 0, &sobel_dist);
    constant cols("cols", expr(COLS), p_int32, true, NULL, 0, &sobel_dist);
    constant one("one", expr(1), p_int32, true, NULL, 0, &sobel_dist);
    constant two("two", expr(2), p_int32, true, NULL, 0, &sobel_dist);
    constant nodes("nodes", expr(NODES), p_int32, true, NULL, 0, &sobel_dist);
    constant nodes_minus_one("nodes_minus_one", expr(NODES-1), p_int32, true, NULL, 0, &sobel_dist);
    constant nodes_minus_two("nodes_minus_two", expr(NODES-2), p_int32, true, NULL, 0, &sobel_dist);
    constant rows_per_node("rows_per_node", expr(ROWS_PER_NODE), p_int32, true, NULL, 0, &sobel_dist);

    var i1("i1"), i2("i2"), d("d"), q("q");
    sobel_x.split(i, ROWS_PER_NODE, i1, i2);
    sobel_x.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    sobel_x.get_update(0).rename_computation("sobel_x_0");
    sobel_x.get_update(1).rename_computation("sobel_x_1");
    sobel_x.get_update(2).rename_computation("sobel_x_2");
    sobel_y.split(i, ROWS_PER_NODE, i1, i2);
    sobel_y.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    sobel_y.get_update(0).rename_computation("sobel_y_0");
    sobel_y.get_update(1).rename_computation("sobel_y_1");
    sobel_y.get_update(2).rename_computation("sobel_y_2");
    sobel.split(i, ROWS_PER_NODE, i1, i2);
    sobel.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    sobel.get_update(0).rename_computation("sobel_0");
    sobel.get_update(1).rename_computation("sobel_1");
    sobel.get_update(2).rename_computation("sobel_2");

    generator::replace_expr_name(sobel.get_update(0).expression, "sobel_x", "sobel_x_0");
    generator::replace_expr_name(sobel.get_update(0).expression, "sobel_y", "sobel_y_0");
    generator::replace_expr_name(sobel.get_update(1).expression, "sobel_x", "sobel_x_1");
    generator::replace_expr_name(sobel.get_update(1).expression, "sobel_y", "sobel_y_1");
    generator::replace_expr_name(sobel.get_update(2).expression, "sobel_x", "sobel_x_2");
    generator::replace_expr_name(sobel.get_update(2).expression, "sobel_y", "sobel_y_2");

    channel sync_block("sync_block", p_float32, {FIFO, SYNC, BLOCK, MPI});
    channel async_block("async_block", p_float32, {FIFO, ASYNC, BLOCK, MPI});

    // transfer the computed rows from gaussian_x
    send_recv exchange = computation::create_transfer("[one, nodes_minus_one, cols]->{exchange_s[q,i,j]: one<=q<nodes_minus_one and 0<=i<2 and 0<=j<cols}",
                                                      "[nodes_minus_two, cols]->{exchange_r[q,i,j]: 0<=q<nodes_minus_two and 0<=i<2 and 0<=j<cols}",
                                                      q, q-1, q+1, q, async_block, sync_block,
                                                      input(i,j), &sobel_dist);

    send_recv exchange_last_node = computation::create_transfer("[one, nodes_minus_one, nodes, cols]->{exchange_last_node_s[q,i,j]: nodes_minus_one<=q<nodes and 0<=i<2 and 0<=j<cols}",
                                                                "[nodes_minus_one, nodes_minus_two, cols]->{exchange_last_node_r[q,i,j]: nodes_minus_two<=q<nodes_minus_one and 0<=i<2 and 0<=j<cols}",
                                                                q, q-1, q+1, q, async_block, sync_block,
                                                                input(i,j), &sobel_dist);

    sobel_x.get_update(0).tag_distribute_level(i1);
    sobel_x.get_update(1).tag_distribute_level(i1);
    sobel_x.get_update(2).tag_distribute_level(i1);
    sobel_x.get_update(0).drop_rank_iter();
    sobel_x.get_update(1).drop_rank_iter();
    sobel_x.get_update(2).drop_rank_iter();

    sobel_y.get_update(0).tag_distribute_level(i1);
    sobel_y.get_update(1).tag_distribute_level(i1);
    sobel_y.get_update(2).tag_distribute_level(i1);
    sobel_y.get_update(0).drop_rank_iter();
    sobel_y.get_update(1).drop_rank_iter();
    sobel_y.get_update(2).drop_rank_iter();

    sobel.get_update(0).tag_distribute_level(i1);
    sobel.get_update(1).tag_distribute_level(i1);
    sobel.get_update(2).tag_distribute_level(i1);
    sobel.get_update(0).drop_rank_iter();
    sobel.get_update(1).drop_rank_iter();
    sobel.get_update(2).drop_rank_iter();

    exchange.s->tag_distribute_level(q);
    exchange.r->tag_distribute_level(q);
    exchange_last_node.s->tag_distribute_level(q);
    exchange_last_node.r->tag_distribute_level(q);

    exchange.s->collapse_many({collapser(2, 0, COLS)});
    exchange.r->collapse_many({collapser(2, 0, COLS)});
    exchange_last_node.s->collapse_many({collapser(2, 0, COLS)});
    exchange_last_node.r->collapse_many({collapser(2, 0, COLS)});

    exchange.s->before(*exchange.r, computation::root);
    exchange.r->before(*exchange_last_node.s, computation::root);
    exchange_last_node.s->before(*exchange_last_node.r, computation::root);
    exchange_last_node.r->before(sobel_x.get_update(0), computation::root);
    sobel_x.get_update(0).before(sobel_y.get_update(0), computation::root);
    sobel_y.get_update(0).before(sobel.get_update(0), computation::root);
    sobel.get_update(0).before(sobel_x.get_update(1), computation::root);
    sobel_x.get_update(1).before(sobel_y.get_update(1), computation::root);
    sobel_y.get_update(1).before(sobel.get_update(1), computation::root);
    sobel.get_update(1).before(sobel_x.get_update(2), computation::root);
    sobel_x.get_update(2).before(sobel_y.get_update(2), computation::root);
    sobel_y.get_update(2).before(sobel.get_update(2), computation::root);

    tiramisu::buffer buff_input("buff_input", {ROWS_PER_NODE + 2, COLS}, p_float32, a_input, &sobel_dist);
    tiramisu::buffer buff_sobel_x("buff_sobel_x", {ROWS_PER_NODE, COLS - 2}, p_float32, a_temporary, &sobel_dist);
    tiramisu::buffer buff_sobel_x_last_node("buff_sobel_x_last_node", {ROWS_PER_NODE - 2, COLS - 2}, p_float32, a_temporary, &sobel_dist);
    tiramisu::buffer buff_sobel_y("buff_sobel_y", {ROWS_PER_NODE, COLS - 2}, p_float32, a_temporary, &sobel_dist);
    tiramisu::buffer buff_sobel_y_last_node("buff_sobel_y_last_node", {ROWS_PER_NODE - 2, COLS - 2}, p_float32, a_temporary, &sobel_dist);
    tiramisu::buffer buff_sobel("buff_sobel", {ROWS_PER_NODE, COLS - 2}, p_float32, a_output, &sobel_dist);
    tiramisu::buffer buff_sobel_last_node("buff_sobel_last_node", {ROWS_PER_NODE - 2, COLS - 2}, p_float32, a_output, &sobel_dist);

    input.set_access("{input[i,j]->buff_input[i,j]}");
    sobel_x.get_update(0).set_access("{sobel_x_0[i,j]->buff_sobel_x[i,j]}");
    sobel_x.get_update(1).set_access("{sobel_x_1[i,j]->buff_sobel_x[i,j]}");
    sobel_x.get_update(2).set_access("{sobel_x_2[i,j]->buff_sobel_x_last_node[i,j]}");
    sobel_y.get_update(0).set_access("{sobel_y_0[i,j]->buff_sobel_y[i,j]}");
    sobel_y.get_update(1).set_access("{sobel_y_1[i,j]->buff_sobel_y[i,j]}");
    sobel_y.get_update(2).set_access("{sobel_y_2[i,j]->buff_sobel_y_last_node[i,j]}");
    sobel.get_update(0).set_access("{sobel_0[i,j]->buff_sobel[i,j]}");
    sobel.get_update(1).set_access("{sobel_1[i,j]->buff_sobel[i,j]}");
    sobel.get_update(2).set_access("{sobel_2[i,j]->buff_sobel_last_node[i,j]}");

    exchange.r->set_access("{exchange_r[q,i,j]->buff_input[" + std::to_string(ROWS_PER_NODE) + "+ i,j]}");
    exchange_last_node.r->set_access("{exchange_last_node_r[q,i,j]->buff_input[" + std::to_string(ROWS_PER_NODE) + "+ i,j]}");

    // Add schedules.

    sobel_dist.set_arguments({&buff_input, /*&buff_sobel_x, &buff_sobel_x_last_node, &buff_sobel_y,
					     &buff_sobel_y_last_node,*/ &buff_sobel, &buff_sobel_last_node});
    sobel_dist.gen_time_space_domain();
    sobel_dist.lift_ops_to_library_calls();
    sobel_dist.gen_isl_ast();
    sobel_dist.gen_halide_stmt();
    sobel_dist.gen_halide_obj("build/generated_sobel_dist.o");
    sobel_dist.dump_halide_stmt();

    return 0;
}
