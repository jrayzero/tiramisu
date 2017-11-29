//
// Created by Jessica Ray on 11/2/17.
//

// This distribution assumes that the data resides across multiple nodes, so communication is only needed to transfer
// over the ghost zones.

// This splits data evenly

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

#include "blur_params.h"

using namespace tiramisu;

int main() {
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(T_LOOP_ITER_TYPE);

    function blur_dist("blur_dist");

    C_LOOP_ITER_TYPE rows = ROWS;
    C_LOOP_ITER_TYPE cols = COLS;
    C_LOOP_ITER_TYPE nodes = NODES;
    C_LOOP_ITER_TYPE rows_per_node = rows / nodes;

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    var y("y"), x("x");
    constant rows_const("rows", expr(rows), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant cols_const("cols", expr(cols), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    computation blur_input("[rows, cols]->{blur_input[i1, i0]: 0<=i1<rows and 0<=i0<cols}", expr(), false,
                           T_DATA_TYPE, &blur_dist);

    computation bx("[rows, cols]->{bx[y, x]: 0<=y<rows and 0<=x<cols-2}",
                   (((blur_input(y, x) + blur_input(y, (x + expr((C_LOOP_ITER_TYPE)1)))) +
                     blur_input(y, (x + expr((C_LOOP_ITER_TYPE)2)))) / expr((C_DATA_TYPE)3)),
                   true, T_DATA_TYPE, &blur_dist);

    computation by("[rows, cols]->{by[y, x]: 0<=y<rows-2 and 0<=x<cols-2}",
                   (((bx(y, x) + bx((y + expr((C_LOOP_ITER_TYPE)1)), x)) +
                           bx((y + expr((C_LOOP_ITER_TYPE)2)), x)) / expr((C_DATA_TYPE)3)),
                   true, T_DATA_TYPE, &blur_dist);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    constant nodes_const("nodes", expr(nodes), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    /*
     * Prep for distribution by splitting the outer dimension
     */

    var y1("y1"), y2("y2"), q("q");

    bx.split(y, rows_per_node, y1, y2);
    by.split(y, rows_per_node, y1, y2);

    // separate off the first and last nodes because they do different communication that the other nodes
    // The first and last nodes correspond to the first and last chunks of rows, respectively
    bx.separate_at(y1, {(C_LOOP_ITER_TYPE)1, nodes-(C_LOOP_ITER_TYPE)1}, nodes);
    by.separate_at(y1, {(C_LOOP_ITER_TYPE)1, nodes-(C_LOOP_ITER_TYPE)1}, nodes);

    generator::update_producer_expr_name(&(by.get_update(0)), "bx", "bx_0");
    generator::update_producer_expr_name(&(by.get_update(1)), "bx", "bx_1");
    generator::update_producer_expr_name(&(by.get_update(2)), "bx", "bx_2");

    /*
     * Create communication
     */

    communication_prop sync_block("sync_block", p_uint64, {FIFO, SYNC, BLOCK, MPI});
    communication_prop async_block("async_block", p_uint64, {FIFO, ASYNC, BLOCK, MPI});

    // transfer the computed rows from bx
    send_recv bx_exchange = computation::create_transfer("[nodes, cols]->{bx_exchange_s[q,y,x]: 1<=q<nodes-1 and 0<=y<2 and 0<=x<cols-2 and nodes>1}",
                                                         "[nodes, cols]->{bx_exchange_r[q,y,x]: 0<=q<nodes-2 and 0<=y<2 and 0<=x<cols-2 and nodes>1}",
                                                         q, q-(C_LOOP_ITER_TYPE)1, q+(C_LOOP_ITER_TYPE)1, q, async_block, sync_block,
                                                         bx.get_update(1)(y,x), &blur_dist);

    send_recv bx_exchange_last_node = computation::create_transfer("[nodes, cols]->{bx_exchange_last_node_s[q,y,x]: nodes-1<=q<nodes and 0<=y<2 and 0<=x<cols-2 and nodes>1}",
                                                                   "[cols, nodes]->{bx_exchange_last_node_r[q,y,x]: nodes-2<=q<nodes-1 and 0<=y<2 and 0<=x<cols-2 and nodes>1}",
                                                                   q, q-(C_LOOP_ITER_TYPE)1, q+(C_LOOP_ITER_TYPE)1, q, async_block, sync_block,
                                                                   bx.get_update(2)(y,x), &blur_dist);

    /*
     * Ordering
     */

    bx.get_update(0).before(bx.get_update(1), computation::root);
    bx.get_update(1).before(bx.get_update(2), computation::root);
    bx.get_update(2).before(*bx_exchange.s, computation::root);
    bx_exchange.s->before(*bx_exchange_last_node.s, computation::root);
    bx_exchange_last_node.s->before(*bx_exchange.r, computation::root);
    bx_exchange.r->before(*bx_exchange_last_node.r, computation::root);
    bx_exchange_last_node.r->before(by.get_update(0), computation::root);
    by.get_update(0).before(by.get_update(1), computation::root);
    by.get_update(1).before(by.get_update(2), computation::root);

    /*
     * Tag distribute level
     */

    bx.get_update(0).tag_distribute_level(y1);
    bx.get_update(1).tag_distribute_level(y1);
    bx.get_update(2).tag_distribute_level(y1);

    by.get_update(0).tag_distribute_level(y1);
    by.get_update(1).tag_distribute_level(y1);
    by.get_update(2).tag_distribute_level(y1);

    bx_exchange.s->tag_distribute_level(q, false);
    bx_exchange.r->tag_distribute_level(q, false);
    bx_exchange_last_node.s->tag_distribute_level(q, false);
    bx_exchange_last_node.r->tag_distribute_level(q, false);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    /*
     * Collapsing
     */

    bx_exchange.s->collapse_many({collapser(2, 0, (C_LOOP_ITER_TYPE)cols), collapser(1, 0, (C_LOOP_ITER_TYPE)2)});
    bx_exchange.r->collapse_many({collapser(2, 0, (C_LOOP_ITER_TYPE)cols), collapser(1, 0, (C_LOOP_ITER_TYPE)2)});
    bx_exchange_last_node.s->collapse_many({collapser(2, 0, (C_LOOP_ITER_TYPE)cols), collapser(1, 0, (C_LOOP_ITER_TYPE)2)});
    bx_exchange_last_node.r->collapse_many({collapser(2, 0, (C_LOOP_ITER_TYPE)cols), collapser(1, 0, (C_LOOP_ITER_TYPE)2)});

    /*
     * Buffers
     */

    tiramisu::expr bx_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == nodes-1, tiramisu::expr(rows_per_node+2), tiramisu::expr(rows_per_node));
    tiramisu::expr by_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == nodes-1, tiramisu::expr(rows_per_node+2), tiramisu::expr(rows_per_node));

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(rows_per_node), tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);

    tiramisu::buffer buff_bx("buff_bx", {bx_select_dim0, tiramisu::expr(cols - 2)},
                             T_DATA_TYPE, tiramisu::a_temporary, &blur_dist);

    tiramisu::buffer buff_by("buff_by", {by_select_dim0, tiramisu::expr(cols - 2)},
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");
    bx.get_update(0).set_access("{bx_0[y, x]->buff_bx[y, x]}");
    bx.get_update(1).set_access("{bx_1[y, x]->buff_bx[y, x]}");
    bx.get_update(2).set_access("{bx_2[y, x]->buff_bx[y, x]}");
    by.get_update(0).set_access("{by_0[y, x]->buff_by[y, x]}");
    by.get_update(1).set_access("{by_1[y, x]->buff_by[y, x]}");
    by.get_update(2).set_access("{by_2[y, x]->buff_by[y, x]}");

    bx_exchange.r->set_access("{bx_exchange_r[q,y,x]->buff_bx[" + std::to_string(rows_per_node) + " + y, x]}");
    bx_exchange_last_node.r->set_access("{bx_exchange_last_node_r[q,y,x]->buff_bx[" + std::to_string(rows_per_node) + " + y, x]}");

    blur_dist.set_arguments({&buff_input, &buff_by});

    // Generate code
    blur_dist.gen_time_space_domain();
    blur_dist.lift_ops_to_library_calls();
    blur_dist.gen_isl_ast();
    blur_dist.gen_halide_stmt();
    blur_dist.gen_halide_obj("./build/generated_blur_dist.o");

    // Some debugging
    blur_dist.dump_halide_stmt();

    return 0;

}
