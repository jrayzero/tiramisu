//
// Created by Jessica Ray on 11/2/17.
//

// This distribution assumes that the data resides across multiple nodes, so communication is only needed to transfer
// over the ghost zones

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

int main() {

    global::set_default_tiramisu_options();

    function dblurxy_dist_data("dblurxy_dist_data");

    int _rows = _ROWS;
    int _cols = _COLS;
    int _nodes = _NODES;

    int _rows_per_node = _rows / _nodes;
    std::cerr << "rows per node: " << _rows_per_node << std::endl;

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    var y("y"), x("x");
    constant rows("rows", expr(_rows), p_int32, true, NULL, 0, &dblurxy_dist_data);
    constant cols("cols", expr(_cols), p_int32, true, NULL, 0, &dblurxy_dist_data);

    computation
            blur_input("[rows, cols]->{blur_input[i1, i0]: 0<=i1<rows and 0<=i0<cols}", expr(), false,
                       p_uint64, &dblurxy_dist_data);

    computation bx("[rows, cols]->{bx[y, x]: 0<=y<rows and 0<=x<cols-2}",
                   (((blur_input(y, x) +
                      blur_input(y, (x + expr((int32_t)1)))) +
                     blur_input(y, (x + expr((int32_t)2)))) / expr((uint64_t)3)),
                   true, p_uint64, &dblurxy_dist_data);

    computation by("[rows, cols]->{by[y, x]: 0<=y<rows-2 and 0<=x<cols-2}",
                   (((bx(y, x) +
                      bx((y + expr((int32_t)1)), x)) +
                     bx((y + expr((int32_t)2)), x)) / expr((uint64_t)3)),
                   true, p_uint64, &dblurxy_dist_data);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    constant one("one", expr(1), p_int32, true, NULL, 0, &dblurxy_dist_data);
    constant two("two", expr(2), p_int32, true, NULL, 0, &dblurxy_dist_data);
    constant nodes("nodes", expr(_nodes), p_int32, true, NULL, 0, &dblurxy_dist_data);
    constant nodes_minus_one("nodes_minus_one", expr(_nodes-1), p_int32, true, NULL, 0, &dblurxy_dist_data);
    constant nodes_minus_two("nodes_minus_two", expr(_nodes-2), p_int32, true, NULL, 0, &dblurxy_dist_data);
    constant rows_per_node("rows_per_node", expr(_rows_per_node), p_int32, true, NULL, 0, &dblurxy_dist_data);

    /*
     * Prep for distribution by splitting the outer dimension
     */

    var y1("y1"), y2("y2"), d("d"), q("q");

    bx.split(y, _rows_per_node, y1, y2);
    by.split(y, _rows_per_node, y1, y2);

    // separate off the first and last nodes because they do different communication that the other nodes
    // The first and last nodes correspond to the first and last chunks of rows, respectively

    bx.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    bx.get_update(0).rename_computation("bx_0");
    bx.get_update(1).rename_computation("bx_1");
    bx.get_update(2).rename_computation("bx_2");

    by.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    by.get_update(0).rename_computation("by_0");
    by.get_update(1).rename_computation("by_1");
    by.get_update(2).rename_computation("by_2");

    generator::replace_expr_name(by.get_update(0).expression, "bx", "bx_0");
    generator::replace_expr_name(by.get_update(1).expression, "bx", "bx_1");
    generator::replace_expr_name(by.get_update(2).expression, "bx", "bx_2");

    /*
     * Create communication
     */

    channel sync_block("sync_block", p_uint64, {FIFO, SYNC, BLOCK, MPI});
    channel async_nonblock("async_nonblock", p_uint64, {FIFO, ASYNC, NONBLOCK, MPI});
    channel async_block("async_block", p_uint64, {FIFO, ASYNC, BLOCK, MPI});

    // transfer the top row to the node above you. That becomes the upper node's bottom row
    /*    send_recv upper_exchanges = computation::create_transfer("[one, nodes, cols]->{upper_s[q,y,x]: one<=q<nodes and 0<=y<2 and 0<=x<cols}",
                                                             "[nodes_minus_one, cols]->{upper_r[q,y,x]: 0<=q<nodes_minus_one and 0<=y<2 and 0<=x<cols}",
                                                             q, q-1, q+1, q, async_nonblock, sync_block,
                                                             blur_input(y,x), &dblurxy_dist_data);
    */
    // transfer the computed rows from bx
    send_recv bx_exchange = computation::create_transfer("[one, nodes_minus_one, cols]->{bx_exchange_s[q,y,x]: one<=q<nodes_minus_one and 0<=y<2 and 0<=x<cols-2}",
							 "[nodes_minus_two, cols]->{bx_exchange_r[q,y,x]: 0<=q<nodes_minus_two and 0<=y<2 and 0<=x<cols-2}",
							 q, q-1, q+1, q, async_block, sync_block,
							 bx.get_update(1)(y,x), &dblurxy_dist_data);

    send_recv bx_exchange_last_node = computation::create_transfer("[one, nodes_minus_one, nodes_minus_two, nodes, cols]->{bx_exchange_last_node_s[q,y,x]: nodes_minus_one<=q<nodes and 0<=y<2 and 0<=x<cols-2}",
							 "[nodes_minus_one, nodes_minus_two, cols]->{bx_exchange_last_node_r[q,y,x]: nodes_minus_two<=q<nodes_minus_one and 0<=y<2 and 0<=x<cols-2}",
							 q, q-1, q+1, q, async_block, sync_block,
							 bx.get_update(2)(y,x), &dblurxy_dist_data);


    tiramisu::wait bx_exchange_wait(bx_exchange.s->operator()(q, y, x), &dblurxy_dist_data);
    tiramisu::wait bx_exchange_last_node_wait(bx_exchange_last_node.s->operator()(q, y, x), &dblurxy_dist_data);
    bx_exchange_wait.set_schedule_this_comp(false);
    bx_exchange_last_node_wait.set_schedule_this_comp(false);

    /*
     * Ordering
     */

    /*        upper_exchanges.s->before(*upper_exchanges.r, computation::root);
    upper_exchanges.r->before(bx.get_update(0), computation::root);
    bx.get_update(0).before(*bx_exchange.s, computation::root);    
    bx_exchange.s->before(by.get_update(0), computation::root);
    by.get_update(0).before(*bx_exchange.r, computation::root);
    bx_exchange.r->before(bx.get_update(1), computation::root);
    bx.get_update(1).before(by.get_update(1), computation::root);
    by.get_update(1).before(bx.get_update(2), computation::root);
    bx.get_update(2).before(by.get_update(2), computation::root);*/

    bx.get_update(0).before(bx.get_update(1), computation::root);    
    bx.get_update(1).before(bx.get_update(2), computation::root);
    bx.get_update(2).before(*bx_exchange.s, computation::root);
    bx_exchange.s->before(*bx_exchange_last_node.s, computation::root);
    bx_exchange_last_node.s->before(*bx_exchange.r, computation::root);
    bx_exchange.r->before(*bx_exchange_last_node.r, computation::root);
    bx_exchange_last_node.r->before(by.get_update(0), computation::root);
    by.get_update(0).before(by.get_update(1), computation::root);
    by.get_update(1).before(by.get_update(2), computation::root);
    by.get_update(2).before(bx_exchange_wait, computation::root);
    bx_exchange_wait.before(bx_exchange_last_node_wait, computation::root);

    /*
     * Tag distribute level
     */

    bx.get_update(0).tag_distribute_level(y1);
    bx.get_update(0).drop_rank_iter();
    bx.get_update(1).tag_distribute_level(y1);
    bx.get_update(2).tag_distribute_level(y1);
    bx.get_update(1).drop_rank_iter();
    bx.get_update(2).drop_rank_iter();

    by.get_update(0).tag_distribute_level(y1);
    by.get_update(0).drop_rank_iter();
    by.get_update(1).tag_distribute_level(y1);
    by.get_update(2).tag_distribute_level(y1);
    by.get_update(1).drop_rank_iter();
    by.get_update(2).drop_rank_iter();

    bx_exchange.s->tag_distribute_level(q);
    bx_exchange.r->tag_distribute_level(q);

    bx_exchange_last_node.s->tag_distribute_level(q);
    bx_exchange_last_node.r->tag_distribute_level(q);

    bx_exchange_wait.tag_distribute_level(q);
    bx_exchange_last_node_wait.tag_distribute_level(q);

    // Do some other scheduling to match the Halide stuff
    /*    var y3("y3"), y4("y4");
    bx.get_update(0).split(y2, 8, y3, y4);
    bx.get_update(1).split(y2, 8, y3, y4);
    bx.get_update(2).split(y2, 8, y3, y4);
    bx.get_update(0).tag_parallel_level(y3);
    bx.get_update(1).tag_parallel_level(y3);
    bx.get_update(2).tag_parallel_level(y3);
    //    bx.get_update(0).set_loop_level_names({3}, {"x"});
    //    bx.get_update(1).set_loop_level_names({3}, {"x"});
//    bx.get_update(2).set_loop_level_names({3}, {"x"});
    //    bx.get_update(0).tag_vector_level(x, 8);
    //    bx.get_update(1).tag_vector_level(x, 8);
    //    bx.get_update(2).tag_vector_level(x, 8);

    by.get_update(0).split(y2, 8, y3, y4);
    by.get_update(1).split(y2, 8, y3, y4);
    by.get_update(2).split(y2, 8, y3, y4);
    by.get_update(0).tag_parallel_level(y3);
    by.get_update(1).tag_parallel_level(y3);
    by.get_update(2).tag_parallel_level(y3);
    //    by.get_update(0).set_loop_level_names({3}, {"x"});
    //    by.get_update(1).set_loop_level_names({3}, {"x"});
    //    by.get_update(2).set_loop_level_names({3}, {"x"});
    //    by.get_update(0).tag_vector_level(x, 8);
    //    by.get_update(1).tag_vector_level(x, 8);
    //    by.get_update(2).tag_vector_level(x, 8);
    */
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    /*
     * Collapsing
     */

    bx_exchange_wait.collapse_many({collapser(2, 0, _cols), collapser(1, 0, 2)});
    bx_exchange_last_node_wait.collapse_many({collapser(2, 0, _cols), collapser(1, 0, 2)});
    bx_exchange.s->collapse_many({collapser(2, 0, _cols), collapser(1, 0, 2)});
    bx_exchange.r->collapse_many({collapser(2, 0, _cols), collapser(1, 0, 2)});
    bx_exchange_last_node.s->collapse_many({collapser(2, 0, _cols), collapser(1, 0, 2)});
    bx_exchange_last_node.r->collapse_many({collapser(2, 0, _cols), collapser(1, 0, 2)});

    /*
     * Buffers
     */

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(_rows_per_node), tiramisu::expr(_cols)}, p_uint64,
                                tiramisu::a_input, &dblurxy_dist_data);

    tiramisu::buffer buff_bx("buff_bx", {tiramisu::expr(_rows_per_node+2), tiramisu::expr(_cols - 2)},
                             p_uint64, tiramisu::a_output, &dblurxy_dist_data);

    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(_rows_per_node), tiramisu::expr(_cols - 2)},
                             p_uint64, tiramisu::a_output, &dblurxy_dist_data);

    tiramisu::buffer buff_bx_last("buff_bx_last", {tiramisu::expr(_rows_per_node), tiramisu::expr(_cols - 2)},
                                  p_uint64, tiramisu::a_output, &dblurxy_dist_data);

    tiramisu::buffer buff_by_last("buff_by_last", {tiramisu::expr(_rows_per_node - 2), tiramisu::expr(_cols - 2)},
                                  p_uint64, tiramisu::a_output, &dblurxy_dist_data);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");
    bx.get_update(0).set_access("{bx_0[y, x]->buff_bx[y, x]}");
    bx.get_update(1).set_access("{bx_1[y, x]->buff_bx[y, x]}");
    bx.get_update(2).set_access("{bx_2[y, x]->buff_bx_last[y, x]}");
    by.get_update(0).set_access("{by_0[y, x]->buff_by[y, x]}");
    by.get_update(1).set_access("{by_1[y, x]->buff_by[y, x]}");
    by.get_update(2).set_access("{by_2[y, x]->buff_by_last[y, x]}");

    bx_exchange.r->set_access("{bx_exchange_r[q,y,x]->buff_bx[" + std::to_string(_rows_per_node) + " + y, x]}");
    bx_exchange_last_node.r->set_access("{bx_exchange_last_node_r[q,y,x]->buff_bx[" + std::to_string(_rows_per_node) + " + y, x]}");

    buffer bx_exchange_wait_buff("bx_exchange_wait_buff", {1}, tiramisu::p_req_ptr, a_temporary, &dblurxy_dist_data);
    bx_exchange.s->set_req_access("{bx_exchange_s[q,y,x]->bx_exchange_wait_buff[0]}");
    bx_exchange_last_node.s->set_req_access("{bx_exchange_last_node_s[q,y,x]->bx_exchange_wait_buff[0]}");

    dblurxy_dist_data.set_arguments({&buff_input, &buff_bx, &buff_bx_last, &buff_by, &buff_by_last});
    // Generate code
    dblurxy_dist_data.gen_time_space_domain();
    dblurxy_dist_data.lift_ops_to_library_calls();
    dblurxy_dist_data.gen_isl_ast();
    dblurxy_dist_data.gen_halide_stmt();
    dblurxy_dist_data.gen_halide_obj("./build/generated_dblurxy_dist_data.o");

    // Some debugging
    dblurxy_dist_data.dump_halide_stmt();

    return 0;

}
