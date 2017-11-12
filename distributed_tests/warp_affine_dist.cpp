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

    int SIZE0 = _COLS;
    int SIZE1 = _ROWS;
    int rows_per_node = _ROWS / NODES;
    tiramisu::function affine_dist("warp_affine_dist");

    // Input params.
    float a00__0 = 0.1;
    float a01__0 = 0.1;
    float a10__0 = 0.1;
    float a11__0 = 0.1;
    float b00__0 = 0.1;
    float b10__0 = 0.1;

    // Output buffers.
    int affine_extent_1 = SIZE1;
    int affine_extent_0 = SIZE0;

    // Input buffers.
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::computation input("[input_extent_1, input_extent_0]->{input[i1, i0]: (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint64, &affine_dist);

    // Define loop bounds for dimension "affine_s0_y".
    tiramisu::constant affine_s0_y_loop_min("affine_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &affine_dist);
    tiramisu::constant affine_s0_y_loop_extent("affine_s0_y_loop_extent", tiramisu::expr((int32_t)affine_extent_1), tiramisu::p_int32, true, NULL, 0, &affine_dist);

    // Define loop bounds for dimension "affine_s0_x".
    tiramisu::constant affine_s0_x_loop_min("affine_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &affine_dist);
    tiramisu::constant affine_s0_x_loop_extent("affine_s0_x_loop_extent", tiramisu::expr((int32_t)affine_extent_0), tiramisu::p_int32, true, NULL, 0, &affine_dist);
    tiramisu::computation affine_s0(
        "[affine_s0_y_loop_min, affine_s0_y_loop_extent, affine_s0_x_loop_min, affine_s0_x_loop_extent]->{affine_s0[affine_s0_y, affine_s0_x]: "
        "(affine_s0_y_loop_min <= affine_s0_y <= ((affine_s0_y_loop_min + affine_s0_y_loop_extent) + -1)) and (affine_s0_x_loop_min <= affine_s0_x <= ((affine_s0_x_loop_min + affine_s0_x_loop_extent) + -1))}",
        tiramisu::expr(), true, tiramisu::p_float32, &affine_dist);
    tiramisu::constant t57("t57", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t58("t58", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t59("t59", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t60("t60", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t61("t61", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t62("t62", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t63("t63", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_0)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    tiramisu::constant t64("t64", tiramisu::expr(tiramisu::o_max, tiramisu::expr(tiramisu::o_min, (tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))) + tiramisu::expr((int32_t)1)), tiramisu::expr(input_extent_1)), tiramisu::expr((int32_t)0)), tiramisu::p_int32, false, &affine_s0, 1, &affine_dist);
    affine_s0.set_expression(((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t57, t58)) * (tiramisu::expr((float)1) - ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t59, t60)) * ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) * (tiramisu::expr((float)1) - ((((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))))) + (((tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t61, t62)) * (tiramisu::expr((float)1) - ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, input(t63, t64)) * ((((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a11__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a10__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b00__0)))))) * ((((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)) - tiramisu::expr(o_floor, (((tiramisu::expr(a01__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_y"))) + (tiramisu::expr(a00__0) * tiramisu::expr(tiramisu::o_cast, tiramisu::p_float32, tiramisu::var("affine_s0_x")))) + tiramisu::expr(b10__0)))))));
    
    // Define compute level for "affine".

    // Declare vars.
    tiramisu::var affine_s0_x("affine_s0_x");
    tiramisu::var affine_s0_y("affine_s0_y");

    var y1("y1"), y2("y2"), q("q"), c("c"), y("y"), x("x");
    
    //    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    //    tiramisu::constant three("three", tiramisu::expr(3), tiramisu::p_int32, true, NULL, 0, &gaussian_tiramisu);
    //    constant nodes("nodes", expr(NODES), p_int32, true, NULL, 0, &gaussian_tiramisu);
    //    constant nodes_minus_one("nodes_minus_one", expr(NODES-1), p_int32, true, NULL, 0, &gaussian_tiramisu);
    //    constant nodes_minus_two("nodes_minus_two", expr(NODES-2), p_int32, true, NULL, 0, &gaussian_tiramisu);

    affine_s0.split(affine_s0_y, rows_per_node, y1, y2);
    t57.split(affine_s0_y, rows_per_node, y1, y2);
    t58.split(affine_s0_y, rows_per_node, y1, y2);
    t59.split(affine_s0_y, rows_per_node, y1, y2);
    t60.split(affine_s0_y, rows_per_node, y1, y2);
    t61.split(affine_s0_y, rows_per_node, y1, y2);
    t62.split(affine_s0_y, rows_per_node, y1, y2);
    t63.split(affine_s0_y, rows_per_node, y1, y2);
    t64.split(affine_s0_y, rows_per_node, y1, y2);

    /*    affine_s0.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    affine_s0.get_update(0).rename_computation("affine_s0_0");
    affine_s0.get_update(1).rename_computation("affine_s0_1");
    affine_s0.get_update(2).rename_computation("affine_s0_2");

    t57.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t57.get_update(0).rename_computation("t57_0");
    t57.get_update(1).rename_computation("t57_1");
    t57.get_update(2).rename_computation("t57_2");

    t58.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t58.get_update(0).rename_computation("t58_0");
    t58.get_update(1).rename_computation("t58_1");
    t58.get_update(2).rename_computation("t58_2");

    t59.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t59.get_update(0).rename_computation("t59_0");
    t59.get_update(1).rename_computation("t59_1");
    t59.get_update(2).rename_computation("t59_2");

    t60.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t60.get_update(0).rename_computation("t60_0");
    t60.get_update(1).rename_computation("t60_1");
    t60.get_update(2).rename_computation("t60_2");

    t61.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t61.get_update(0).rename_computation("t61_0");
    t61.get_update(1).rename_computation("t61_1");
    t61.get_update(2).rename_computation("t61_2");

    t62.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t62.get_update(0).rename_computation("t62_0");
    t62.get_update(1).rename_computation("t62_1");
    t62.get_update(2).rename_computation("t62_2");

    t63.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t63.get_update(0).rename_computation("t63_0");
    t63.get_update(1).rename_computation("t63_1");
    t63.get_update(2).rename_computation("t63_2");

    t64.separate_at(0, {one, nodes_minus_one}, nodes, -3);
    t64.get_update(0).rename_computation("t64_0");
    t64.get_update(1).rename_computation("t64_1");
    t64.get_update(2).rename_computation("t64_2");
    */
    /*    channel async_nonblock("async_nonblock", p_float32, {FIFO, ASYNC, NONBLOCK, MPI});
    channel async_block("async_block", p_float32, {FIFO, ASYNC, BLOCK, MPI});

    // transfer the computed rows from gaussian_x    
    send_recv exchange = computation::create_transfer("[one, nodes_minus_one, cols, channels]->{gaussian_exchange_s[q,c,y,x]: one<=q<nodes_minus_one and 0<=y<4 and 0<=x<cols-4 and 0<=c<channels}",
							 "[nodes_minus_two, cols, channels]->{gaussian_exchange_r[q,c,y,x]: 0<=c<channels and 0<=q<nodes_minus_two and 0<=y<4 and 0<=x<cols-4}",
								 q, q-1, q+1, q, async_block, sync_block,
								 gaussian_x_s0.get_update(1)(c,y,x), &gaussian_tiramisu);
    
    send_recv exchange_last_node = computation::create_transfer("[channels, one, nodes_minus_one, nodes, cols]->{gaussian_exchange_last_node_s[q,c,y,x]: nodes_minus_one<=q<nodes and 0<=y<4 and 0<=x<cols-4 and 0<=c<channels}",
									  "[nodes_minus_one, nodes_minus_two, cols, channels]->{gaussian_exchange_last_node_r[q,c,y,x]: nodes_minus_two<=q<nodes_minus_one and 0<=y<4 and 0<=x<cols-4 and 0<=c<channels}",
									  q, q-1, q+1, q, async_block, sync_block,
									  gaussian_x_s0.get_update(2)(c,y,x), &gaussian_tiramisu);*/


    affine_s0.get_update(0).tag_distribute_level(y1);
    affine_s0.get_update(0).drop_rank_iter();
    //    affine_s0.get_update(1).tag_distribute_level(y1);
    //    affine_s0.get_update(1).drop_rank_iter();
    //    affine_s0.get_update(2).tag_distribute_level(y1);
    //    affine_s0.get_update(2).drop_rank_iter();
    t57.get_update(0).tag_distribute_level(y1);
    t57.get_update(0).drop_rank_iter();
    //    t57.get_update(1).tag_distribute_level(y1);
    //    t57.get_update(1).drop_rank_iter();
    //    t57.get_update(2).tag_distribute_level(y1);
    //    t57.get_update(2).drop_rank_iter();
    t58.get_update(0).tag_distribute_level(y1);
    t58.get_update(0).drop_rank_iter();
    //    t58.get_update(1).tag_distribute_level(y1);
    //    t58.get_update(1).drop_rank_iter();
//    t58.get_update(2).tag_distribute_level(y1);
    //    t58.get_update(2).drop_rank_iter();
    t59.get_update(0).tag_distribute_level(y1);
    t59.get_update(0).drop_rank_iter();
    //    t59.get_update(1).tag_distribute_level(y1);
    //    t59.get_update(1).drop_rank_iter();
    //    t59.get_update(2).tag_distribute_level(y1);
    //    t59.get_update(2).drop_rank_iter();
    t60.get_update(0).tag_distribute_level(y1);
    t60.get_update(0).drop_rank_iter();
    //    t60.get_update(1).tag_distribute_level(y1);
    //    t60.get_update(1).drop_rank_iter();
    //    t60.get_update(2).tag_distribute_level(y1);
    //    t60.get_update(2).drop_rank_iter();
    t61.get_update(0).tag_distribute_level(y1);
    t61.get_update(0).drop_rank_iter();
    //    t61.get_update(1).tag_distribute_level(y1);
    //    t61.get_update(1).drop_rank_iter();
    //    t61.get_update(2).tag_distribute_level(y1);
    //    t61.get_update(2).drop_rank_iter();
    t62.get_update(0).tag_distribute_level(y1);
    t62.get_update(0).drop_rank_iter();
    //    t62.get_update(1).tag_distribute_level(y1);
    //    t62.get_update(1).drop_rank_iter();
    //    t62.get_update(2).tag_distribute_level(y1);
    //    t62.get_update(2).drop_rank_iter();
    t63.get_update(0).tag_distribute_level(y1);
    t63.get_update(0).drop_rank_iter();
    //    t63.get_update(1).tag_distribute_level(y1);
    //    t63.get_update(1).drop_rank_iter();
    //    t63.get_update(2).tag_distribute_level(y1);
    //    t63.get_update(2).drop_rank_iter();
    t64.get_update(0).tag_distribute_level(y1);
    t64.get_update(0).drop_rank_iter();
    //    t64.get_update(1).tag_distribute_level(y1);
    //    t64.get_update(1).drop_rank_iter();
    //    t64.get_update(2).tag_distribute_level(y1);
    //    t64.get_update(2).drop_rank_iter();
    
    //    exchange.s->before(*exchange_last_node.s, computation::root);
    //    exchange_last_node.s->before(*exchange.r, computation::root);
    //    exchange.r->before(*exchange_last_node.r, computation::root);
    //    exchange_last_node.r->before(t57.get_update(0), computation::root);
    t58.get_update(0).after(t57.get_update(0), affine_s0_x);
    t59.get_update(0).after(t58.get_update(0), affine_s0_x);
    t60.get_update(0).after(t59.get_update(0), affine_s0_x);
    t61.get_update(0).after(t60.get_update(0), affine_s0_x);
    t62.get_update(0).after(t61.get_update(0), affine_s0_x);
    t63.get_update(0).after(t62.get_update(0), affine_s0_x);
    t64.get_update(0).after(t63.get_update(0), affine_s0_x);
    affine_s0.get_update(0).after(t64.get_update(0), affine_s0_x);
    //    t57.get_update(1).after(affine_s0.get_update(0), affine_s0_x);
    //    t58.get_update(1).after(t57.get_update(1), affine_s0_x);
    //    t59.get_update(1).after(t58.get_update(1), affine_s0_x);
    //    t60.get_update(1).after(t59.get_update(1), affine_s0_x);
    //    t61.get_update(1).after(t60.get_update(1), affine_s0_x);
    //    t62.get_update(1).after(t61.get_update(1), affine_s0_x);
    //    t63.get_update(1).after(t62.get_update(1), affine_s0_x);
    //    t64.get_update(1).after(t63.get_update(1), affine_s0_x);
    //    affine_s0.get_update(1).after(t64.get_update(1), affine_s0_x);
    //    t57.get_update(2).after(affine_s0.get_update(1), affine_s0_x);
    //    t58.get_update(2).after(t57.get_update(2), affine_s0_x);
    //    t59.get_update(2).after(t58.get_update(2), affine_s0_x);
    //    t60.get_update(2).after(t59.get_update(2), affine_s0_x);
    //    t61.get_update(2).after(t60.get_update(2), affine_s0_x);
    //    t62.get_update(2).after(t61.get_update(2), affine_s0_x);
    //    t63.get_update(2).after(t62.get_update(2), affine_s0_x);
    //    t64.get_update(2).after(t63.get_update(2), affine_s0_x);
    //    affine_s0.get_update(2).after(t64.get_update(2), affine_s0_x);    

    tiramisu::buffer buff_affine("buff_affine", {tiramisu::expr(affine_extent_1 / 2), tiramisu::expr(affine_extent_0)}, tiramisu::p_float32, tiramisu::a_output, &affine_dist);
    //    tiramisu::buffer buff_affine_last_node("buff_affine_last_node", {tiramisu::expr(affine_extent_1) - 1, tiramisu::expr(affine_extent_0) - 1}, tiramisu::p_float32, tiramisu::a_output, &affine_dist);
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_1 / 2), tiramisu::expr(input_extent_0)}, tiramisu::p_uint64, tiramisu::a_input, &affine_dist);
    //    tiramisu::buffer buff_input_last_node("buff_input_last_node", {tiramisu::expr(input_extent_1) + 1, tiramisu::expr(input_extent_0)}, tiramisu::p_uint64, tiramisu::a_input, &affine_dist);
    input.set_access("{input[i1, i0]->buff_input[i1, i0]}");
    affine_s0.set_access("{affine_s0[affine_s0_y, affine_s0_x]->buff_affine[affine_s0_y, affine_s0_x]}");

    affine_dist.set_arguments({&buff_input, &buff_affine});
    affine_dist.gen_time_space_domain();
    affine_dist.gen_isl_ast();
    affine_dist.gen_halide_stmt();
    affine_dist.dump_halide_stmt();
    affine_dist.gen_halide_obj("build/generated_warp_affine_dist.o");

    return 0;
}
