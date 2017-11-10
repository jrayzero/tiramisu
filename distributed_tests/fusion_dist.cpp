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

    tiramisu::function fusion_dist("fusion_dist");

    int COLS = _COLS;
    int ROWS = _ROWS;
    int CHANNELS = _CHANNELS;
    int rows_per_node = ROWS / NODES;

    // Output buffers.
    int f_extent_2 = CHANNELS;
    int f_extent_1 = ROWS;
    int f_extent_0 = COLS;

    // Input buffers.
    int input_extent_2 = CHANNELS;
    int input_extent_1 = ROWS;
    int input_extent_0 = COLS;
    tiramisu::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint64, &fusion_dist);

    tiramisu::constant f_s0_c_loop_min("f_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_dist);
    tiramisu::constant f_s0_c_loop_extent("f_s0_c_loop_extent", tiramisu::expr(f_extent_2), tiramisu::p_int32, true, NULL, 0, &fusion_dist);
    tiramisu::constant f_s0_y_loop_min("f_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_dist);
    tiramisu::constant f_s0_y_loop_extent("f_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &fusion_dist);
    tiramisu::constant f_s0_x_loop_min("f_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_dist);
    tiramisu::constant f_s0_x_loop_extent("f_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &fusion_dist);
    tiramisu::computation f_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{f_s0[f_s0_c, f_s0_y, f_s0_x]: "
                        "(f_s0_c_loop_min <= f_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (tiramisu::expr((uint64_t)255) - input(tiramisu::var("f_s0_c"), tiramisu::var("f_s0_y"), tiramisu::var("f_s0_x"))), true, tiramisu::p_uint64, &fusion_dist);


    tiramisu::computation g_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{g_s0[g_s0_c, g_s0_y, g_s0_x]: "
                        "(f_s0_c_loop_min <= g_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= g_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= g_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (tiramisu::expr((uint64_t)2) * input(tiramisu::var("g_s0_c"), tiramisu::var("g_s0_y"), tiramisu::var("g_s0_x"))), true, tiramisu::p_uint64, &fusion_dist);


    tiramisu::computation h_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{h_s0[h_s0_c, h_s0_y, h_s0_x]: "
                        "(f_s0_c_loop_min <= h_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= h_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= h_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (f_s0(tiramisu::var("h_s0_c"), tiramisu::var("h_s0_y"), tiramisu::var("h_s0_x")) + g_s0(tiramisu::var("h_s0_c"), tiramisu::var("h_s0_y"), tiramisu::var("h_s0_x"))), true, tiramisu::p_uint64, &fusion_dist);

    tiramisu::computation k_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{k_s0[k_s0_c, k_s0_y, k_s0_x]: "
                        "(f_s0_c_loop_min <= k_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= k_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= k_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (f_s0(tiramisu::var("k_s0_c"), tiramisu::var("k_s0_y"), tiramisu::var("k_s0_x")) - g_s0(tiramisu::var("k_s0_c"), tiramisu::var("k_s0_y"), tiramisu::var("k_s0_x"))), true, tiramisu::p_uint64, &fusion_dist);

    var y1("y1"), y2("y2");
    f_s0.interchange(var("f_s0_c"), var("f_s0_y"));
    f_s0.split(var("f_s0_y"), rows_per_node, y1, y2);
    f_s0.tag_distribute_level(y1);
    f_s0.tag_distribute_level(y1);
    f_s0.tag_distribute_level(y1);
    f_s0.drop_rank_iter();

    g_s0.interchange(var("g_s0_c"), var("g_s0_y"));
    g_s0.split(var("g_s0_y"), rows_per_node, y1, y2);
    g_s0.tag_distribute_level(y1);
    g_s0.tag_distribute_level(y1);
    g_s0.tag_distribute_level(y1);
    g_s0.drop_rank_iter();

    h_s0.interchange(var("h_s0_c"), var("h_s0_y"));
    h_s0.split(var("h_s0_y"), rows_per_node, y1, y2);
    h_s0.tag_distribute_level(y1);
    h_s0.tag_distribute_level(y1);
    h_s0.tag_distribute_level(y1);
    h_s0.drop_rank_iter();

    k_s0.interchange(var("k_s0_c"), var("k_s0_y"));
    k_s0.split(var("k_s0_y"), rows_per_node, y1, y2);
    k_s0.tag_distribute_level(y1);
    k_s0.tag_distribute_level(y1);
    k_s0.tag_distribute_level(y1);
    k_s0.drop_rank_iter();

    f_s0.before(g_s0, computation::root);
    g_s0.before(h_s0, computation::root);
    h_s0.before(k_s0, computation::root);

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_2), tiramisu::expr(rows_per_node), tiramisu::expr(input_extent_0)}, tiramisu::p_uint64, tiramisu::a_input, &fusion_dist);
    tiramisu::buffer buff_f("buff_f", {tiramisu::expr(f_extent_2), tiramisu::expr(rows_per_node), tiramisu::expr(f_extent_0)}, tiramisu::p_uint64, tiramisu::a_output, &fusion_dist);
    tiramisu::buffer buff_g("buff_g", {tiramisu::expr(f_extent_2), tiramisu::expr(rows_per_node), tiramisu::expr(f_extent_0)}, tiramisu::p_uint64, tiramisu::a_output, &fusion_dist);
    tiramisu::buffer buff_h("buff_h", {tiramisu::expr(f_extent_2), tiramisu::expr(rows_per_node), tiramisu::expr(f_extent_0)}, tiramisu::p_uint64, tiramisu::a_output, &fusion_dist);
    tiramisu::buffer buff_k("buff_k", {tiramisu::expr(f_extent_2), tiramisu::expr(rows_per_node), tiramisu::expr(f_extent_0)}, tiramisu::p_uint64, tiramisu::a_output, &fusion_dist);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    f_s0.set_access("{f_s0[f_s0_c, f_s0_y, f_s0_x]->buff_f[f_s0_c, f_s0_y, f_s0_x]}");
    g_s0.set_access("{g_s0[g_s0_c, g_s0_y, g_s0_x]->buff_g[g_s0_c, g_s0_y, g_s0_x]}");
    h_s0.set_access("{h_s0[h_s0_c, h_s0_y, h_s0_x]->buff_h[h_s0_c, h_s0_y, h_s0_x]}");
    k_s0.set_access("{k_s0[k_s0_c, k_s0_y, k_s0_x]->buff_k[k_s0_c, k_s0_y, k_s0_x]}");

    fusion_dist.set_arguments({&buff_input, &buff_f, &buff_g, &buff_h, &buff_k});
    fusion_dist.gen_time_space_domain();
    fusion_dist.gen_isl_ast();
    fusion_dist.gen_halide_stmt();
    fusion_dist.dump_halide_stmt();
    fusion_dist.gen_halide_obj("build/generated_fusion_dist.o");


    return 0;
}

