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
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    //    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    //    int SIZE0 = _COLS;
    //    int SIZE1 = _ROWS;
    //    int rows_per_node = SIZE1 / NODES;
    tiramisu::function cvtcolor_dist("cvtcolor_dist");

    // Output buffers.
    //    int RGB2Gray_extent_1 = SIZE1;
    //    int RGB2Gray_extent_0 = SIZE0;

    // Input buffers.
    //    int input_extent_2 = 3;
    //    int input_extent_1 = SIZE1;
    //    int input_extent_0 = SIZE0;
    
    int rows_per_node = _ROWS / NODES;
    constant rows("rows", expr(_ROWS), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant cols("cols", expr(_COLS), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant channels("channels", expr(_CHANNELS), p_int32, true, NULL, 0, &cvtcolor_dist);
    var i("i"), j("j"), c("c");
    tiramisu::computation input("[channels, rows, cols]->{input[c, i, j]: 0<=c<channels and 0<=i<rows and 0<=j<cols}", expr(), false, tiramisu::p_uint64, &cvtcolor_dist);
    tiramisu::expr _a(tiramisu::o_cast, tiramisu::p_uint64, 1868);
    tiramisu::expr _b(tiramisu::o_cast, tiramisu::p_uint64, 9617);
    tiramisu::expr _c(tiramisu::o_cast, tiramisu::p_uint64, 4899);
    tiramisu::expr rgb_expr(input(2, i, j) * _a + input(1, i, j) * _b + input(0, i, j) * _c);
    tiramisu::expr descale(expr(o_cast, tiramisu::p_uint64, (expr(o_cast, tiramisu::p_uint32, rgb_expr) + (uint32_t)(1 << 13)) >> (uint32_t)14));//expr(tiramisu::o_cast, tiramisu::p_uint64, ((1 << (14-1)) >> (14))));
    tiramisu::computation RGB2Gray_s0("[channels, rows, cols]->{RGB2Gray_s0[i, j]: 0<=i<rows and 0<=j<cols}", expr(tiramisu::o_cast, tiramisu::p_uint64, descale), true, tiramisu::p_uint64, &cvtcolor_dist);

    // Define loop bounds for dimension "RGB2Gray_s0_v4".
    //    tiramisu::constant RGB2Gray_s0_v4_loop_min("RGB2Gray_s0_v4_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_dist);
    //    tiramisu::constant RGB2Gray_s0_v4_loop_extent("RGB2Gray_s0_v4_loop_extent", tiramisu::expr(RGB2Gray_extent_1), tiramisu::p_int32, true, NULL, 0, &cvtcolor_dist);

    // Define loop bounds for dimension "RGB2Gray_s0_v3".
    //    tiramisu::constant RGB2Gray_s0_v3_loop_min("RGB2Gray_s0_v3_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_dist);
    //    tiramisu::constant RGB2Gray_s0_v3_loop_extent("RGB2Gray_s0_v3_loop_extent", tiramisu::expr(RGB2Gray_extent_0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_dist);
    //    tiramisu::computation RGB2Gray_s0("[RGB2Gray_s0_v4_loop_min, RGB2Gray_s0_v4_loop_extent, RGB2Gray_s0_v3_loop_min, RGB2Gray_s0_v3_loop_extent]->{RGB2Gray_s0[RGB2Gray_s0_v4, RGB2Gray_s0_v3]: "
    //    //                        "(RGB2Gray_s0_v4_loop_min <= RGB2Gray_s0_v4 <= ((RGB2Gray_s0_v4_loop_min + RGB2Gray_s0_v4_loop_extent) + -1)) and (RGB2Gray_s0_v3_loop_min <= RGB2Gray_s0_v3 <= ((RGB2Gray_s0_v3_loop_min + RGB2Gray_s0_v3_loop_extent) + -1))}",
    //                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, input(tiramisu::expr((int32_t)2), tiramisu::var("RGB2Gray_s0_v4"), tiramisu::var("RGB2Gray_s0_v3")))*tiramisu::expr((uint64_t)1868)) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, input(tiramisu::expr((int32_t)1), tiramisu::var("RGB2Gray_s0_v4"), tiramisu::var("RGB2Gray_s0_v3")))*tiramisu::expr((uint64_t)9617))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, input(tiramisu::expr((int32_t)0), tiramisu::var("RGB2Gray_s0_v4"), tiramisu::var("RGB2Gray_s0_v3")))*tiramisu::expr((uint64_t)4899))) + (tiramisu::expr((uint64_t)1) << (tiramisu::expr((uint64_t)14) - tiramisu::expr((uint64_t)1)))) >> tiramisu::expr((uint64_t)14))), true, tiramisu::p_uint64, &cvtcolor_dist);

    // distribute stuff
    var i1("i1"), i2("i2");
    
    RGB2Gray_s0.split(i, rows_per_node, i1, i2);
    RGB2Gray_s0.tag_distribute_level(i1);
    RGB2Gray_s0.drop_rank_iter();

    tiramisu::buffer buff_input("buff_input", {_CHANNELS, rows_per_node, _COLS}, tiramisu::p_uint64, tiramisu::a_input, &cvtcolor_dist);
    tiramisu::buffer buff_RGB2Gray("buff_RGB2Gray", {rows_per_node, _COLS}, tiramisu::p_uint64, tiramisu::a_output, &cvtcolor_dist);
    input.set_access("{input[c, i, j]->buff_input[c, i, j]}");
    RGB2Gray_s0.set_access("{RGB2Gray_s0[i,j]->buff_RGB2Gray[i,j]}");


    cvtcolor_dist.set_arguments({&buff_input, &buff_RGB2Gray});
    cvtcolor_dist.gen_time_space_domain();
    cvtcolor_dist.gen_isl_ast();
    cvtcolor_dist.gen_halide_stmt();
    cvtcolor_dist.dump_halide_stmt();
    cvtcolor_dist.gen_halide_obj("./build/generated_cvtcolor_dist.o");

    return 0;
}

