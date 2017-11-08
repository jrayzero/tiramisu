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

    //    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = _COLS;
    int SIZE1 = _ROWS;
    int rows_per_node = SIZE1 / _NODES;
    tiramisu::function cvtcolor_tiramisu("cvtcolor_dist");

    // Output buffers.
    int RGB2Gray_extent_1 = SIZE1;
    int RGB2Gray_extent_0 = SIZE0;


    // Input buffers.
    int input_extent_2 = 3;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;

    tiramisu::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint64, &cvtcolor_tiramisu);

    // Define loop bounds for dimension "RGB2Gray_s0_v4".
    tiramisu::constant RGB2Gray_s0_v4_loop_min("RGB2Gray_s0_v4_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::constant RGB2Gray_s0_v4_loop_extent("RGB2Gray_s0_v4_loop_extent", tiramisu::expr(RGB2Gray_extent_1), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);

    // Define loop bounds for dimension "RGB2Gray_s0_v3".
    tiramisu::constant RGB2Gray_s0_v3_loop_min("RGB2Gray_s0_v3_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::constant RGB2Gray_s0_v3_loop_extent("RGB2Gray_s0_v3_loop_extent", tiramisu::expr(RGB2Gray_extent_0), tiramisu::p_int32, true, NULL, 0, &cvtcolor_tiramisu);
    tiramisu::computation RGB2Gray_s0("[RGB2Gray_s0_v4_loop_min, RGB2Gray_s0_v4_loop_extent, RGB2Gray_s0_v3_loop_min, RGB2Gray_s0_v3_loop_extent]->{RGB2Gray_s0[RGB2Gray_s0_v4, RGB2Gray_s0_v3]: "
                        "(RGB2Gray_s0_v4_loop_min <= RGB2Gray_s0_v4 <= ((RGB2Gray_s0_v4_loop_min + RGB2Gray_s0_v4_loop_extent) + -1)) and (RGB2Gray_s0_v3_loop_min <= RGB2Gray_s0_v3 <= ((RGB2Gray_s0_v3_loop_min + RGB2Gray_s0_v3_loop_extent) + -1))}",
                        tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, (((((tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, input(tiramisu::expr((int32_t)2), tiramisu::var("RGB2Gray_s0_v4"), tiramisu::var("RGB2Gray_s0_v3")))*tiramisu::expr((uint64_t)1868)) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, input(tiramisu::expr((int32_t)1), tiramisu::var("RGB2Gray_s0_v4"), tiramisu::var("RGB2Gray_s0_v3")))*tiramisu::expr((uint64_t)9617))) + (tiramisu::expr(tiramisu::o_cast, tiramisu::p_uint64, input(tiramisu::expr((int32_t)0), tiramisu::var("RGB2Gray_s0_v4"), tiramisu::var("RGB2Gray_s0_v3")))*tiramisu::expr((uint64_t)4899))) + (tiramisu::expr((uint64_t)1) << (tiramisu::expr((uint64_t)14) - tiramisu::expr((uint64_t)1)))) >> tiramisu::expr((uint64_t)14))), true, tiramisu::p_uint64, &cvtcolor_tiramisu);

    // distribute stuff
    var y1("y1"), y2("y2");
    
    RGB2Gray_s0.split(tiramisu::var("RGB2Gray_s0_v4"), rows_per_node, y1, y2);
    RGB2Gray_s0.tag_distribute_level(y1);
    RGB2Gray_s0.drop_rank_iter();

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(input_extent_2), rows_per_node, tiramisu::expr(input_extent_0)}, tiramisu::p_uint64, tiramisu::a_input, &cvtcolor_tiramisu);
    tiramisu::buffer buff_RGB2Gray("buff_RGB2Gray", {rows_per_node, tiramisu::expr(RGB2Gray_extent_0)}, tiramisu::p_uint64, tiramisu::a_output, &cvtcolor_tiramisu);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    RGB2Gray_s0.set_access("{RGB2Gray_s0[RGB2Gray_s0_v4, RGB2Gray_s0_v3]->buff_RGB2Gray[RGB2Gray_s0_v4, RGB2Gray_s0_v3]}");


    cvtcolor_tiramisu.set_arguments({&buff_input, &buff_RGB2Gray});
    cvtcolor_tiramisu.gen_time_space_domain();
    cvtcolor_tiramisu.gen_isl_ast();
    cvtcolor_tiramisu.gen_halide_stmt();
    cvtcolor_tiramisu.dump_halide_stmt();
    cvtcolor_tiramisu.gen_halide_obj("./build/generated_cvtcolor_dist.o");

    return 0;
}

