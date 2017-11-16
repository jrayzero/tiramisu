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

    tiramisu::function cvtcolor_dist("cvtcolor_dist");
    
    int rows_per_rank = ROWS;
    int total_rows = rows_per_rank * NODES;
    constant rows("rows", expr(total_rows), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant cols("cols", expr(COLS), p_int32, true, NULL, 0, &cvtcolor_dist);
    constant channels("channels", expr(3), p_int32, true, NULL, 0, &cvtcolor_dist);
    var i("i"), j("j"), c("c");
    tiramisu::computation input("[channels, rows, cols]->{input[c, i, j]: 0<=c<channels and 0<=i<rows and 0<=j<cols}", expr(), false, tiramisu::p_float32, &cvtcolor_dist);
    tiramisu::expr _a(tiramisu::o_cast, tiramisu::p_float32, 1868);
    tiramisu::expr _b(tiramisu::o_cast, tiramisu::p_float32, 9617);
    tiramisu::expr _c(tiramisu::o_cast, tiramisu::p_float32, 4899);
    //    tiramisu::expr rgb_expr(input(2, i, j) * _a + input(1, i, j) * _b + input(0, i, j) * _c);
    //    tiramisu::expr descale(expr(o_cast, tiramisu::p_float32, (expr(o_cast, tiramisu::p_uint32, rgb_expr) + (uint32_t)(1 << 13)) >> (uint32_t)14));
    tiramisu::expr rgb_expr(input(2, i, j) * _a + input(1, i, j) * _b + input(0, i, j) * _c);
    tiramisu::expr descale(expr(o_cast, tiramisu::p_float32, (rgb_expr + 8192.0f)/16384.0f));
    tiramisu::computation RGB2Gray_s0("[channels, rows, cols]->{RGB2Gray_s0[i, j]: 0<=i<rows and 0<=j<cols}", expr(tiramisu::o_cast, tiramisu::p_float32, descale), true, tiramisu::p_float32, &cvtcolor_dist);

    // distribute stuff
    var i1("i1"), i2("i2");
    
    RGB2Gray_s0.split(i, rows_per_rank, i1, i2);    
    RGB2Gray_s0.tag_distribute_level(i1);
    RGB2Gray_s0.drop_rank_iter();

    tiramisu::buffer buff_input("buff_input", {3, rows_per_rank, COLS}, tiramisu::p_float32, tiramisu::a_input, &cvtcolor_dist);
    tiramisu::buffer buff_RGB2Gray("buff_RGB2Gray", {rows_per_rank, COLS}, tiramisu::p_float32, tiramisu::a_output, &cvtcolor_dist);
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

