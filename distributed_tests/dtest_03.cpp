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

#define USE_FIXED_VALUES 0
#define SEPARATED 0

using namespace tiramisu;

// TODO need to separate the bx computation so that rank0 reads from b_input, not b_input_temp
// Add fan_in communication

int main(int argc, char **argv)
{

    global::set_default_tiramisu_options();

    tiramisu::function dtest_03("dtest_03");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("/Users/JRay/ClionProjects/tiramisu/images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    int by_ext_2 = SIZE2;
    int by_ext_1 = SIZE1 - 8;
    int by_ext_0 = SIZE0 - 8;

    tiramisu::var c("c"), z("z"), y("y"), x("x"), y1("y1"), x1("x1"), y2("y2"), x2("x2");

    tiramisu::computation
            blur_input("[SIZE2, SIZE1, SIZE0]->{blur_input[i2, i1, i0]: (0 <= i2 <= (SIZE2 -1)) and "
                               "(0 <= i1 <= (SIZE1 -1)) and (0 <= i0 <= (SIZE0 -1))}", expr(), false, tiramisu::p_uint8,
                       &dtest_03);


    tiramisu::constant Nc("Nc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    tiramisu::constant Ny("Ny", (tiramisu::expr(by_ext_1) + tiramisu::expr((int32_t)2)), tiramisu::p_int32, true, NULL,
                          0, &dtest_03);
    tiramisu::constant Nx("Nx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    tiramisu::computation bx("[Nc, Ny, Nx]->{bx[c, y, x]: (0 <= c <= (Nc -1)) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}",
                             (((blur_input(c, y, x) + blur_input(c, y, (x + tiramisu::expr((int32_t)1)))) +
                               blur_input(c, y, (x + tiramisu::expr((int32_t)2)))) / tiramisu::expr((uint8_t)3)), true,
                             tiramisu::p_uint8, &dtest_03);

    tiramisu::constant Mc("Mc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    tiramisu::constant My("My", tiramisu::expr(by_ext_1), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    tiramisu::constant Mx("Mx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    tiramisu::computation
            by("[Mc, My, Mx]->{by[c, y, x]: (0 <= c <= (Mc -1)) and (0 <= y <= (My -1)) and (0 <= x <= (Mx -1))}",
               (((bx(c, y, x) + bx(c, (y + tiramisu::expr((int32_t)1)), x)) +
                 bx(c, (y + tiramisu::expr((int32_t)2)), x)) /
                tiramisu::expr((uint8_t)3)), true, tiramisu::p_uint8, &dtest_03);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    bx.separate_at(0, 3, tiramisu::expr(1), -3); // split by the color channel
    bx.get_update(0).rename_computation("bx_0");
    bx.get_update(1).rename_computation("bx_1");
    by.separate_at(0, 3, tiramisu::expr(1), -3); // split by the color channel
    by.get_update(0).rename_computation("by_0");
    by.get_update(1).rename_computation("by_1");
    bx.get_update(0).tag_distribute_level(c);
    bx.get_update(1).tag_distribute_level(c);
    by.get_update(0).tag_distribute_level(c);
    by.get_update(1).tag_distribute_level(c);

    channel chan_sync_block("chan_sync_block", p_uint8, {FIFO, SYNC, BLOCK, MPI});

    send_recv fan_out = computation::create_transfer(
            "[Ny, Nx]->{send_0_1[c,z,y,x]: 0<=c<1 and 1<=z<3 and 0<=y<Ny and 0<=x<Nx}",
            "[Ny, Nx]->{recv_0_1[z,y,x]: 1<=z<3 and 0<=y<Ny and 0<=x<Nx}", 0, z, &chan_sync_block,
            &chan_sync_block, blur_input(z, y, x), {&bx.get_update(1)}, &dtest_03);

    fan_out.s->tag_distribute_level(c);
    fan_out.r->tag_distribute_level(z);

    // I believe these are for boundary constraints or something like that
    dtest_03.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx]->{: Nc=3 and Mc=3 and Ny=3514 and My=3512 and Nx=2104 and Mx=2104}");
    fan_out.s->before(bx.get_update(0), computation::root);
    bx.get_update(0).before(*fan_out.r, computation::root);
    fan_out.r->before(bx.get_update(1), computation::root);
    bx.get_update(1).before(by.get_update(0), computation::root);
    by.get_update(0).before(by.get_update(1), computation::root);

    // TODO need to figure out how to automate this conversion
    generator::replace_expr_name(by.get_update(0).expression, "bx", "bx_0");
    generator::replace_expr_name(by.get_update(1).expression, "bx", "bx_1");

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(SIZE2), tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                             tiramisu::p_uint8, tiramisu::a_input, &dtest_03);
    buff_input.distribute({1, tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)}, "b_input_temp");
    tiramisu::buffer buff_bx("buff_bx", {1, tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, tiramisu::a_temporary, &dtest_03);
    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(by_ext_2), tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, tiramisu::a_output, &dtest_03);
    tiramisu::buffer buff_by_inter("buff_by_inter", {1, tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, tiramisu::a_temporary, &dtest_03);

    blur_input.set_access("{blur_input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    fan_out.r->set_access("{recv_0_1[z,y,x]->b_input_temp[0, y, x]}");
    bx.get_update(0).set_access("{bx_0[c, y, x]->buff_bx[c, y, x]}");
    bx.get_update(1).set_access("{bx_1[c, y, x]->buff_bx[c, y, x]}");
    by.get_update(0).set_access("{by_0[c, y, x]->buff_by[c, y, x]}");
    by.get_update(1).set_access("{by_1[c, y, x]->buff_by_inter[c, y, x]}");

    dtest_03.set_arguments({&buff_input, &buff_by});
    // Generate code
    dtest_03.gen_time_space_domain();
    dtest_03.lift_ops_to_library_calls();
    dtest_03.gen_isl_ast();
    dtest_03.gen_halide_stmt();
    dtest_03.gen_halide_obj("build/generated_fct_dtest_03.o");

    // Some debugging
    dtest_03.dump_halide_stmt();

    return 0;
}