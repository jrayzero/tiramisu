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
//#include "../t_blur_sizes.h"

using namespace tiramisu;

#ifdef CHECK_RESULTS
#define TYPE p_uint8
#define CTYPE uint8_t
#else
#define TYPE p_uint64
#define CTYPE uint64_t
#endif

int main(int argc, char **argv)
{

    global::set_default_tiramisu_options();

    tiramisu::function dtest_04("dtest_04");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    int SIZE0 = 35000;//COLS;
    int SIZE1 = 40000;//ROWS;
    int SIZE2 = 3;//CHANNELS;

    int by_ext_2 = SIZE2;
    int by_ext_1 = SIZE1 - 8;
    int by_ext_0 = SIZE0 - 8;

    tiramisu::constant _Nc("_Nc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    tiramisu::constant _Ny("_Ny", (tiramisu::expr(by_ext_1) + tiramisu::expr((int32_t)2)), tiramisu::p_int32, true, NULL,
                           0, &dtest_04);
    tiramisu::constant _Nx("_Nx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_04);

    tiramisu::var c("c"), z("z"), y("y"), x("x"), y1("y1"), x1("x1"), y2("y2"), x2("x2");

    tiramisu::computation
            blur_input("[SIZE2, SIZE1, SIZE0]->{blur_input[i2, i1, i0]: (0 <= i2 <= (SIZE2 -1)) and "
                               "(0 <= i1 <= (SIZE1 -1)) and (0 <= i0 <= (SIZE0 -1))}", expr(), false, tiramisu::TYPE,
                       &dtest_04);


    tiramisu::constant Nc("Nc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    tiramisu::constant Ny("Ny", (tiramisu::expr(by_ext_1) + tiramisu::expr((int32_t)2)), tiramisu::p_int32, true, NULL,
                          0, &dtest_04);
    tiramisu::constant Nx("Nx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    tiramisu::computation bx("[Nc, Ny, Nx]->{bx[c, y, x]: (0 <= c <= (Nc -1)) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}",
                             (((blur_input(c, y, x) + blur_input(c, y, (x + tiramisu::expr((int32_t)1)))) +
                               blur_input(c, y, (x + tiramisu::expr((int32_t)2)))) / tiramisu::expr((CTYPE)3)), true,
                             tiramisu::TYPE, &dtest_04);

    tiramisu::constant Mc("Mc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    tiramisu::constant My("My", tiramisu::expr(by_ext_1), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    tiramisu::constant Mx("Mx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    tiramisu::computation
            by("[Mc, My, Mx]->{by[c, y, x]: (0 <= c <= (Mc -1)) and (0 <= y <= (My -1)) and (0 <= x <= (Mx -1))}",
               (((bx(c, y, x) + bx(c, (y + tiramisu::expr((int32_t)1)), x)) +
                 bx(c, (y + tiramisu::expr((int32_t)2)), x)) /
                tiramisu::expr((CTYPE)3)), true, tiramisu::TYPE, &dtest_04);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &dtest_04);
    bx.separate_at(0, one, -3); // split by the color channel
    bx.get_update(0).rename_computation("bx_0");
    bx.get_update(1).rename_computation("bx_1");
    by.separate_at(0, one, -3); // split by the color channel
    by.get_update(0).rename_computation("by_0");
    by.get_update(1).rename_computation("by_1");

    /*
     * Create communication
     */
    
    channel sync_block("sync_block", TYPE, {FIFO, SYNC, BLOCK, MPI});
    channel async_nonblock("sync_block", TYPE, {FIFO, ASYNC, NONBLOCK, MPI});
    send_recv fan_out = computation::create_transfer(
            "[Ny, Nx, Nc, one]->{fan_out_s[c,z,y,x]: 0<=c<one and one<=z<Nc and 0<=y<Ny+6 and 0<=x<Nx+8}",
            "[Ny, Nx, Nc, one]->{fan_out_r[z,y,x]: one<=z<Nc and 0<=y<Ny+6 and 0<=x<Nx+8}", 0, z, sync_block,
            sync_block, blur_input(z, y, x), {&bx.get_update(1)}, &dtest_04);
    send_recv fan_in = computation::create_transfer(
            "[My, Mx, Mc, one]->{fan_in_s[z,y,x]: one<=z<Mc and 0<=y<My and 0<=x<Mx}",
            "[My, Mx, Mc, one]->{fan_in_r[c,z,y,x]: 0<=c<one and one<=z<Mc and 0<=y<My and 0<=x<Mx}", z, 0, sync_block,
            sync_block, by.get_update(1)(0, y, x), {}, &dtest_04);
    
    tiramisu::send *fan_out_s = fan_out.s;
    tiramisu::recv *fan_out_r = fan_out.r;
    tiramisu::send *fan_in_s = fan_in.s;
    tiramisu::recv *fan_in_r = fan_in.r;
    
    fan_out_s->separate_at(2, 6, -3);
    fan_out_s->get_update(0).rename_computation("fan_out_s_0");
    fan_out_s->get_update(1).rename_computation("fan_out_s_1");
    fan_out_s->get_update(1).shift(y, -6);
    fan_out_s->get_update(1).interchange(z,y);
    fan_out_r->separate_at(1, 6, -3);
    fan_out_r->get_update(0).rename_computation("fan_out_r_0");
    fan_out_r->get_update(1).rename_computation("fan_out_r_1");
    fan_out_r->get_update(1).shift(y, -6);

    /*
     * Additional constraints
     */

    dtest_04.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx, _Nc, _Ny, _Nx]->{: Nc=_Nc and Mc=_Nc and Ny=_Ny+2 and My=_Ny and Nx=_Nx and Mx=_Nx}");

    /*
     * Collapsing
     */

    static_cast<send*>(&fan_out_s->get_update(0))->collapse_many({collapser(3, 0, SIZE0)});
    static_cast<send*>(&fan_out_s->get_update(1))->collapse_many({collapser(3, 0, SIZE0)});
    static_cast<send*>(&fan_out_r->get_update(0))->collapse_many({collapser(2, 0, SIZE0)});
    static_cast<send*>(&fan_out_r->get_update(1))->collapse_many({collapser(2, 0, SIZE0)});
    fan_in.s->collapse_many({collapser(2, 0, by_ext_0)});
    fan_in.r->collapse_many({collapser(3, 0, by_ext_0)});

    /*
     * Other scheduling (match halide code)
     */

//    by.get_update(0).split(y, 1000, y1, y2);
//    by.get_update(0).tag_parallel_level(y1);
//    by.get_update(0).set_loop_level_names({3}, {"x"});
//    by.get_update(0).vectorize(x, 8);
//    by.get_update(0).set_loop_level_names({0, 1, 3}, {"c", "y1", "x"});
//
//    bx.get_update(0).vectorize(x, 8);
//    bx.get_update(0).set_loop_level_names({0, 1, 2}, {"c", "y", "x"});
//
//    by.get_update(1).split(y, 1000, y1, y2);
//    by.get_update(1).tag_parallel_level(y1);
//    by.get_update(1).set_loop_level_names({3}, {"x"});
//    by.get_update(1).vectorize(x, 8);
//    by.get_update(1).set_loop_level_names({0, 1, 3}, {"c", "y1", "x"});
//
//    bx.get_update(1).vectorize(x, 8);
//    bx.get_update(1).set_loop_level_names({0, 1}, {"c", "z"});

    /*
     * Tag distribute level
     */

    bx.get_update(0).tag_distribute_level(c);
    bx.get_update(1).tag_distribute_level(c);
    by.get_update(0).tag_distribute_level(c);
    by.get_update(1).tag_distribute_level(c);
    fan_out_s->get_update(0).tag_distribute_level(c);
    fan_out_s->get_update(1).tag_distribute_level(c);
    fan_out_r->get_update(0).tag_distribute_level(z);
    fan_out_r->get_update(1).tag_distribute_level(z);
    fan_in_s->tag_distribute_level(z);
    fan_in_r->tag_distribute_level(c);

    /*
     * Ordering
     */

    fan_out_s->get_update(0).before(fan_out_s->get_update(1), computation::root);
    fan_out_s->get_update(1).before(bx.get_update(0), y);
    bx.get_update(0).before(by.get_update(0), computation::root);
    by.get_update(0).before(fan_out_r->get_update(0), computation::root);
    fan_out_r->get_update(0).before(fan_out_r->get_update(1), computation::root);
    fan_out_r->get_update(1).before(bx.get_update(1), y);
    bx.get_update(1).before(by.get_update(1), computation::root);
    by.get_update(1).before(*fan_in_s, computation::root);
    fan_in_s->before(*fan_in_r, computation::root);

    /*
     * Renaming
     */
    
    generator::replace_expr_name(by.get_update(0).expression, "bx", "bx_0");
    generator::replace_expr_name(by.get_update(1).expression, "bx", "bx_1");
    generator::replace_expr_name(bx.get_update(1).expression, "blur_input", "fan_out_r_0"); // doesn't really matter which one you give it

    /*
     * Set the message tags
     */

    static_cast<send*>(&fan_out_s->get_update(0))->override_msg_tag(y);
    static_cast<send*>(&fan_out_s->get_update(1))->override_msg_tag(y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(SIZE2), tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                                tiramisu::TYPE, tiramisu::a_input, &dtest_04);
    tiramisu::buffer buff_bx("buff_bx", {tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                             tiramisu::TYPE, tiramisu::a_output, &dtest_04);
    tiramisu::buffer buff_input_temp("buff_input_temp", {tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                                     tiramisu::TYPE, tiramisu::a_output, &dtest_04);
    tiramisu::buffer buff_bx_inter("buff_bx_inter", {tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                                   tiramisu::TYPE, tiramisu::a_output, &dtest_04);
    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(by_ext_2), tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                             tiramisu::TYPE, tiramisu::a_output, &dtest_04);
    tiramisu::buffer buff_by_inter("buff_by_inter", {tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                                   tiramisu::TYPE, tiramisu::a_output, &dtest_04);

    blur_input.set_access("{blur_input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    fan_out_r->get_update(0).set_access("{fan_out_r_0[z,y,x]->buff_input_temp[y, x]}");
    fan_out_r->get_update(1).set_access("{fan_out_r_1[z,y,x]->buff_input_temp[y, x]}");
    fan_in.r->set_access("{fan_in_r[z,c,y,x]->buff_by[c,y,x]}");
    bx.get_update(0).set_access("{bx_0[c, y, x]->buff_bx[y, x]}");
    bx.get_update(1).set_access("{bx_1[c, y, x]->buff_bx_inter[y, x]}");
    by.get_update(0).set_access("{by_0[c, y, x]->buff_by[c, y, x]}");
    by.get_update(1).set_access("{by_1[c, y, x]->buff_by_inter[y, x]}");

    dtest_04.set_arguments({&buff_input, &buff_bx, &buff_input_temp, &buff_bx_inter, &buff_by_inter, &buff_by});

    // Generate code
    dtest_04.gen_time_space_domain();
    dtest_04.lift_ops_to_library_calls();
    dtest_04.gen_isl_ast();
    dtest_04.gen_halide_stmt();
    dtest_04.gen_halide_obj("./build/generated_fct_dtest_04.o");

    // Some debugging
    dtest_04.dump_halide_stmt();

    return 0;
}
