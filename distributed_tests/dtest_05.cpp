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

using namespace tiramisu;

#ifdef CHECK_RESULTS
#define TYPE p_uint8
#define CTYPE uint8_t
#else
#define TYPE p_uint64
#define CTYPE uint64_t
#endif

#define DISTRIBUTED

int main(int argc, char **argv)
{

    global::set_default_tiramisu_options();

    tiramisu::function dtest_05("dtest_05");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    int SIZE1 = 40000; // Columns
    int SIZE0 = 35000; // Rows

    int by_ext_1 = SIZE1 - 8;
    int by_ext_0 = SIZE0 - 8;

    tiramisu::constant _Ny("_Ny", (tiramisu::expr(by_ext_1) + tiramisu::expr((int32_t)2)), tiramisu::p_int32, true, NULL,
                           0, &dtest_05);
    tiramisu::constant _Nx("_Nx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_05);

    tiramisu::var c("c"), d("d"), z("z"), y("y"), x("x"), y1("y1"), x1("x1"), y2("y2"), x2("x2");

    tiramisu::computation
            blur_input("[SIZE1, SIZE0]->{blur_input[i1, i0]: (0 <= i1 <= (SIZE1 -1)) and (0 <= i0 <= (SIZE0 -1))}", expr(), false, tiramisu::TYPE,
                       &dtest_05);


    tiramisu::constant Ny("Ny", (tiramisu::expr(by_ext_1) + tiramisu::expr((int32_t)2)), tiramisu::p_int32, true, NULL,
                          0, &dtest_05);
    tiramisu::constant Nx("Nx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_05);
    tiramisu::computation bx("[Ny, Nx]->{bx[y, x]: (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}",
                             (((blur_input(y, x) + blur_input(y, (x + tiramisu::expr((int32_t)1)))) +
                               blur_input(y, (x + tiramisu::expr((int32_t)2)))) / tiramisu::expr((CTYPE)3)), true,
                             tiramisu::TYPE, &dtest_05);

    tiramisu::constant My("My", tiramisu::expr(by_ext_1), tiramisu::p_int32, true, NULL, 0, &dtest_05);
    tiramisu::constant Mx("Mx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0, &dtest_05);
    tiramisu::computation
            by("[My, Mx]->{by[y, x]: (0 <= y <= (My -1)) and (0 <= x <= (Mx -1))}",
               (((bx(y, x) + bx((y + tiramisu::expr((int32_t)1)), x)) +
                 bx((y + tiramisu::expr((int32_t)2)), x)) /
                tiramisu::expr((CTYPE)3)), true, tiramisu::TYPE, &dtest_05);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

#ifdef DISTRIBUTED
    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &dtest_05);

    bx.split(y, 4000, y1, y2);
    by.split(y, 4000, y1, y2);

    bx.separate_at(0, one, -3);
    bx.get_update(0).rename_computation("bx_0");
    bx.get_update(1).rename_computation("bx_1");
    by.separate_at(0, one, -3);
    by.get_update(0).rename_computation("by_0");
    by.get_update(1).rename_computation("by_1");

    //    bx.get_update(1).set_schedule_this_comp(false);
    //    by.get_update(1).set_schedule_this_comp(false);

     /*
      * Create communication
      */

     generator::replace_expr_name(by.get_update(0).expression, "bx", "bx_0");
     generator::replace_expr_name(by.get_update(1).expression, "bx", "bx_1");

     channel sync_block("sync_block", TYPE, {FIFO, SYNC, BLOCK, MPI});
     channel async_nonblock("async_nonblock", TYPE, {FIFO, ASYNC, NONBLOCK, MPI});
     channel async_block("async_block", TYPE, {FIFO, ASYNC, BLOCK, MPI});
     send_recv fan_out = computation::create_transfer(
             "[Ny, Nx, one]->{fan_out_s[c,d,y,x]: 0<=c<one and one<=d<10 and d*4000<=y<(d+1)*4000 and 0<=x<Nx+8}", // we want 10 nodes
             "[Ny, Nx, one]->{fan_out_r[y,x]: 0<=y<Ny+6-4000 and 0<=x<Nx+8}", 0, var("c3"), 0, var("c1"), sync_block, // TODO I have to specify the transformed iterator here. Not good.
             sync_block, blur_input(y, x), {&bx.get_update(1)}, &dtest_05);
     send_recv fan_in = computation::create_transfer(
             "[My, Mx, one]->{fan_in_s[y,x]: 0<=y<My+8-4000 and 0<=x<Mx}",
             "[My, Mx, one]->{fan_in_r[c,d,y,x]: 0<=c<one and one<=d<10 and d*4000<=y<(d+1)*4000 and 0<=x<Mx}", var("c1"), 0, var("c3"), 0, sync_block,
             sync_block, by.get_update(1)(y, x), {}, &dtest_05);

     generator::replace_expr_name(bx.get_update(1).expression, "blur_input", "fan_out_r"); // doesn't really matter which five you give it

     tiramisu::send *fan_out_s = fan_out.s;
     tiramisu::recv *fan_out_r = fan_out.r;
     tiramisu::send *fan_in_s = fan_in.s;
     tiramisu::recv *fan_in_r = fan_in.r;

     fan_out_r->split(y, 4000, y1, y2);
     fan_out_r->shift(y1, 1);
     fan_in_s->split(y, 4000, y1, y2);
     fan_in_s->shift(y1, 1);

     //fan_in_s->set_schedule_this_comp(false);
     //     fan_in_r->set_schedule_this_comp(false);
     //
     /*
      * Additional constraints
      */

     dtest_05.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx, _Nc, _Ny, _Nx]->{: Nc=_Nc and Mc=_Nc and Ny=_Ny+2 and My=_Ny and Nx=_Nx and Mx=_Nx}");

    /*
     * Collapsing
     */

     static_cast<send*>(&fan_out_s->get_update(0))->collapse_many({collapser(3, 0, SIZE0)});
     static_cast<send*>(&fan_out_r->get_update(0))->collapse_many({collapser(2, 0, SIZE0)});
     fan_in.s->collapse_many({collapser(2, 0, by_ext_0)});
     fan_in.r->collapse_many({collapser(3, 0, by_ext_0)});

     /*
      * Other scheduling (match halide code)
      */

 //    by.get_update(0).split(y, 1000, y1, y2);
 //    by.get_update(0).tag_parallel_level(y1);
 //    by.get_update(0).set_loop_level_names({3}, {"x"});
 //    by.get_update(0).vectorize(x, 8);
 //    by.get_update(0).set_loop_level_names({0}, {"c"});
 //
 //    bx.get_update(0).vectorize(x, 8);
 //    bx.get_update(0).set_loop_level_names({0, 1, 2}, {"c", "y", "x"});
 //
 //    by.get_update(1).split(y, 1000, y1, y2);
 //    by.get_update(1).tag_parallel_level(y1);
 //    by.get_update(1).set_loop_level_names({3}, {"x"});
 //    by.get_update(1).vectorize(x, 8);
 //    by.get_update(1).set_loop_level_names({0}, {"c"});
 //
 //    bx.get_update(1).vectorize(x, 8);
 //    bx.get_update(1).set_loop_level_names({0, 1, 2}, {"c", "y", "x"});

     /*
      * Ordering
      */

 //    fan_out_s->get_update(0).before(fan_out_s->get_update(1), computation::root);
 //    fan_out_s->get_update(1).before(bx.get_update(0), computation::root);//y);
     fan_out_s->before(bx.get_update(0), computation::root);
     bx.get_update(0).before(by.get_update(0), computation::root);
     by.get_update(0).before(fan_out_r->get_update(0), computation::root);
 //    fan_out_r->get_update(0).before(fan_out_r->get_update(1), computation::root);
 //    fan_out_r->get_update(1).before(bx.get_update(1), computation::root);//y);
     fan_out_r->get_update(0).before(bx.get_update(1), computation::root);//y);
    bx.get_update(1).before(by.get_update(1), computation::root);
    by.get_update(1).before(*fan_in_s, computation::root);
    fan_in_s->before(*fan_in_r, computation::root);
//    fan_in_s->before(wait_fan_out_s, computation::root);
//    wait_fan_out_s.before(*fan_in_r, computation::root);//y);

    /*
     * Tag distribute level
     */

    bx.get_update(0).tag_distribute_level(y1);
    bx.get_update(1).tag_distribute_level(y1);
    by.get_update(0).tag_distribute_level(y1);
    by.get_update(1).tag_distribute_level(y1);
    fan_out_s->get_update(0).tag_distribute_level(c);
    fan_out_r->get_update(0).tag_distribute_level(y1);
    fan_in_s->tag_distribute_level(y1);
    fan_in_r->tag_distribute_level(c);
//    wait_fan_out_s.tag_distribute_level(y);

    /*
     * Set the message tags
     */

    static_cast<send*>(&fan_out_s->get_update(0))->override_msg_tag(var("c5")-var("c3")*4000);
    fan_out_r->override_msg_tag(var("c3"));
//    static_cast<send*>(&fan_out_s->get_update(1))->override_msg_tag(y);

    /*
     * Specify necessary offsets
     */

    bx.get_update(1).offset = -(var("c1") - 1) * 4000 * by_ext_0;
    by.get_update(1).offset = -(var("c1") - 1) * 4000 * by_ext_0;
    fan_out_r->offset = -(var("c1") - 1) * 4000 * SIZE0;
    fan_in_s->offset = -(var("c1") - 1) * 4000 * by_ext_0;

#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

#ifdef DISTRIBUTED
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                                tiramisu::TYPE, tiramisu::a_input, &dtest_05);
    tiramisu::buffer buff_bx("buff_bx", {tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                             tiramisu::TYPE, tiramisu::a_output, &dtest_05);
    tiramisu::buffer buff_input_temp("buff_input_temp", {tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                                     tiramisu::TYPE, tiramisu::a_output, &dtest_05);
    tiramisu::buffer buff_bx_inter("buff_bx_inter", {tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                                   tiramisu::TYPE, tiramisu::a_output, &dtest_05);
    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                             tiramisu::TYPE, tiramisu::a_output, &dtest_05);
    tiramisu::buffer buff_by_inter("buff_by_inter", {tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                                   tiramisu::TYPE, tiramisu::a_output, &dtest_05);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");
    fan_out_r->get_update(0).set_access("{fan_out_r[y, x]->buff_input_temp[y, x]}");
//    fan_out_r->get_update(1).set_access("{fan_out_r_1[y,x]->buff_input_temp[y, x]}");
    fan_in_r->set_access("{fan_in_r[c,d,y,x]->buff_by[y, x]}");
    bx.get_update(0).set_access("{bx_0[y, x]->buff_bx[y, x]}");
    bx.get_update(1).set_access("{bx_1[y, x]->buff_bx_inter[y, x]}");
    by.get_update(0).set_access("{by_0[y, x]->buff_by[y, x]}");
    by.get_update(1).set_access("{by_1[y, x]->buff_by_inter[y, x]}");

    buffer fan_out_req_s_buff("fan_out_req_s_buff", {9, 4000}, tiramisu::p_req_ptr, a_temporary, &dtest_05);
    static_cast<send*>(&fan_out_s->get_update(0))->set_req_access("{fan_out_s[c,z,y,x]->fan_out_req_s_buff[z-1, y]}");

    // TODO This is problematic here. For some reason I have to make this a 3D buffer???
//    buffer fan_out_req_s_buff("fan_out_req_s_buff", {2, SIZE1, 1}, tiramisu::p_req_ptr, a_temporary, &dtest_05);
//    static_cast<send*>(&fan_out_s->get_update(0))->set_req_access("{fan_out_s_0[y,x]->fan_out_req_s_buff[z-1, y, 0]}");
//    static_cast<send*>(&fan_out_s->get_update(1))->set_req_access("{fan_out_s_1[y,x]->fan_out_req_s_buff[z-1, y, 0]}");

    dtest_05.set_arguments({&buff_input, &buff_bx, &buff_input_temp, &buff_bx_inter, &buff_by_inter, &buff_by});
#endif
    // Generate code
    dtest_05.gen_time_space_domain();
    dtest_05.lift_ops_to_library_calls();
    dtest_05.gen_isl_ast();
    dtest_05.gen_halide_stmt();
    dtest_05.gen_halide_obj("./build/generated_fct_dtest_05.o");

    // Some debugging
    dtest_05.dump_halide_stmt();

    return 0;
}
