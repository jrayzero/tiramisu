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

int main(int argc, char **argv)
{

    global::set_default_tiramisu_options();

    tiramisu::function dtest_03("dtest_03");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
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


    tiramisu::constant size1("SIZE1", tiramisu::expr(SIZE1), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    tiramisu::constant size0("SIZE0", tiramisu::expr(SIZE0), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    bx.separate_at(0, 3, tiramisu::expr(1), -3); // split by the color channel
    bx.get_update(0).rename_computation("bx_0");
    bx.get_update(1).rename_computation("bx_1");
    by.separate_at(0, 3, tiramisu::expr(1), -3); // split by the color channel
    by.get_update(0).rename_computation("by_0");
    by.get_update(1).rename_computation("by_1");

    channel chan("chan", p_uint8, {FIFO, ASYNC, NONBLOCK, MPI});
    channel chan_sync("chan", p_uint8, {FIFO, SYNC, BLOCK, MPI});
    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &dtest_03);
    send_recv fan_out = computation::create_transfer(
            "[SIZE1, SIZE0, one]->{send_0_1[c,z,y,x]: 0<=c<one and 1<=z<3 and 0<=y<SIZE1 and 0<=x<SIZE0}",
            "[SIZE1, SIZE0]->{recv_0_1[z,y,x]: 1<=z<3 and 0<=y<SIZE1 and 0<=x<SIZE0}", 0, z, &chan,
            &chan, blur_input(z, y, x), {&bx.get_update(1)}, &dtest_03);
    send_recv fan_in = computation::create_transfer(
						    "[My, Mx]->{send_1_0[z,y,x]: 1<=z<3 and 0<=y<My and 0<=x<Mx}",
						    "[My, Mx, one]->{recv_1_0[c,z,y,x]: 0<=c<one and 1<=z<3 and 0<=y<My and 0<=x<Mx}", z, 0, &chan,
						    &chan_sync, by.get_update(1)(0, y, x), {}, &dtest_03);

    tiramisu::wait wait_fan_out_r(fan_out.r->operator()(z, y, x), &dtest_03);
    wait_fan_out_r.separate_at(1, SIZE1, 6, -3);
    wait_fan_out_r.get_update(0).rename_computation("wait_r_0");
    wait_fan_out_r.get_update(1).rename_computation("wait_r_1");
    //    wait_fan_out_r.get_update(0).set_schedule_this_comp(false);
    //    wait_fan_out_r.get_update(1).set_schedule_this_comp(false);
    tiramisu::wait wait_fan_out_s(fan_out.s->operator()(c, z, y, x), &dtest_03);
    tiramisu::wait wait_fan_in_s(fan_in.s->operator()(z,y,x), &dtest_03);
    tiramisu::wait wait_fan_in_r(fan_in.r->operator()(c,z,y,x), &dtest_03);
    //   wait_fan_out_s.set_schedule_this_comp(false);
    //    wait_fan_in_s.set_schedule_this_comp(false);
    wait_fan_in_r.set_schedule_this_comp(false);
	//
    /*
     * Tag distribute level
     */

    bx.get_update(0).tag_distribute_level(c);
    bx.get_update(1).tag_distribute_level(c);
    by.get_update(0).tag_distribute_level(c);
    by.get_update(1).tag_distribute_level(c);    
    fan_out.s->tag_distribute_level(c);
    fan_out.r->tag_distribute_level(z);
    wait_fan_out_r.get_update(0).tag_distribute_level(z);
    wait_fan_out_r.get_update(1).tag_distribute_level(z);
    wait_fan_out_s.tag_distribute_level(c);
    fan_in.s->tag_distribute_level(z);
    fan_in.r->tag_distribute_level(c);
    wait_fan_in_s.tag_distribute_level(z);
    wait_fan_in_r.tag_distribute_level(c);

    /*
     * Additional constraints
     */

    dtest_03.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx]->{: Nc=3 and Mc=3 and Ny=3514 and My=3512 and Nx=2104 and Mx=2104}");

    /*
     * Scheduling
     */

    fan_out.s->collapse_many({collapser(3, 0, 2112)});
    wait_fan_out_s.collapse_many({collapser(3, 0, 2112)});
    fan_out.r->collapse_many({collapser(2, 0, 2112)});
    fan_in.s->collapse_many({collapser(2, 0, 2104)});
    fan_in.r->collapse_many({collapser(3, 0, 2104)});
    wait_fan_out_r.get_update(1).shift(y, -6);
    static_cast<tiramisu::wait*>(&wait_fan_out_r.get_update(0))->collapse_many({collapser(2, 0, 2112)});
    static_cast<tiramisu::wait*>(&wait_fan_out_r.get_update(1))->collapse_many({collapser(2, 0, 2112)});
    wait_fan_in_s.collapse_many({collapser(2, 0, 2104)});
    wait_fan_in_r.collapse_many({collapser(3, 0, 2104)});

    /*
     * Ordering
     */

    //    fan_out.s->before(bx.get_update(0), computation::root);
    fan_out.s->before(wait_fan_out_s, computation::root);
    wait_fan_out_s.before(bx.get_update(0), computation::root);
    bx.get_update(0).before(*fan_out.r, computation::root);
    fan_out.r->before(wait_fan_out_r.get_update(0), computation::root);
    wait_fan_out_r.get_update(0).before(wait_fan_out_r.get_update(1), computation::root);
    wait_fan_out_r.get_update(1).before(bx.get_update(1), computation::root);//y);
    bx.get_update(1).before(by.get_update(0), computation::root);
    by.get_update(0).before(by.get_update(1), computation::root);
    by.get_update(1).before(*fan_in.s, computation::root);//y);
    fan_in.r->after(by.get_update(0), computation::root);
    wait_fan_in_s.after(*fan_in.s, computation::root);
    wait_fan_in_r.after(*fan_in.r, z);//computation::root);

    /*
     * Name replacement
     */

    // TODO need to figure out how to automate this conversion
    generator::replace_expr_name(by.get_update(0).expression, "bx", "bx_0");
    generator::replace_expr_name(by.get_update(1).expression, "bx", "bx_1");
    generator::replace_expr_name(bx.get_update(1).expression, "blur_input", "recv_0_1");

    /*
     * Other scheduling
     */

    //    bx.get_update(0).tag_parallel_level(y);
    //    bx.get_update(1).tag_parallel_level(y);
    //    by.get_update(0).tag_parallel_level(y);
    //    by.get_update(1).tag_parallel_level(y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(SIZE2), tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                                tiramisu::p_uint8, tiramisu::a_input, &dtest_03);
    tiramisu::buffer buff_bx("buff_bx", {tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, tiramisu::a_temporary, &dtest_03);
    tiramisu::buffer buff_input_temp("buff_input_temp", {tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                             tiramisu::p_uint8, tiramisu::a_temporary, &dtest_03);
    tiramisu::buffer buff_bx_inter("buff_bx_inter", {tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, tiramisu::a_temporary, &dtest_03);
    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(by_ext_2), tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, tiramisu::a_output, &dtest_03);
    tiramisu::buffer buff_by_inter("buff_by_inter", {tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                                   tiramisu::p_uint8, tiramisu::a_temporary, &dtest_03);
    
    blur_input.set_access("{blur_input[i2, i1, i0]->buff_input[i2, i1, i0]}");
    fan_out.r->set_access("{recv_0_1[z,y,x]->buff_input_temp[y, x]}");
    bx.get_update(0).set_access("{bx_0[c, y, x]->buff_bx[y, x]}");
    bx.get_update(1).set_access("{bx_1[c, y, x]->buff_bx_inter[y, x]}");
    by.get_update(0).set_access("{by_0[c, y, x]->buff_by[c, y, x]}");
    by.get_update(1).set_access("{by_1[c, y, x]->buff_by_inter[y, x]}");
    fan_in.r->set_access("{recv_1_0[z,c,y,x]->buff_by[c,y,x]}");

    buffer fan_out_req_r_buff("fan_out_req_r_buff", {SIZE1}, tiramisu::p_req_ptr,
                              a_temporary, &dtest_03);
    buffer fan_out_req_s_buff("fan_out_req_s_buff", {2, SIZE1}, tiramisu::p_req_ptr,
                              a_temporary, &dtest_03);
    fan_out.r->set_req_access("{recv_0_1[z,y,x]->fan_out_req_r_buff[y]}");
    fan_out.s->set_req_access("{send_0_1[c,z,y,x]->fan_out_req_s_buff[z-1, y]}");

    buffer fan_in_req_r_buff("fan_in_req_r_buff", {2, by_ext_1}, tiramisu::p_req_ptr,
                              a_temporary, &dtest_03);
    buffer fan_in_req_s_buff("fan_in_req_s_buff", {by_ext_1}, tiramisu::p_req_ptr,
                              a_temporary, &dtest_03);
    fan_in.r->set_req_access("{recv_1_0[c,z,y,x]->fan_in_req_r_buff[z-1, y]}");
    fan_in.s->set_req_access("{send_1_0[z,y,x]->fan_in_req_s_buff[y]}");

    dtest_03.set_arguments({&buff_input, &buff_by});
    // Generate code
    dtest_03.gen_time_space_domain();
    dtest_03.lift_ops_to_library_calls();
    dtest_03.gen_isl_ast();
    dtest_03.gen_halide_stmt();
    dtest_03.gen_halide_obj("./build/generated_fct_dtest_03.o");

    // Some debugging
    dtest_03.dump_halide_stmt();

    return 0;
}
