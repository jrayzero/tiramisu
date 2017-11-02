//
// Created by Jessica Ray on 11/2/17.
//

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

    function dblurxy("dblurxy");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    int ROWS = _ROWS;
    int COLS = _COLS;

    int by_rows = ROWS - 8;
    int by_cols = COLS - 8;

    constant _Ny("_Ny", (expr(by_rows) + expr((int32_t)2)), p_int32, true, NULL,
                           0, &dblurxy);
    constant _Nx("_Nx", expr(by_cols), p_int32, true, NULL, 0, &dblurxy);

    var y("y"), x("x");

    computation
            blur_input("[ROWS, COLS]->{blur_input[i1, i0]: 0<=i1<ROWS and 0<=i0<COLS}", expr(), false,
                       p_uint64, &dblurxy);


    constant Ny("Ny", (expr(by_rows) + expr((int32_t)2)), p_int32, true, NULL,
                          0, &dblurxy);
    constant Nx("Nx", expr(by_cols), p_int32, true, NULL, 0, &dblurxy);
    constant My("My", expr(by_rows), p_int32, true, NULL, 0, &dblurxy);
    constant Mx("Mx", expr(by_cols), p_int32, true, NULL, 0, &dblurxy);

    computation bx("[Ny, Nx]->{bx[y, x]: 0<=y<Ny and 0<=x<Nx}",
                             (((blur_input(y, x) +
                                blur_input(y, (x + expr((int32_t)1)))) +
                               blur_input(y, (x + expr((int32_t)2)))) / expr((uint64_t)3)),
                             true, p_uint64, &dblurxy);

    computation by("[My, Mx]->{by[y, x]: 0<=y<My and 0<=x<Mx}",
                             (((bx(y, x) +
                                bx((y + expr((int32_t)1)), x)) +
                               bx((y + expr((int32_t)2)), x)) / expr((uint64_t)3)),
                             true, p_uint64, &dblurxy);

    /*
     * Additional constraints
     */

    dblurxy.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx, _Nc, _Ny, _Nx]->{: Nc=_Nc and Mc=_Nc and Ny=_Ny+2 and My=_Ny and Nx=_Nx and Mx=_Nx}");

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    
    constant one("one", expr(1), p_int32, true, NULL, 0, &dblurxy);
    constant ten("ten", expr(10), p_int32, true, NULL, 0, &dblurxy);
    constant twenty("twenty", expr(20), p_int32, true, NULL, 0, &dblurxy);

    /*
     * Prep for distribution by splitting the outer dimension
     */
    
    var y1("y1"), y2("y2"), d("d"), q("q");

    bx.split(y, 4000, y1, y2);
    by.split(y, 4000, y1, y2);

    bx.separate_at(0, one, -3);
    bx.get_update(0).rename_computation("bx_0");
    bx.get_update(1).rename_computation("bx_1");
    by.separate_at(0, one, -3);
    by.get_update(0).rename_computation("by_0");
    by.get_update(1).rename_computation("by_1");

    generator::replace_expr_name(by.get_update(0).expression, "bx", "bx_0");
    generator::replace_expr_name(by.get_update(1).expression, "bx", "bx_1");

    /*
     * Create communication
     */

    channel sync_block("sync_block", p_uint64, {FIFO, SYNC, BLOCK, MPI});
    channel async_nonblock("async_nonblock", p_uint64, {FIFO, ASYNC, NONBLOCK, MPI});
    channel async_block("async_block", p_uint64, {FIFO, ASYNC, BLOCK, MPI});
    send_recv fan_out = computation::create_transfer(
            "[Ny, Nx, one, twenty]->{fan_out_s[q,d,y,x]: 0<=q<one and one<=d<twenty and d*4000<=y<(d+1)*4000 and 0<=x<Nx+8}",
            "[Ny, Nx, one, twenty]->{fan_out_r[q,y,x]: one<=q<twenty and 0<=y<4000 and 0<=x<Nx+8}", 0, d, 0, q, async_nonblock,
            sync_block, blur_input(y, x), {&bx.get_update(1)}, &dblurxy);
    send_recv fan_in = computation::create_transfer(
            "[My, Mx, one, twenty]->{fan_in_s[q,y,x]: one<=q<twenty and 0<=y<My+8 and 0<=x<Mx}",
            "[My, Mx, one, twenty]->{fan_in_r[q,d,y,x]: 0<=q<one and one<=d<twenty and 0<=y<4000 and 0<=x<Mx}",
            q, 0, d, 0, sync_block,
            sync_block, by.get_update(1)(y, x), {}, &dblurxy);

    generator::replace_expr_name(bx.get_update(1).expression, "blur_input", "fan_out_r", true);

    tiramisu::send *fan_out_s = fan_out.s;
    tiramisu::recv *fan_out_r = fan_out.r;
    tiramisu::send *fan_in_s = fan_in.s;
    tiramisu::recv *fan_in_r = fan_in.r;

    tiramisu::wait wait_fan_in_s(fan_in_s->operator()(q, y, x), &dblurxy);
    tiramisu::wait wait_fan_out_s(fan_out_s->operator()(q, d, y, x), &dblurxy);

    /*
     * Collapsing
     */

    fan_out_s->collapse_many({collapser(3, 0, COLS)});
    fan_out_r->collapse_many({collapser(2, 0, COLS)});
    wait_fan_out_s.collapse_many({collapser(3, 0, COLS)});
    wait_fan_in_s.collapse_many({collapser(2, 0, by_cols)});
    fan_in.s->collapse_many({collapser(2, 0, by_cols)});
    fan_in.r->collapse_many({collapser(3, 0, by_cols)});

    /*
     * Ordering
     */

    fan_out_s->before(bx.get_update(0), computation::root);
    bx.get_update(0).before(by.get_update(0), computation::root);
    by.get_update(0).before(fan_out_r->get_update(0), computation::root);
    fan_out_r->get_update(0).before(bx.get_update(1), computation::root);
    bx.get_update(1).before(by.get_update(1), computation::root);
    by.get_update(1).before(*fan_in_s, computation::root);
    fan_in_s->before(wait_fan_out_s, computation::root);
    wait_fan_out_s.before(*fan_in_r, computation::root);
    fan_in_r->before(wait_fan_in_s, computation::root);

    /*
     * Tag distribute level
     */

    bx.get_update(0).tag_distribute_level(y1);
    bx.get_update(0).drop_rank_iter();
    bx.get_update(1).tag_distribute_level(y1);
    bx.get_update(1).drop_rank_iter();
    by.get_update(0).tag_distribute_level(y1);
    by.get_update(0).drop_rank_iter();
    by.get_update(1).tag_distribute_level(y1);
    by.get_update(1).drop_rank_iter();

    fan_out_s->tag_distribute_level(q);
    fan_out_r->tag_distribute_level(q);
    fan_in_s->tag_distribute_level(q);
    fan_in_r->tag_distribute_level(q);
    wait_fan_out_s.tag_distribute_level(q);
    wait_fan_in_s.tag_distribute_level(q);

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(ROWS), tiramisu::expr(COLS)},
                                p_uint64, tiramisu::a_input, &dblurxy);

    tiramisu::buffer buff_bx("buff_bx", {tiramisu::expr(by_rows + 2), tiramisu::expr(by_cols)},
                             p_uint64, tiramisu::a_output, &dblurxy);

    tiramisu::buffer buff_input_temp("buff_input_temp", {tiramisu::expr(ROWS), tiramisu::expr(COLS)},
                                     p_uint64, tiramisu::a_output, &dblurxy);

    tiramisu::buffer buff_bx_inter("buff_bx_inter", {tiramisu::expr(by_rows + 2), tiramisu::expr(by_cols)},
                                   p_uint64, tiramisu::a_output, &dblurxy);

    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(by_rows), tiramisu::expr(by_cols)},
                             p_uint64, tiramisu::a_output, &dblurxy);

    tiramisu::buffer buff_by_inter("buff_by_inter", {tiramisu::expr(by_rows), tiramisu::expr(by_cols)},
                                   p_uint64, tiramisu::a_output, &dblurxy);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");
    fan_out_r->get_update(0).set_access("{fan_out_r[q, y, x]->buff_input_temp[y, x]}");
    fan_in_r->set_access("{fan_in_r[q,d,y,x]->buff_by[d*4000+y, x]}");
    bx.get_update(0).set_access("{bx_0[y, x]->buff_bx[y, x]}");
    bx.get_update(1).set_access("{bx_1[y, x]->buff_bx_inter[y, x]}");
    by.get_update(0).set_access("{by_0[y, x]->buff_by[y, x]}");
    by.get_update(1).set_access("{by_1[y, x]->buff_by_inter[y, x]}");

    buffer fan_out_req_s_buff("fan_out_req_s_buff", {19,4000}, tiramisu::p_req_ptr, a_temporary, &dblurxy);
    fan_out_s->set_req_access("{fan_out_s[q,d,y,x]->fan_out_req_s_buff[d,y]}");
    buffer fan_in_req_s_buff("fan_in_req_s_buff", {4000}, tiramisu::p_req_ptr, a_temporary, &dblurxy);
    fan_in_s->set_req_access("{fan_in_s[q,y,x]->fan_in_req_s_buff[y]}");

    dblurxy.set_arguments({&buff_input, &buff_bx, &buff_input_temp, &buff_bx_inter, &buff_by_inter, &buff_by});
    // Generate code
    dblurxy.gen_time_space_domain();
    dblurxy.lift_ops_to_library_calls();
    dblurxy.gen_isl_ast();
    dblurxy.gen_halide_stmt();
    dblurxy.gen_halide_obj("./build/generated_dblurxy.o");

    // Some debugging
    dblurxy.dump_halide_stmt();

    return 0;

}