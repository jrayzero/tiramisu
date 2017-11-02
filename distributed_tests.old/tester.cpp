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

#define TTYPE p_int32
#define CTYPE int
#define LOOP_TYPE int

#define _X_MIN (LOOP_TYPE)0
#define _X_MAX (LOOP_TYPE)5
#define _Y_MIN (LOOP_TYPE)0
#define _Y_MAX (LOOP_TYPE)1000
#define _Z_MIN (LOOP_TYPE)0
#define _Z_MAX (LOOP_TYPE)2000
#define X_DELTA abs(_X_MAX - _X_MIN)
#define Y_DELTA abs(_Y_MAX - _Y_MIN)
#define Z_DELTA abs(_Z_MAX - _Z_MIN)

#define USE_DIST

int main(int argc, char **argv)
{
    global::set_default_tiramisu_options();

    tiramisu::function tester("tester");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::constant X_MIN("X_MIN", tiramisu::expr(_X_MIN), tiramisu::TTYPE, true, NULL, 0, &tester);
    tiramisu::constant X_MAX("X_MAX", tiramisu::expr(_X_MAX), tiramisu::TTYPE, true, NULL, 0, &tester);
    tiramisu::constant Y_MIN("Y_MIN", tiramisu::expr(_Y_MIN), tiramisu::TTYPE, true, NULL, 0, &tester);
    tiramisu::constant Y_MAX("Y_MAX", tiramisu::expr(_Y_MAX), tiramisu::TTYPE, true, NULL, 0, &tester);
    tiramisu::constant Z_MIN("Z_MIN", tiramisu::expr(_Z_MIN), tiramisu::TTYPE, true, NULL, 0, &tester);
    tiramisu::constant Z_MAX("Z_MAX", tiramisu::expr(_Z_MAX), tiramisu::TTYPE, true, NULL, 0, &tester);

    tiramisu::var x("x"), y("y"), z("z");

    tiramisu::computation input("[X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX]->{input[x, y, z]: X_MIN<=x<X_MAX and Y_MIN<=y<Y_MAX and Z_MIN<=z<Z_MAX}",
                                expr(), false, tiramisu::TTYPE, &tester);
    tiramisu::computation c0("[X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX]->{c0[x, y, z]: X_MIN<=x<X_MAX and Y_MIN<=y<Y_MAX and Z_MIN<=z<Z_MAX}",
                             input(x,y,z) * 10, true, tiramisu::TTYPE, &tester);
    tiramisu::computation c1("[X_MIN, X_MAX, Y_MIN, Y_MAX]->{c1[x, y]: X_MIN<=x<X_MAX and Y_MIN<=y<Y_MAX}",
                             c0(x,y,y+y) * 11, true, tiramisu::TTYPE, &tester);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    /*
     * NON-DISTRIBUTED
     */

#ifndef USE_DIST
    c0.before(c1, computation::root);
#endif

    /*
     * DISTRIBUTED
     */

#ifdef USE_DIST

    tiramisu::var h("h");

    // Split up
    tiramisu::constant one("one", tiramisu::expr(1), tiramisu::p_int32, true, NULL, 0, &tester); // Need these to prevent the code generator from removing that loop
    c0.separate_at(0, one, -3);
    c0.get_update(0).rename_computation("c0_0");
    c0.get_update(1).rename_computation("c0_1");
    c1.separate_at(0, one, -3);
    c1.get_update(0).rename_computation("c1_0");
    c1.get_update(1).rename_computation("c1_1");

    // Create the communication
    channel sync_block("sync_block", TTYPE, {FIFO, SYNC, BLOCK, MPI});
    channel async_nonblock("sync_block", TTYPE, {FIFO, ASYNC, NONBLOCK, MPI});

    send_recv fan_out = computation::create_transfer("[X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, one]->{fan_out_s[h,x,y,z]: 0<=h<one and one<=x<X_MAX and Y_MIN<=y<Y_MAX and Z_MIN<=z<Z_MAX}",
                                                     "[X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, one]->{fan_out_r[x,y,z]: one<=x<X_MAX and Y_MIN<=y<Y_MAX and Z_MIN<=z<Z_MAX}}",
                                                     0, x, sync_block, sync_block, input(x, y, z), {&c0.get_update(1)},
                                                     &tester);
    tiramisu::send *fan_out_s = fan_out.s;
    tiramisu::recv *fan_out_r = fan_out.r;

    // Collapsing
    fan_out_s->collapse_many({collapser(3, _Z_MIN, _Z_MAX)});
    fan_out_r->collapse_many({collapser(2, _Z_MIN, _Z_MAX)});

    // Transformations
    fan_out_s->interchange(x, y);
    fan_out_s->interchange(x, z);

    // Order things
    fan_out_s->before(c0.get_update(0), y);//computation::root);
    c0.get_update(0).before(c1.get_update(0), computation::root);
    c1.get_update(0).before(*fan_out_r, computation::root); // This is a little odd because they are different ranks, so we shouldn't have to schedule them relative to eachother
    fan_out_r->before(c0.get_update(1), computation::root);
    c0.get_update(1).before(c1.get_update(1), y);

    // Tag as being distributed
    c0.get_update(0).tag_distribute_level(x);
    c0.get_update(1).tag_distribute_level(x);
    c1.get_update(0).tag_distribute_level(x);
    c1.get_update(1).tag_distribute_level(x);
    fan_out_s->tag_distribute_level(h);
    fan_out_r->tag_distribute_level(x);

    // Need to update the computations to access the separate stuff now
    generator::replace_expr_name(c1.get_update(0).expression, "c0", "c0_0");
    generator::replace_expr_name(c1.get_update(1).expression, "c0", "c0_1");
    generator::replace_expr_name(c0.get_update(0).expression, "input", "fan_out_r");

#endif

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    /*
     * NON-DISTRIBUTED
     */

#ifndef USE_DIST
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(X_DELTA), tiramisu::expr(Y_DELTA), tiramisu::expr(Z_DELTA)},
                                tiramisu::TTYPE, tiramisu::a_input, &tester);
    tiramisu::buffer buff_c0("buff_c0", {tiramisu::expr(X_DELTA), tiramisu::expr(Y_DELTA), tiramisu::expr(Z_DELTA)},
                             tiramisu::TTYPE, tiramisu::a_temporary, &tester);
    tiramisu::buffer buff_c1("buff_c1", {tiramisu::expr(X_DELTA), tiramisu::expr(Y_DELTA)},
                             tiramisu::TTYPE, tiramisu::a_output, &tester);

    input.set_access("{input[x,y,z]->buff_input[x,y,z]}");
    c0.set_access("{c0[x,y,z]->buff_c0[x,y,z]}");
    c1.set_access("{c1[x,y]->buff_c1[x,y]}");
    tester.set_arguments({&buff_input, &buff_c1});
#endif

    /*
     * DISTRIBUTED
     */

#ifdef USE_DIST
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(X_DELTA), tiramisu::expr(Y_DELTA), tiramisu::expr(Z_DELTA)},
                                tiramisu::TTYPE, tiramisu::a_input, &tester);
    tiramisu::buffer buff_c0_0("buff_c0_0", {1, tiramisu::expr(Y_DELTA), tiramisu::expr(Z_DELTA)},
                               tiramisu::TTYPE, tiramisu::a_temporary, &tester);
    tiramisu::buffer buff_c1_0("buff_c1_0", {tiramisu::expr(X_DELTA), tiramisu::expr(Y_DELTA)},
                               tiramisu::TTYPE, tiramisu::a_output, &tester);

    tiramisu::buffer buff_input_temp("buff_input_temp", {tiramisu::expr(Y_DELTA), tiramisu::expr(Z_DELTA)},
                                     tiramisu::TTYPE, tiramisu::a_temporary, &tester);
    tiramisu::buffer buff_c0_1("buff_c0_1", {tiramisu::expr(Y_DELTA), tiramisu::expr(Z_DELTA)},
                               tiramisu::TTYPE, tiramisu::a_temporary, &tester);
    tiramisu::buffer buff_c1_1("buff_c1_1", {tiramisu::expr(Y_DELTA)},
                               tiramisu::TTYPE, tiramisu::a_temporary, &tester);

    input.set_access("{input[x,y,z]->buff_input[x,y,z]}");
    c0.get_update(0).set_access("{c0_0[x,y,z]->buff_c0_0[x,y,z]}");
    c1.get_update(0).set_access("{c1_0[x,y]->buff_c1_0[x,y]}");
    c0.get_update(1).set_access("{c0_1[x,y,z]->buff_c0_1[y,z]}");
    c1.get_update(1).set_access("{c1_1[x,y]->buff_c1_1[y]}");
    fan_out_r->set_access("{fan_out_r[x,y,z]->buff_input_temp[y,z]}");
    tester.set_arguments({&buff_input, &buff_c1_0});
#endif

    tester.gen_time_space_domain();
    tester.lift_ops_to_library_calls();
    tester.gen_isl_ast();
    tester.gen_halide_stmt();
    tester.dump_halide_stmt();
    tester.gen_halide_obj("./build/generated_fct_dtest_03.o");

    return 0;
}

