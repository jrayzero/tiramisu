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

//#define USE_DIST

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
#endif

    tester.gen_time_space_domain();
    tester.lift_ops_to_library_calls();
    tester.gen_isl_ast();
    tester.gen_halide_stmt();
    tester.dump_halide_stmt();
    tester.gen_halide_obj("./build/generated_fct_dtest_03.o");

    return 0;
}

