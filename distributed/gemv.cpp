//
// Created by Jessica Ray on 1/17/18.
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

#include "gemv_params.h"

using namespace tiramisu;

var c("c"), r("r");

std::vector<computation *> make_algorithm(function *f, int64_t rows, int64_t cols) {

    computation *vector = new computation("{vector[c]: 0<=c<" + std::to_string(cols) + "}", expr(), false, p_float32, f);
    computation *matrix = new computation("{matrix[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                          expr(), false, p_float32, f);
    // Initializes the reduction
    computation *gemv_dummy = new computation("{gemv_dummy[r]: 0<=r<" + std::to_string(rows) + "}", expr(0.0f),
                                              true, p_float32, f);
    // Does the reduction
    computation *gemv = new computation("{gemv[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                        expr(matrix->operator()(r,c) * vector->operator()(c) + gemv_dummy->operator()(r)),
                                        true, p_float32, f);
    return {vector, matrix, gemv_dummy, gemv};
}

void postprocess(function *f, std::string obj_file_name) {
    f->gen_time_space_domain();
    f->gen_isl_ast();
    f->gen_halide_stmt();
    f->gen_halide_obj(obj_file_name);
    f->dump_halide_stmt();
}

void create_cpu_version() {
    function *gemv_cpu = new function("gemv_cpu");

    // SINGLE CPU VERSION
    std::vector<computation *> comps = make_algorithm(gemv_cpu, ROWS, COLS);
    computation *vector = comps[0];
    computation *matrix = comps[1];
    computation *gemv_dummy = comps[2];
    computation *gemv = comps[3];

#ifdef CPU_OPTS
    var r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1");
    gemv_dummy->split(r, ROWS/20, r0, r1);
    gemv_dummy->parallelize(r0);
    gemv->tile(r, c, 100, 100, r0, c0, r1, c1);
    gemv->split(r0, ROWS/100/20, r2, r3);
    gemv->parallelize(r2);
#endif

    gemv_dummy->before(*gemv, r2);

    buffer vector_buff("vector_buff", {COLS}, p_float32, a_input, gemv_cpu);
    buffer matrix_buff("matrix_buff", {ROWS, COLS}, p_float32, a_input, gemv_cpu);
    buffer result_buff("result_buff", {ROWS}, p_float32, a_output, gemv_cpu);

    vector->set_access("{vector[c]->vector_buff[c]}");
    matrix->set_access("{matrix[r,c]->matrix_buff[r,c]}");
    gemv_dummy->set_access("{gemv_dummy[r]->result_buff[r]}");
    gemv->set_access("{gemv[r,c]->result_buff[r]}");

    gemv_cpu->set_arguments({&vector_buff, &matrix_buff, &result_buff});
    postprocess(gemv_cpu, "/tmp/generated_gemv_cpu.o");
}

int main() {

    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int64);


//    function gemv_cpu_parallel("gemv_cpu_parallel");
//    function gemv_gpu("gemv_gpu");
//    function gemv_coop("gemv_coop");

    create_cpu_version();

    return 0;

}