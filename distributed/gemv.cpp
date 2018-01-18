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

std::vector<computation *> make_algorithm(function *f, int64_t rows, int64_t cols) {
  var c("c"), r("r");
    computation *vector = new computation("{vector[r,c]: 0<=r<1 and 0<=c<" + std::to_string(cols) + "}", expr(), false, p_float32, f);
    computation *matrix = new computation("{matrix[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                          expr(), false, p_float32, f);
    // Initializes the reduction
    computation *gemv_dummy = new computation("{gemv_dummy[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<1}", expr(0.0f),
                                              true, p_float32, f);
    // Does the reduction
    computation *gemv = new computation("{gemv[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                        expr(matrix->operator()(r,c) * vector->operator()(0,c) + gemv_dummy->operator()(r,0)),
                                        true, p_float32, f);
    return {vector, matrix, gemv_dummy, gemv};
}

void postprocess(function *f, std::string obj_file_name) {
    f->lift_dist_comps();
    f->gen_time_space_domain();
    f->gen_isl_ast();
    f->gen_halide_stmt();
    f->gen_halide_obj(obj_file_name);
    f->dump_halide_stmt();
}

void create_cpu_version() {
    var c("c"), r("r"), r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1");    
    function *gemv_cpu = new function("gemv_cpu");

    std::vector<computation *> comps = make_algorithm(gemv_cpu, ROWS, COLS);
    computation *vector = comps[0];
    computation *matrix = comps[1];
    computation *gemv_dummy = comps[2];
    computation *gemv = comps[3];

#ifdef CPU_OPTS
    gemv_dummy->split(r, ROWS/(int64_t)20, r0, r1);
    gemv_dummy->parallelize(r0);
    gemv->tile(r, c, (int64_t)100, (int64_t)100, r0, c0, r1, c1);
    gemv->split(r0, ROWS/(int64_t)100/(int64_t)20, r2, r3);
    gemv->parallelize(r2);
#endif

    gemv_dummy->before(*gemv, r2);

    buffer vector_buff("vector_buff", {(int64_t)1, COLS}, p_float32, a_input, gemv_cpu);
    buffer matrix_buff("matrix_buff", {ROWS, COLS}, p_float32, a_input, gemv_cpu);
    buffer result_buff("result_buff", {ROWS, (int64_t)1}, p_float32, a_output, gemv_cpu);

    vector->set_access("{vector[r,c]->vector_buff[r,c]}");
    matrix->set_access("{matrix[r,c]->matrix_buff[r,c]}");
    gemv_dummy->set_access("{gemv_dummy[r,c]->result_buff[r,c]}");
    gemv->set_access("{gemv[r,c]->result_buff[r,0]}");

    gemv_cpu->set_arguments({&vector_buff, &matrix_buff, &result_buff});
    postprocess(gemv_cpu, "/tmp/generated_gemv.o");
}

void create_gpu_version() {
    var c("c"), r("r"), r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1");
    function *gemv_gpu = new function("gemv_gpu");
    std::vector<computation *> comps = make_algorithm(gemv_gpu, ROWS, COLS);
    computation *vector = comps[0];
    computation *matrix = comps[1];
    computation *gemv_dummy = comps[2];
    computation *gemv = comps[3];

    int64_t rows_resident_on_gpu = 200;
    int64_t threads_per_block = 1000;

    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);
    xfer_prop h2d_cuda_sync(p_float32, {SYNC, CUDA, CPU2GPU}, 0);
    xfer_prop d2h_cuda_sync(p_float32, {SYNC, CUDA, GPU2CPU}, 0);
    xfer vector_copy = computation::create_xfer("{vector_copy[r,c]: 0<=r<1 and 0<=c<" + std::to_string(COLS) + "}", h2d_cuda_async,
                                                vector->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "vector", "vector_copy", false);
    xfer matrix_row_copy = computation::create_xfer("{matrix_row_copy[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<" +
                                                    std::to_string(COLS) + "}", h2d_cuda_async,
                                                    matrix->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "matrix", "matrix_row_copy", false);
    xfer init_reduction = computation::create_xfer("{init_reduction[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", h2d_cuda_async, gemv_dummy->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "gemv_dummy", "init_reduction", false);

    xfer copy_back_results = computation::create_xfer("{copy_back[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", d2h_cuda_async, gemv->operator()(r,c), gemv_gpu);

    //    gemv->split(c, threads_per_block, c0, c1);

    matrix_row_copy.os->split(r, rows_resident_on_gpu, r0, r1);
    gemv->split(r, rows_resident_on_gpu, r0, r1);
    gemv_dummy->split(r, rows_resident_on_gpu, r0, r1);
    init_reduction.os->split(r, rows_resident_on_gpu, r0, r1);
    copy_back_results.os->split(r, rows_resident_on_gpu, r0, r1);

    vector_copy.os->before(*matrix_row_copy.os, computation::root);
    matrix_row_copy.os->before(*gemv_dummy, r1);
    gemv_dummy->before(*init_reduction.os, r1);
    init_reduction.os->before(*gemv, r1);
    gemv->before(*copy_back_results.os, r1);

    vector_copy.os->collapse_many({collapser(1, (int64_t)0, COLS)});
    matrix_row_copy.os->collapse_many({collapser(2, (int64_t)0, COLS)});

    // this has to go after all the other things have been scheduled
    //    gemv->tag_gpu_level2(c0, c1, 0);

    buffer vector_buff("vector_buff", {1,COLS}, p_float32, a_input, gemv_gpu);
    buffer matrix_buff("matrix_buff", {ROWS, COLS}, p_float32, a_input, gemv_gpu);
    buffer result_buff("result_buff", {ROWS,1}, p_float32, a_output, gemv_gpu);
    buffer zero_buff("zero_buff", {1,1}, p_float32, a_temporary, gemv_gpu);
    buffer vector_gpu_buff("vector_gpu_buff", {1,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer matrix_gpu_buff("matrix_gpu_buff", {rows_resident_on_gpu,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // copy up one row at a time
    buffer result_gpu_buff("result_gpu_buff", {rows_resident_on_gpu,1}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer buff_bx_literals("buff_gemv_literals", {ROWS, 3}, p_int64, tiramisu::a_temporary_gpu, gemv_gpu);
    buffer null_buffer("null_buffer", {1}, p_wait_ptr, tiramisu::a_temporary, gemv_gpu);

    vector->set_access("{vector[r,c]->vector_buff[r,c]}");
    vector_copy.os->set_access("{vector_copy[r,c]->vector_gpu_buff[r,c]}");
    vector_copy.os->set_wait_access("{vector_copy[r,c]->null_buffer[0]}");
    matrix->set_access("{matrix[r,c]->matrix_buff[r,c]}");
    matrix_row_copy.os->set_access("{matrix_row_copy[r,c]->matrix_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    matrix_row_copy.os->set_wait_access("{matrix_row_copy[r,c]->null_buffer[0]}");
    gemv_dummy->set_access("{gemv_dummy[r,c]->zero_buff[0,0]}");
    init_reduction.os->set_access("{init_reduction[r,c]->result_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    init_reduction.os->set_wait_access("{init_reduction[r,c]->null_buffer[0]}");
    gemv->set_access("{gemv[r,c]->result_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",0]}");
    copy_back_results.os->set_access("{copy_back[r,c]->result_buff[r,c]}");    
    copy_back_results.os->set_wait_access("{copy_back[r,c]->null_buffer[0]}");    

    gemv_gpu->set_arguments({&vector_buff, &matrix_buff, &result_buff});
    postprocess(gemv_gpu, "/tmp/generated_gemv.o");
    print_tiramisu_cuda_runtime();
    compile_kernels_to_obj();
}

int main() {

    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int64);

#ifdef CPU
    create_cpu_version();
#elif defined(GPU)
    create_gpu_version();
#endif

    return 0;

}
