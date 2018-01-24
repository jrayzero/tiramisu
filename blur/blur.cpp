#include <tiramisu/core.h>
#include "blur.h"

using namespace tiramisu;

std::vector<computation *> algorithm(function *f, std::string srows, std::string scols) {
    var r(p_int64, "r");
    var c(p_int64, "c");
    computation *blur_input = new computation("{blur_input[r,c]: 0<=r<" + srows + " and 0<=c<" +
                                              scols + "}", expr(), false, p_float32, f);
    expr bx_expr = (blur_input->operator()(var("r"), var("c")) +
                    blur_input->operator()(var("r"), var("c") + (int64_t)1) +
                    blur_input->operator()(var("r"), var("c") + (int64_t)2)) / 3.0f;
    computation *bx = new computation("{bx[r,c]: 0<=r<" + srows + " and 0<=c<" + scols + "}",
                                      bx_expr, true, p_float32, f);
    expr by_expr = (bx->operator()(var("r"), var("c")) +
                    bx->operator()(var("r") + (int64_t)1, var("c")) +
                    bx->operator()(var("r") + (int64_t)2, var("c"))) / 3.0f;
    computation *by = new computation("{by[r,c]: 0<=r<" + srows + " and 0<=c<" + scols + "}",
                                      by_expr, true, p_float32, f);
    return {blur_input, bx, by};
}

std::vector<computation *> big_box_algorithm(function *f, std::string srows, std::string scols) {
    var r(p_int64, "r");
    var c(p_int64, "c");
    computation *blur_input = new computation("{blur_input[r,c]: 0<=r<" + srows + " and 0<=c<" +
                                              scols + "}", expr(), false, p_float32, f);
    expr bx_expr = 0.0f;
    for (int64_t i = 0; i < 10; i++) {
        bx_expr = bx_expr + (blur_input->operator()(var("r"), var("c") + i));
    }
    bx_expr = bx_expr / 10.0f;
    computation *bx = new computation("{bx[r,c]: 0<=r<" + srows + " and 0<=c<" + scols + "-10}",
                                      bx_expr, true, p_float32, f);
    expr by_expr = 0.0f;
    for (int64_t i = 0; i < 10; i++) {
        by_expr = by_expr + (bx->operator()(var("r") + i, var("c")));
    }
    by_expr = by_expr / 10.0f;
    computation *by = new computation("{by[r,c]: 0<=r<" + srows + "-10 and 0<=c<" + scols + "-10}",
                                      by_expr, true, p_float32, f);
    return {blur_input, bx, by};
}

//void generate_cooperative_multi() {
//    std::string srows = std::to_string(ROWS);
//    std::string scols = std::to_string(COLS);
//    std::string sresident = std::to_string(RESIDENT);
//    // number of rows to go on the GPU
//    int64_t gpu_rows = (ROWS / 10) * GPU_PERCENTAGE;
//    // number of rows to go on the CPU
//    int64_t cpu_rows = ROWS - gpu_rows;
//    int64_t gpu_rows_per_proc = gpu_rows / (PROCS/2);
//    int64_t cpu_rows_per_proc = cpu_rows / (PROCS/2); // cause each node and each gpu get one proc
//
//    function blur("blur_cooperative");
//    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1"), r2(p_int64, "r2"), r3(p_int64, "r3"), r4(p_int64, "r4"), r5(p_int64, "r5");
//    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1"), c2(p_int64, "c2"), c3(p_int64, "c3");
//    var q(p_int64, "q");
//
//    std::vector<computation *> comps = algorithm(&blur, srows, scols);
//    computation *blur_input = comps[0];
//    computation *bx = comps[1];
//    computation *by = comps[2];
//
//    // First, separate our entire computation into two sets of rows: cpu rows and gpu rows
//    bx->separate_at(r, {cpu_rows}, ROWS); // the first set of rows for the gpu
//    by->separate_at(r, {cpu_rows}, ROWS);
//
//    generator::update_producer_expr_name(&(by->get_update(0)), "bx", "bx_0", false);
//    generator::update_producer_expr_name(&(by->get_update(1)), "bx", "bx_1", false);
//
//    // rename the dimensions cause separate at changes them
//    bx->get_update(0).schedule = isl_map_set_dim_name(bx->get_update(0).get_schedule(), isl_dim_in, 0, "r");
//    by->get_update(0).schedule = isl_map_set_dim_name(by->get_update(0).get_schedule(), isl_dim_in, 0, "r");
//    bx->get_update(0).schedule = isl_map_set_dim_name(bx->get_update(0).get_schedule(), isl_dim_in, 1, "c");
//    by->get_update(0).schedule = isl_map_set_dim_name(by->get_update(0).get_schedule(), isl_dim_in, 1, "c");
//
//    bx->get_update(1).schedule = isl_map_set_dim_name(bx->get_update(1).get_schedule(), isl_dim_in, 0, "r");
//    by->get_update(1).schedule = isl_map_set_dim_name(by->get_update(1).get_schedule(), isl_dim_in, 0, "r");
//    bx->get_update(1).schedule = isl_map_set_dim_name(bx->get_update(1).get_schedule(), isl_dim_in, 1, "c");
//    by->get_update(1).schedule = isl_map_set_dim_name(by->get_update(1).get_schedule(), isl_dim_in, 1, "c");
//
////    expr border_expr = (blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r,c) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r,c+(int64_t)1) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r,c+(int64_t)2) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r+(int64_t)1,c) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r+(int64_t)1,c+(int64_t)1) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r+(int64_t)1,c+(int64_t)2) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r+(int64_t)2,c) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r+(int64_t)2,c+(int64_t)1) +
////                        blur_input->operator()((int64_t)(ROWS/PROCS)-(int64_t)2+r+(int64_t)2,c+(int64_t)2)) / 9.0f;
////    computation border("{border[q,r,c]: 0<=q<" + std::to_string(PROCS) + " and 0<=r<2 and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);
//
//
//    bx->get_update(0).split(r, gpu_rows_per_proc, r1, r2);
//    bx->get_update(1).split(r, cpu_rows_per_proc, r1, r2);
//    by->get_update(0).split(r, gpu_rows_per_proc, r1, r2);
//    by->get_update(1).split(r, cpu_rows_per_proc, r1, r2);
//
//    bx->get_update(0).tag_distribute_level(r1, true);
//    bx->get_update(1).tag_distribute_level(r1, true);
//    by->get_update(0).tag_distribute_level(r1, true);
//    by->get_update(1).tag_distribute_level(r1, true);
////    border.tag_distribute_level(q);
//
//    bx->get_update(0).set_schedule_this_comp(false);
//    by->get_update(0).set_schedule_this_comp(false);
//
////    bx->get_update(1).tile(r2, c, (int64_t)100, (int64_t)1000, r0, c0, r5, c1);
////    by->get_update(1).tile(r2, c, (int64_t)100, (int64_t)1000, r0, c0, r5, c1);
////    bx->get_update(1).split(r0, 8, r3, r4);
////    by->get_update(1).split(r0, 8, r3, r4);
////    bx->get_update(1).split(c1, 8, c2, c3);
////    by->get_update(1).split(c1, 8, c2, c3);
////    bx->get_update(1).tag_vector_level(c3, 8);
////    by->get_update(1).tag_vector_level(c3, 8);
////    bx->get_update(1).tag_parallel_level(r3);
////    by->get_update(1).tag_parallel_level(r3);
//
//    bx->get_update(1).before(by->get_update(1), computation::root);
////    by->get_update(1).before(border, computation::root);
//
//    tiramisu::buffer buff_input("buff_input", {ROWS/PROCS+(int64_t)2, COLS+(int64_t)2}, p_float32,
//                                tiramisu::a_input, &blur);
//
//    tiramisu::buffer buff_bx("buff_bx", {ROWS/PROCS, COLS},
//                             p_float32, tiramisu::a_output, &blur);
//
//    tiramisu::buffer buff_by("buff_by", {ROWS/PROCS, COLS},
//                             p_float32, tiramisu::a_output, &blur);
//
//    blur_input->set_access("{blur_input[i1, i0]->buff_input[i1,i0]}");
//    by->get_update(1).set_access("{by_1[y, x]->buff_by[y,x]}");
//    bx->get_update(1).set_access("{bx_1[y, x]->buff_bx[y, x]}");
////    border.set_access("{border[q,r,c]->buff_by[r+" + std::to_string(ROWS/PROCS) + "-2,c]}");
//
//    tiramisu::buffer null_buff("null", {ROWS/PROCS+(int64_t)2, COLS+(int64_t)2}, p_float32,
//                                tiramisu::a_temporary, &blur);
//
//    bx->get_update(0).set_access("{bx_0[r,c]->null[0]}");
//    by->get_update(0).set_access("{by_0[r,c]->null[0]}");
//
//
//    blur.set_arguments({&buff_input, &buff_bx, &buff_by});
//    blur.lift_dist_comps();
//    blur.gen_time_space_domain();
//    blur.gen_isl_ast();
//    blur.gen_halide_stmt();
//    blur.gen_halide_obj("/tmp/generated_coop_blur.o");
//    blur.dump_halide_stmt();
//    compile_kernels_to_obj();
//}

void generate_single_cpu() {
    std::string srows = std::to_string(ROWS);
    std::string scols = std::to_string(COLS);
    function blur("blur_single_cpu");
    std::vector<computation *> comps = algorithm(&blur, srows, scols);
    computation *blur_input = comps[0];
    computation *bx = comps[1];
    computation *by = comps[2];
    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1"), r2(p_int64, "r2"), r3(p_int64, "r3"), r4(p_int64, "r4"), r5(p_int64, "r5");
    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1"), c2(p_int64, "c2"), c3(p_int64, "c3");
    var z(p_int64, "z");

    bx->tile(r, c, (int64_t)100, (int64_t)1000, r0, c0, r1, c1);
    by->tile(r, c, (int64_t)100, (int64_t)1000, r0, c0, r1, c1);
    bx->split(r0, 8, r2, r3);
    by->split(r0, 8, r2, r3);
    bx->split(c1, 8, c2, c3);
    by->split(c1, 8, c2, c3);
    bx->tag_vector_level(c3, 8);
    by->tag_vector_level(c3, 8);
    bx->tag_parallel_level(r2);
    by->tag_parallel_level(r2);

    /*  bx->split(r, 32, r0, r1);
    bx->split(r0, 25, r2, r3);
    bx->split(c, 8, c0, c1);
    bx->tag_parallel_level(r2);
    bx->split(c0, 100, c2, c3);
    bx->tag_vector_level(c1, 8);

    by->split(r, 32, r0, r1);
    by->split(r0, 25, r2, r3);
    by->split(c, 8, c0, c1);
    by->split(c0, 100, c2, c3);
    by->tag_parallel_level(r2);
    by->tag_vector_level(c1, 8);*/

    //  bx->store_at(*by, r);
    //  bx->compute_at(*by, r1);

    bx->before(*by, r3);//computation::root);

    tiramisu::buffer buff_input("buff_input", {ROWS+(int64_t)2, COLS+(int64_t)2}, p_float32,
                                tiramisu::a_input, &blur);

    tiramisu::buffer buff_bx("buff_bx", {ROWS, COLS},
                             p_float32, tiramisu::a_output, &blur);

    tiramisu::buffer buff_by("buff_by", {ROWS, COLS},
                             p_float32, tiramisu::a_output, &blur);

    blur_input->set_access("{blur_input[i1, i0]->buff_input[i1,i0]}");
    by->set_access("{by[y, x]->buff_by[y,x]}");
    bx->set_access("{bx[y, x]->buff_bx[y, x]}");
    blur.lift_dist_comps();
    blur.set_arguments({&buff_input, &buff_bx, &buff_by});
    blur.gen_time_space_domain();
    blur.gen_isl_ast();
    blur.gen_halide_stmt();
    blur.gen_halide_obj("/tmp/generated_single_cpu_blur.o");

}

void generate_single_cpu_big_box() {
    std::string srows = std::to_string(ROWS);
    std::string scols = std::to_string(COLS);
    function blur("blur_single_cpu");
    std::vector<computation *> comps = big_box_algorithm(&blur, srows, scols);
    computation *blur_input = comps[0];
    computation *bx = comps[1];
    computation *by = comps[2];
    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1"), r2(p_int64, "r2"), r3(p_int64, "r3"), r4(p_int64, "r4"), r5(p_int64, "r5");
    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1"), c2(p_int64, "c2"), c3(p_int64, "c3");
    var z(p_int64, "z");

    bx->tile(r, c, (int64_t)100, (int64_t)1000, r0, c0, r1, c1);
    by->tile(r, c, (int64_t)100, (int64_t)1000, r0, c0, r1, c1);
    bx->split(r0, 8, r2, r3);
    by->split(r0, 8, r2, r3);
    bx->split(c1, 8, c2, c3);
    by->split(c1, 8, c2, c3);
    bx->tag_vector_level(c3, 8);
    by->tag_vector_level(c3, 8);
    bx->tag_parallel_level(r2);
    by->tag_parallel_level(r2);

    /*  bx->split(r, 32, r0, r1);
    bx->split(r0, 25, r2, r3);
    bx->split(c, 8, c0, c1);
    bx->tag_parallel_level(r2);
    bx->split(c0, 100, c2, c3);
    bx->tag_vector_level(c1, 8);

    by->split(r, 32, r0, r1);
    by->split(r0, 25, r2, r3);
    by->split(c, 8, c0, c1);
    by->split(c0, 100, c2, c3);
    by->tag_parallel_level(r2);
    by->tag_vector_level(c1, 8);*/

    //  bx->store_at(*by, r);
    //  bx->compute_at(*by, r1);

    bx->before(*by, r3);//computation::root);

    tiramisu::buffer buff_input("buff_input", {ROWS+(int64_t)2, COLS+(int64_t)2}, p_float32,
                                tiramisu::a_input, &blur);

    tiramisu::buffer buff_bx("buff_bx", {ROWS, COLS},
                             p_float32, tiramisu::a_output, &blur);

    tiramisu::buffer buff_by("buff_by", {ROWS, COLS},
                             p_float32, tiramisu::a_output, &blur);

    blur_input->set_access("{blur_input[i1, i0]->buff_input[i1,i0]}");
    by->set_access("{by[y, x]->buff_by[y,x]}");
    bx->set_access("{bx[y, x]->buff_bx[y, x]}");
    blur.lift_dist_comps();
    blur.set_arguments({&buff_input, &buff_bx, &buff_by});
    blur.gen_time_space_domain();
    blur.gen_isl_ast();
    blur.gen_halide_stmt();
    blur.gen_halide_obj("/tmp/generated_single_cpu_blur.o");

}


void generate_multi_cpu() {

    std::string srows = std::to_string(CPU_ROWS);
    std::string scols = std::to_string(COLS);
    function blur("blur_multi_cpu");
    std::vector<computation *> comps = algorithm(&blur, srows, scols);
    computation *blur_input = comps[0];
    computation *bx = comps[1];
    computation *by = comps[2];
    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1"), r2(p_int64, "r2"), r3(p_int64, "r3");
    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1");
    var z(p_int64, "z");

    var q("q"), q1("q1"), q2("q2");


    expr border_expr = (blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r,c) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r,c+(int64_t)1) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r,c+(int64_t)2) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r+(int64_t)1,c) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r+(int64_t)1,c+(int64_t)1) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r+(int64_t)1,c+(int64_t)2) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r+(int64_t)2,c) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r+(int64_t)2,c+(int64_t)1) +
                        blur_input->operator()((int64_t)(CPU_ROWS/CPU_PROCS)-(int64_t)2+r+(int64_t)2,c+(int64_t)2)) / 9.0f;
    computation border("{border[q,r,c]: 0<=q<" + std::to_string(CPU_PROCS) + " and 0<=r<2 and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);

    bx->split(r, CPU_ROWS/CPU_PROCS, q1, q2);
    by->split(r, CPU_ROWS/CPU_PROCS, q1, q2);

    var c2(p_int64, "c2"), c3(p_int64, "c3");

    bx->tile(q2, c, (int64_t)100, (int64_t)1000, r0, c0, r1, c1);
    by->tile(q2, c, (int64_t)100, (int64_t)1000, r0, c0, r1, c1);
    bx->split(r0, 8, r2, r3);
    by->split(r0, 8, r2, r3);
    bx->split(c1, 8, c2, c3);
    by->split(c1, 8, c2, c3);
    bx->tag_vector_level(c3, 8);
    by->tag_vector_level(c3, 8);
//    bx->tag_parallel_level(r2);
//    by->tag_parallel_level(r2);

    bx->tag_distribute_level(q1);
    by->tag_distribute_level(q1);
    border.tag_distribute_level(q);


    bx->before(*by, computation::root);
    by->before(border, computation::root);

    tiramisu::buffer buff_input("buff_input", {CPU_ROWS/CPU_PROCS+(int64_t)2, COLS+(int64_t)2}, p_float32,
                                tiramisu::a_input, &blur);

    tiramisu::buffer buff_bx("buff_bx", {CPU_ROWS/CPU_PROCS, COLS},
                             p_float32, tiramisu::a_output, &blur);

    tiramisu::buffer buff_by("buff_by", {CPU_ROWS/CPU_PROCS, COLS},
                             p_float32, tiramisu::a_output, &blur);

    blur_input->set_access("{blur_input[i1, i0]->buff_input[i1,i0]}");
    by->set_access("{by[y, x]->buff_by[y,x]}");
    bx->set_access("{bx[y, x]->buff_bx[y, x]}");
    border.set_access("{border[q,r,c]->buff_by[r+" + std::to_string(CPU_ROWS/CPU_PROCS) + "-2,c]}");
    blur.lift_dist_comps();
    blur.set_arguments({&buff_input, &buff_bx, &buff_by});
    blur.gen_time_space_domain();
    blur.gen_isl_ast();
    blur.gen_halide_stmt();
    blur.gen_halide_obj("/tmp/generated_multi_cpu_blur.o");

}


//void generate_single_gpu_big_box() {
//    std::string srows = std::to_string(ROWS);
//    std::string scols = std::to_string(COLS);
//    std::string sresident = std::to_string(RESIDENT);
//    function blur("blur_single_gpu");
//    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1");
//    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1");
//    var z(p_int64, "z");
//
//    std::vector<computation *> comps = big_box_algorithm(&blur, srows, scols);
//    computation *blur_input = comps[0];
//    computation *bx = comps[1];
//    computation *by = comps[2];
//
//    // split up bx and by based on how much we can fit on the GPU at once
//    bx->split(r, RESIDENT, r0, r1);
//    by->split(r, RESIDENT, r0, r1);
//
//    // split up the columns for bx and by for GPU level tagging
//    bx->split(c, BLOCK_SIZE, c0, c1);
//    by->split(c, BLOCK_SIZE, c0, c1);
//
//    // transfer the input from cpu to gpu
//    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 1);
//    xfer_prop stream1(p_float32, {ASYNC, CUDA}, 1);
//    xfer_prop stream2(p_float32, {ASYNC, CUDA}, 2);
//    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 2);
//    xfer_prop d2h_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);
//    xfer_prop h2d_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
//    // copy up the last two rows later on
//    xfer h2d = computation::create_xfer("{h2d[r,c]: 0<=r<" + srows + " and 0<=c<2+" + scols +
//                                        "}", h2d_cuda_async, blur_input->operator()(r,c), &blur);
//    tiramisu::wait h2d_wait("{h2d_wait[r,c]: 0<=r<" + srows + " and 0<=c<1}", h2d.os->operator()(r,0), h2d_cuda_async_kernel_stream, true, &blur);
//    generator::update_producer_expr_name(bx, "blur_input", "h2d", false);
//    // split up the h2d transfer based on how much we can transfer at a time
//    h2d.os->split(r, RESIDENT, r0, r1);
//    h2d_wait.split(r, RESIDENT, r0, r1);
//    // collapse the h2d so it sends the whole row as a chunk
//    h2d.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});
//
//    // create a wait on d2h for the next iteration of bx. Also wait on the by kernel
//    tiramisu::wait d2h_wait_for_by("{d2h_wait_for_by[r,c]: 0<=r<" + srows + " and 0<=c<1}", by->operator()(r, 0), stream2, true, &blur);
//    xfer d2h = computation::create_xfer("{d2h[r,c]: 0<=r<" + srows + " and 0<=c<" + scols +
//                                        "}", d2h_cuda_async, by->operator()(r,0), &blur);
//    tiramisu::wait d2h_wait("{d2h_wait[r,c]: " + sresident + "<=r<" + srows + " and 0<=c<1}", d2h.os->operator()(r-RESIDENT,0), h2d_cuda_async, true, &blur);
//    d2h.os->split(r, RESIDENT, r0, r1);
//    d2h_wait.split(r, RESIDENT, r0, r1);
//    d2h_wait_for_by.split(r, RESIDENT, r0, r1);
//    d2h.os->collapse_many({collapser(2, (int64_t)0, COLS)});
//
//    // special transfer that copies up just the two extra rows needed to get the boundary region correct
//    // TODO JESS this access to blur input needs to be fixed!
//    xfer ghost = computation::create_xfer("{ghost[r,z,c]: 0<=r<(" + std::to_string(ROWS/RESIDENT) + ") and 0<=z<4 and 0<=c<2+" + scols +
//                                          "}", h2d_cuda_async_kernel_stream, blur_input->operator()(/*(r+(int64_t)1)*(ROWS/RESIDENT)+z*/r*RESIDENT+(int64_t)RESIDENT+z-(int64_t)2,c), &blur);
//    //  ghost.os->split(r, 2, r0, r1);
//    ghost.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});
//
//
//    expr border_expr = (ghost.os->operator()(0, z, c) + ghost.os->operator()(0, z, c+(int64_t)1) + ghost.os->operator()(0, z, c+(int64_t)2) +
//                        ghost.os->operator()(0, z+(int64_t)1, c) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)2) +
//                        ghost.os->operator()(0, z+(int64_t)2, c) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)2)) / 9.0f;
//    computation border("{border[r,z,c]: 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=z<2 and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);
//    border.split(c, BLOCK_SIZE, c0, c1);
//
//    xfer d2h_border = computation::create_xfer("{d2h_border[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=c<" + scols + "}", d2h_cuda_async_kernel_stream, border(r,z,c), &blur);
//    d2h_border.os->collapse_many({collapser(2, (int64_t)0, COLS)});
//    tiramisu::wait d2h_border_wait("{d2h_border_wait[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=c<1}", d2h_border.os->operator()(r,z,0), stream1, true, &blur);
//    d2h_border_wait.set_schedule_this_comp(false);
//    // order
//    d2h_wait.before(*h2d.os, r1);
//    h2d.os->before(h2d_wait, r1);
//    h2d_wait.before(*bx, r1);
//    bx->before(*by, r0);
//    by->before(d2h_wait_for_by, r1);
//    d2h_wait_for_by.before(*d2h.os, r1);
//    d2h.os->before(*ghost.os, r);
//    ghost.os->before(border, r);
//    border.before(*d2h_border.os, z);
//    d2h_border.os->before(d2h_border_wait, z);
//
//    // tag for the GPU
//    bx->tag_gpu_level2(c0, c1, 0);
//    by->tag_gpu_level2(c0, c1, 0);
//    border.tag_gpu_level2(c0, c1, 0);
//
//    // buffers
//    // add on +2 to the rows in case we are actually needing the next rank's rows (we will need to do the blur on those separately)
//    buffer b_blur_input("b_blur_input", {ROWS+(int64_t)2, COLS+(int64_t)2}, p_float32, a_input, &blur);
//    buffer b_blur_input_gpu("b_blur_input_gpu", {RESIDENT+(int64_t)2, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
//    // the last two rows are junk rows, but we need to make sure we don't give ourselves an
//    // out of bound access
//    buffer b_bx_gpu("b_bx_gpu", {RESIDENT+(int64_t)2, COLS}, p_float32, a_temporary_gpu, &blur);
//    buffer b_ghost("b_ghost", {(int64_t)4, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
//    buffer b_border("b_border", {(int64_t)4, COLS}, p_float32, a_temporary_gpu, &blur);
//    buffer b_by_gpu("b_by_gpu", {RESIDENT, COLS}, p_float32, a_temporary_gpu, &blur);
//    buffer b_by("b_by", {ROWS, COLS}, p_float32, a_output, &blur);
//    buffer b_h2d_wait("b_h2d_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
//    buffer b_d2h_wait("b_d2h_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
//    buffer b_d2h_wait_for_by("b_d2h_wait_for_by", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
//    buffer b_ghost_wait("b_ghost_wait", {(int64_t)4*ROWS/RESIDENT, 1}, p_wait_ptr, a_temporary, &blur);
//    // buffers that are required for code gen, but we don't directly use
//    buffer buff_bx_literals("buff_bx_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
//    buffer buff_border_literals("buff_border_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
//    buffer buff_by_literals("buff_by_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
//    buffer buff_null("buff_null", {ROWS,2}, p_wait_ptr, a_temporary, &blur);
//
//    // access functions
//    blur_input->set_access("{blur_input[r,c]->b_blur_input[r,c]}");
//    bx->set_access("{bx[r,c]->b_bx_gpu[r%" + sresident + ",c]}");
//    by->set_access("{by[r,c]->b_by_gpu[r%" + sresident + ",c]}");
//    by->set_wait_access("{by[r,c]->b_d2h_wait_for_by[r,0]}");
//    h2d.os->set_access("{h2d[r,c]->b_blur_input_gpu[r%" + sresident + ",c]}");
//    h2d.os->set_wait_access("{h2d[r,c]->b_h2d_wait[r,0]}");
//    d2h.os->set_access("{d2h[r,c]->b_by[r,c]}");
//    d2h.os->set_wait_access("{d2h[r,c]->b_d2h_wait[r%" + sresident + ",0]}");
//    ghost.os->set_access("{ghost[r,z,c]->b_ghost[z,c]}");
//    ghost.os->set_wait_access("{ghost[r,z,c]->b_ghost_wait[r+z,0]}");
//    border.set_access("{border[r,z,c]->b_border[z,c]}");
//    d2h_border.os->set_access("{d2h_border[r,z,c]->b_by[r*" + sresident + "+" + sresident + "-2+z,c]}");
//    d2h_border.os->set_wait_access("{d2h_border[r,z,c]->buff_null[r,z]}");
//
//    // code generation
//    blur.lift_dist_comps();
//    blur.set_arguments({&b_blur_input, &b_by});
//    blur.gen_time_space_domain();
//    blur.gen_isl_ast();
//    blur.gen_halide_stmt();
//    blur.gen_halide_obj("/tmp/generated_single_gpu_blur.o");
//    blur.dump_halide_stmt();
//    compile_kernels_to_obj();
//}

void generate_single_gpu() {

    std::string srows = std::to_string(ROWS);
    std::string scols = std::to_string(COLS);
    std::string sresident = std::to_string(RESIDENT);
    function blur("blur_single_gpu");
    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1");
    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1");
    var z(p_int64, "z");

    std::vector<computation *> comps = algorithm(&blur, srows, scols);
    computation *blur_input = comps[0];
    computation *bx = comps[1];
    computation *by = comps[2];

    // split up bx and by based on how much we can fit on the GPU at once
    bx->split(r, RESIDENT, r0, r1);
    by->split(r, RESIDENT, r0, r1);

    // split up the columns for bx and by for GPU level tagging
    bx->split(c, BLOCK_SIZE, c0, c1);
    by->split(c, BLOCK_SIZE, c0, c1);

    // transfer the input from cpu to gpu
    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 1);
    xfer_prop stream1(p_float32, {ASYNC, CUDA}, 1);
    xfer_prop stream2(p_float32, {ASYNC, CUDA}, 2);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 2);
    xfer_prop d2h_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);
    xfer_prop h2d_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    // copy up the last two rows later on
    xfer h2d = computation::create_xfer("{h2d[r,c]: 0<=r<" + srows + " and 0<=c<2+" + scols +
                                        "}", h2d_cuda_async, blur_input->operator()(r,c), &blur);
    tiramisu::wait h2d_wait("{h2d_wait[r,c]: 0<=r<" + srows + " and 0<=c<1}", h2d.os->operator()(r,0), h2d_cuda_async_kernel_stream, true, &blur);
    generator::update_producer_expr_name(bx, "blur_input", "h2d", false);
    // split up the h2d transfer based on how much we can transfer at a time
    h2d.os->split(r, RESIDENT, r0, r1);
    h2d_wait.split(r, RESIDENT, r0, r1);
    // collapse the h2d so it sends the whole row as a chunk
    h2d.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});

    // create a wait on d2h for the next iteration of bx. Also wait on the by kernel
    tiramisu::wait d2h_wait_for_by("{d2h_wait_for_by[r,c]: 0<=r<" + srows + " and 0<=c<1}", by->operator()(r, 0), stream2, true, &blur);
    xfer d2h = computation::create_xfer("{d2h[r,c]: 0<=r<" + srows + " and 0<=c<" + scols +
                                        "}", d2h_cuda_async, by->operator()(r,0), &blur);
    tiramisu::wait d2h_wait("{d2h_wait[r,c]: " + sresident + "<=r<" + srows + " and 0<=c<1}", d2h.os->operator()(r-RESIDENT,0), h2d_cuda_async, true, &blur);
    d2h.os->split(r, RESIDENT, r0, r1);
    d2h_wait.split(r, RESIDENT, r0, r1);
    d2h_wait_for_by.split(r, RESIDENT, r0, r1);
    d2h.os->collapse_many({collapser(2, (int64_t)0, COLS)});

    // special transfer that copies up just the two extra rows needed to get the boundary region correct
    // TODO JESS this access to blur input needs to be fixed!
    xfer ghost = computation::create_xfer("{ghost[r,z,c]: 0<=r<(" + std::to_string(ROWS/RESIDENT) + ") and 0<=z<4 and 0<=c<2+" + scols +
                                          "}", h2d_cuda_async_kernel_stream, blur_input->operator()(/*(r+(int64_t)1)*(ROWS/RESIDENT)+z*/r*RESIDENT+(int64_t)RESIDENT+z-(int64_t)2,c), &blur);
    //  ghost.os->split(r, 2, r0, r1);
    ghost.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});


    expr border_expr = (ghost.os->operator()(0, z, c) + ghost.os->operator()(0, z, c+(int64_t)1) + ghost.os->operator()(0, z, c+(int64_t)2) +
                        ghost.os->operator()(0, z+(int64_t)1, c) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)2) +
                        ghost.os->operator()(0, z+(int64_t)2, c) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)2)) / 9.0f;
    computation border("{border[r,z,c]: 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=z<2 and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);
    border.split(c, BLOCK_SIZE, c0, c1);

    xfer d2h_border = computation::create_xfer("{d2h_border[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=c<" + scols + "}", d2h_cuda_async_kernel_stream, border(r,z,c), &blur);
    d2h_border.os->collapse_many({collapser(2, (int64_t)0, COLS)});
    tiramisu::wait d2h_border_wait("{d2h_border_wait[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=c<1}", d2h_border.os->operator()(r,z,0), stream1, true, &blur);
    d2h_border_wait.set_schedule_this_comp(false);
    // order
    d2h_wait.before(*h2d.os, r1);
    h2d.os->before(h2d_wait, r1);
    h2d_wait.before(*bx, r1);
    bx->before(*by, r0);
    by->before(d2h_wait_for_by, r1);
    d2h_wait_for_by.before(*d2h.os, r1);
    d2h.os->before(*ghost.os, r);
    ghost.os->before(border, r);
    border.before(*d2h_border.os, z);
    d2h_border.os->before(d2h_border_wait, z);

    // tag for the GPU
    bx->tag_gpu_level2(c0, c1, 0);
    by->tag_gpu_level2(c0, c1, 0);
    border.tag_gpu_level2(c0, c1, 0);

    // buffers
    // add on +2 to the rows in case we are actually needing the next rank's rows (we will need to do the blur on those separately)
    buffer b_blur_input("b_blur_input", {ROWS+(int64_t)2, COLS+(int64_t)2}, p_float32, a_input, &blur);
    buffer b_blur_input_gpu("b_blur_input_gpu", {RESIDENT+(int64_t)2, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
    // the last two rows are junk rows, but we need to make sure we don't give ourselves an
    // out of bound access
    buffer b_bx_gpu("b_bx_gpu", {RESIDENT+(int64_t)2, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_ghost("b_ghost", {(int64_t)4, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
    buffer b_border("b_border", {(int64_t)4, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_by_gpu("b_by_gpu", {RESIDENT, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_by("b_by", {ROWS, COLS}, p_float32, a_output, &blur);
    buffer b_h2d_wait("b_h2d_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_d2h_wait("b_d2h_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_d2h_wait_for_by("b_d2h_wait_for_by", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_ghost_wait("b_ghost_wait", {(int64_t)4*ROWS/RESIDENT, 1}, p_wait_ptr, a_temporary, &blur);
    // buffers that are required for code gen, but we don't directly use
    buffer buff_bx_literals("buff_bx_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_border_literals("buff_border_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_by_literals("buff_by_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_null("buff_null", {ROWS,2}, p_wait_ptr, a_temporary, &blur);

    // access functions
    blur_input->set_access("{blur_input[r,c]->b_blur_input[r,c]}");
    bx->set_access("{bx[r,c]->b_bx_gpu[r%" + sresident + ",c]}");
    by->set_access("{by[r,c]->b_by_gpu[r%" + sresident + ",c]}");
    by->set_wait_access("{by[r,c]->b_d2h_wait_for_by[r,0]}");
    h2d.os->set_access("{h2d[r,c]->b_blur_input_gpu[r%" + sresident + ",c]}");
    h2d.os->set_wait_access("{h2d[r,c]->b_h2d_wait[r,0]}");
    d2h.os->set_access("{d2h[r,c]->b_by[r,c]}");
    d2h.os->set_wait_access("{d2h[r,c]->b_d2h_wait[r%" + sresident + ",0]}");
    ghost.os->set_access("{ghost[r,z,c]->b_ghost[z,c]}");
    ghost.os->set_wait_access("{ghost[r,z,c]->b_ghost_wait[r+z,0]}");
    border.set_access("{border[r,z,c]->b_border[z,c]}");
    d2h_border.os->set_access("{d2h_border[r,z,c]->b_by[r*" + sresident + "+" + sresident + "-2+z,c]}");
    d2h_border.os->set_wait_access("{d2h_border[r,z,c]->buff_null[r,z]}");

    // code generation
    blur.lift_dist_comps();
    blur.set_arguments({&b_blur_input, &b_by});
    blur.gen_time_space_domain();
    blur.gen_isl_ast();
    blur.gen_halide_stmt();
    blur.gen_halide_obj("/tmp/generated_single_gpu_blur.o");
    blur.dump_halide_stmt();
    compile_kernels_to_obj();
}

void generate_single_gpu_shared() {
    std::string srows = std::to_string(ROWS);
    std::string scols = std::to_string(COLS);
    std::string sresident = std::to_string(RESIDENT);
    function blur("blur_single_gpu_shared");
    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1");
    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1");
    var z(p_int64, "z");

    std::vector<computation *> comps = algorithm(&blur, srows, scols);
    computation *blur_input = comps[0];
    computation *bx = comps[1];
    computation *by = comps[2];

    // split up bx and by based on how much we can fit on the GPU at once
    bx->split(r, RESIDENT, r0, r1);
    by->split(r, RESIDENT, r0, r1);

    // split up the columns for bx and by for GPU level tagging
    bx->split(c, BLOCK_SIZE, c0, c1);
    by->split(c, BLOCK_SIZE, c0, c1);

    // transfer the input from cpu to gpu
    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 1);
    xfer_prop stream1(p_float32, {ASYNC, CUDA}, 1);
    xfer_prop stream2(p_float32, {ASYNC, CUDA}, 2);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 2);
    xfer_prop d2h_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);
    xfer_prop h2d_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    // copy up the last two rows later on
    xfer h2d = computation::create_xfer("{h2d[r,c]: 0<=r<" + srows + " and 0<=c<2+" + scols +
                                        "}", h2d_cuda_async, blur_input->operator()(r,c), &blur);
    tiramisu::wait h2d_wait("{h2d_wait[r,c]: 0<=r<" + srows + " and 0<=c<1}", h2d.os->operator()(r,0), h2d_cuda_async_kernel_stream, true, &blur);
    generator::update_producer_expr_name(bx, "blur_input", "h2d", false);
    // split up the h2d transfer based on how much we can transfer at a time
    h2d.os->split(r, RESIDENT, r0, r1);
    h2d_wait.split(r, RESIDENT, r0, r1);
    // collapse the h2d so it sends the whole row as a chunk
    h2d.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});

    // create a wait on d2h for the next iteration of bx. Also wait on the by kernel
    tiramisu::wait d2h_wait_for_by("{d2h_wait_for_by[r,c]: 0<=r<" + srows + " and 0<=c<1}", by->operator()(r, 0), stream2, true, &blur);
    xfer d2h = computation::create_xfer("{d2h[r,c]: 0<=r<" + srows + " and 0<=c<" + scols +
                                        "}", d2h_cuda_async, by->operator()(r,0), &blur);
    tiramisu::wait d2h_wait("{d2h_wait[r,c]: " + sresident + "<=r<" + srows + " and 0<=c<1}", d2h.os->operator()(r-RESIDENT,0), h2d_cuda_async, true, &blur);
    d2h.os->split(r, RESIDENT, r0, r1);
    d2h_wait.split(r, RESIDENT, r0, r1);
    d2h_wait_for_by.split(r, RESIDENT, r0, r1);
    d2h.os->collapse_many({collapser(2, (int64_t)0, COLS)});

    // special transfer that copies up just the two extra rows needed to get the boundary region correct
    // TODO JESS this access to blur input needs to be fixed!
    xfer ghost = computation::create_xfer("{ghost[r,z,c]: 0<=r<(" + std::to_string(ROWS/RESIDENT) + ") and 0<=z<4 and 0<=c<2+" + scols +
                                          "}", h2d_cuda_async_kernel_stream, blur_input->operator()(/*(r+(int64_t)1)*(ROWS/RESIDENT)+z*/r*RESIDENT+(int64_t)RESIDENT+z-(int64_t)2,c), &blur);
    //  ghost.os->split(r, 2, r0, r1);
    ghost.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});


    expr border_expr = (ghost.os->operator()(0, z, c) + ghost.os->operator()(0, z, c+(int64_t)1) + ghost.os->operator()(0, z, c+(int64_t)2) +
                        ghost.os->operator()(0, z+(int64_t)1, c) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)2) +
                        ghost.os->operator()(0, z+(int64_t)2, c) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)2)) / 9.0f;
    computation border("{border[r,z,c]: 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=z<2 and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);
    border.split(c, BLOCK_SIZE, c0, c1);

    xfer d2h_border = computation::create_xfer("{d2h_border[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=c<" + scols + "}", d2h_cuda_async_kernel_stream, border(r,z,c), &blur);
    d2h_border.os->collapse_many({collapser(2, (int64_t)0, COLS)});
    tiramisu::wait d2h_border_wait("{d2h_border_wait[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=c<1}", d2h_border.os->operator()(r,z,0), stream1, true, &blur);
    d2h_border_wait.set_schedule_this_comp(false);
    // order
    d2h_wait.before(*h2d.os, r1);
    h2d.os->before(h2d_wait, r1);
    h2d_wait.before(*bx, r1);
    bx->before(*by, r0);
    by->before(d2h_wait_for_by, r1);
    d2h_wait_for_by.before(*d2h.os, r1);
    d2h.os->before(*ghost.os, r);
    ghost.os->before(border, r);
    border.before(*d2h_border.os, z);
    d2h_border.os->before(d2h_border_wait, z);

    // tag for the GPU
    bx->tag_gpu_level2(c0, c1, 0);
    by->tag_gpu_level2(c0, c1, 0);
    border.tag_gpu_level2(c0, c1, 0);

    // buffers
    // add on +2 to the rows in case we are actually needing the next rank's rows (we will need to do the blur on those separately)
    buffer b_blur_input("b_blur_input", {ROWS+(int64_t)2, COLS+(int64_t)2}, p_float32, a_input, &blur);
    buffer b_blur_input_gpu("b_blur_input_gpu", {RESIDENT+(int64_t)2, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
    // the last two rows are junk rows, but we need to make sure we don't give ourselves an
    // out of bound access
    buffer b_bx_gpu("b_bx_gpu", {RESIDENT+(int64_t)2, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_ghost("b_ghost", {(int64_t)4, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
    buffer b_border("b_border", {(int64_t)4, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_by_gpu("b_by_gpu", {RESIDENT, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_by("b_by", {ROWS, COLS}, p_float32, a_output, &blur);
    buffer b_h2d_wait("b_h2d_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_d2h_wait("b_d2h_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_d2h_wait_for_by("b_d2h_wait_for_by", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_ghost_wait("b_ghost_wait", {(int64_t)4*ROWS/RESIDENT, 1}, p_wait_ptr, a_temporary, &blur);
    // buffers that are required for code gen, but we don't directly use
    buffer buff_bx_literals("buff_bx_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_border_literals("buff_border_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_by_literals("buff_by_literals", {ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_null("buff_null", {ROWS,2}, p_wait_ptr, a_temporary, &blur);

    // access functions
    blur_input->set_access("{blur_input[r,c]->b_blur_input[r,c]}");
    bx->set_access("{bx[r,c]->b_bx_gpu[r%" + sresident + ",c]}");
    by->set_access("{by[r,c]->b_by_gpu[r%" + sresident + ",c]}");
    by->set_wait_access("{by[r,c]->b_d2h_wait_for_by[r,0]}");
    h2d.os->set_access("{h2d[r,c]->b_blur_input_gpu[r%" + sresident + ",c]}");
    h2d.os->set_wait_access("{h2d[r,c]->b_h2d_wait[r,0]}");
    d2h.os->set_access("{d2h[r,c]->b_by[r,c]}");
    d2h.os->set_wait_access("{d2h[r,c]->b_d2h_wait[r%" + sresident + ",0]}");
    ghost.os->set_access("{ghost[r,z,c]->b_ghost[z,c]}");
    ghost.os->set_wait_access("{ghost[r,z,c]->b_ghost_wait[r+z,0]}");
    border.set_access("{border[r,z,c]->b_border[z,c]}");
    d2h_border.os->set_access("{d2h_border[r,z,c]->b_by[r*" + sresident + "+" + sresident + "-2+z,c]}");
    d2h_border.os->set_wait_access("{d2h_border[r,z,c]->buff_null[r,z]}");

    // code generation
    blur.lift_dist_comps();
    blur.set_arguments({&b_blur_input, &b_by});
    blur.gen_time_space_domain();
    blur.gen_isl_ast();
    blur.gen_halide_stmt();
    blur.gen_halide_obj("/tmp/generated_single_gpu_blur.o");
    blur.dump_halide_stmt();
    compile_kernels_to_obj();
}

// single node
void generate_multi_gpu() {
    std::string srows = std::to_string(GPU_ROWS);
    std::string scols = std::to_string(COLS);
    std::string sresident = std::to_string(RESIDENT);
    function blur("blur_multi_gpu");
    var r(p_int64, "r"), r0(p_int64, "r0"), r1(p_int64, "r1");
    var c(p_int64, "c"), c0(p_int64, "c0"), c1(p_int64, "c1");
    var z(p_int64, "z");

    std::vector<computation *> comps = algorithm(&blur, srows, scols);
    computation *blur_input = comps[0];
    computation *bx = comps[1];
    computation *by = comps[2];

    var r2("r2"), q("q"), rq("rq");

    bx->split(r, GPU_ROWS/GPU_PROCS, q, rq);
    by->split(r, GPU_ROWS/GPU_PROCS, q, rq);

    // split up bx and by based on how much we can fit on the GPU at once
    bx->split(rq, RESIDENT, r2, r1);
    by->split(rq, RESIDENT, r2, r1);

    // split up the columns for bx and by for GPU level tagging
    bx->split(c, BLOCK_SIZE, c0, c1);
    by->split(c, BLOCK_SIZE, c0, c1);

    //  by->set_schedule_this_comp(false);

    // transfer the input from cpu to gpu
    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);//1);
    xfer_prop stream1(p_float32, {ASYNC, CUDA}, 0);//1);
    xfer_prop stream2(p_float32, {ASYNC, CUDA}, 0);//2);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);//2);
    xfer_prop d2h_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);
    xfer_prop h2d_cuda_async_kernel_stream(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    // copy up the last two rows later on
    xfer h2d = computation::create_xfer("{h2d[r,c]: 0<=r<" + srows + " and 0<=c<2+" + scols +
                                        "}", h2d_cuda_async, blur_input->operator()(r,c), &blur);
    tiramisu::wait h2d_wait("{h2d_wait[r,c]: 0<=r<" + srows + " and 0<=c<1}", h2d.os->operator()(r,0), h2d_cuda_async_kernel_stream, true, &blur);
    generator::update_producer_expr_name(bx, "blur_input", "h2d", false);
    // split up the h2d transfer based on how much we can transfer at a time
    h2d.os->split(r, GPU_ROWS/GPU_PROCS, q, rq);
    h2d_wait.split(r, GPU_ROWS/GPU_PROCS, q, rq);
    h2d.os->split(rq, RESIDENT, r2, r1);
    h2d_wait.split(rq, RESIDENT, r2, r1);
    // collapse the h2d so it sends the whole row as a chunk
    h2d.os->collapse_many({collapser(3, (int64_t)0, COLS+(int64_t)2)});

    // create a wait on d2h for the next iteration of bx. Also wait on the by kernel
    tiramisu::wait d2h_wait_for_by("{d2h_wait_for_by[r,c]: 0<=r<" + srows + " and 0<=c<1}", by->operator()(r, 0), stream2, true, &blur);
    xfer d2h = computation::create_xfer("{d2h[r,c]: 0<=r<" + srows + " and 0<=c<" + scols +
                                        "}", d2h_cuda_async, by->operator()(r,0), &blur);
    tiramisu::wait d2h_wait("{d2h_wait[r,c]: 0<=r<" + srows + " and 0<=c<1}", d2h.os->operator()(r-RESIDENT,0), h2d_cuda_async, true, &blur);
    d2h.os->split(r, GPU_ROWS/GPU_PROCS, q, rq);
    d2h_wait.split(r, GPU_ROWS/GPU_PROCS, q, rq);
    d2h_wait_for_by.split(r, GPU_ROWS/GPU_PROCS, q, rq);

    d2h.os->split(rq, RESIDENT, r2, r1);
    d2h_wait.split(rq, RESIDENT, r2, r1);
    d2h_wait_for_by.split(rq, RESIDENT, r2, r1);
    d2h.os->collapse_many({collapser(3, (int64_t)0, COLS)});

    // special transfer that copies up just the two extra rows needed to get the boundary region correct
    // TODO JESS this access to blur input needs to be fixed!
    xfer ghost = computation::create_xfer("{ghost[r,z,c]: 0<=r<(" + std::to_string(GPU_ROWS/RESIDENT) + ") and 0<=z<4 and 0<=c<2+" + scols +
                                          "}", h2d_cuda_async_kernel_stream, blur_input->operator()(r*RESIDENT+(int64_t)RESIDENT+z-(int64_t)2,c), &blur);
    ghost.os->split(r, GPU_ROWS/RESIDENT/GPU_PROCS, q, rq);
    ghost.os->collapse_many({collapser(3, (int64_t)0, COLS+(int64_t)2)});


    expr border_expr = (ghost.os->operator()(0, z, c) + ghost.os->operator()(0, z, c+(int64_t)1) + ghost.os->operator()(0, z, c+(int64_t)2) +
                        ghost.os->operator()(0, z+(int64_t)1, c) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)1, c+(int64_t)2) +
                        ghost.os->operator()(0, z+(int64_t)2, c) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)1) + ghost.os->operator()(0, z+(int64_t)2, c+(int64_t)2)) / 9.0f;
    computation border("{border[r,z,c]: 0<=r<" + std::to_string(GPU_ROWS/RESIDENT) + " and 0<=z<2 and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);
    border.split(r, GPU_ROWS/RESIDENT/GPU_PROCS, q, rq);
    border.split(c, BLOCK_SIZE, c0, c1);

    xfer d2h_border = computation::create_xfer("{d2h_border[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(GPU_ROWS/RESIDENT) + " and 0<=c<" + scols + "}", d2h_cuda_async_kernel_stream, border(r,z,c), &blur);
    d2h_border.os->split(r, GPU_ROWS/RESIDENT/GPU_PROCS, q, rq);
    d2h_border.os->collapse_many({collapser(3, (int64_t)0, COLS)});
    tiramisu::wait d2h_border_wait("{d2h_border_wait[r,z,c]: 0<=z<2 and 0<=r<" + std::to_string(GPU_ROWS/RESIDENT) + " and 0<=c<1}", d2h_border.os->operator()(r,z,0), stream1, true, &blur);
    d2h_border_wait.split(r, GPU_ROWS/GPU_PROCS, q, rq);
    d2h_border_wait.set_schedule_this_comp(false);


    // order
    d2h_wait.before(*h2d.os, r1);
    h2d.os->before(h2d_wait, r1);
    h2d_wait.before(*bx, r1);
    bx->before(*by, r2);
    by->before(d2h_wait_for_by, r1);
    d2h_wait_for_by.before(*d2h.os, r1);
    d2h.os->before(*ghost.os, rq);
    ghost.os->before(border, rq);
    border.before(*d2h_border.os, z);
    d2h_border.os->before(d2h_border_wait, z);

    bx->tag_distribute_level(q);
    by->tag_distribute_level(q);
    d2h_wait.tag_distribute_level(q);
    h2d_wait.tag_distribute_level(q);
    h2d.os->tag_distribute_level(q);
    d2h_wait_for_by.tag_distribute_level(q);
    d2h.os->tag_distribute_level(q);
    ghost.os->tag_distribute_level(q);
    border.tag_distribute_level(q);
    d2h_border.os->tag_distribute_level(q);
    d2h_border_wait.tag_distribute_level(q);


#ifdef COOP
    // shift the ranks of everything so we start from 0. CPU gets [0 to CPU_PROCS) as ranks, then the gpu gets the rest
    blur.set_rank_offset(-CPU_PROCS); // actual MPI rank will be slightly different than the rank conditioned on
#endif

    // tag for the GPU
    bx->tag_gpu_level2(c0, c1, 0);
    by->tag_gpu_level2(c0, c1, 0);
    border.tag_gpu_level2(c0, c1, 0);

    //  by->set_schedule_this_comp(false);
    //  d2h_wait_for_by.set_schedule_this_comp(false);

    // buffers
    // add on +2 to the rows in case we are actually needing the next rank's rows (we will need to do the blur on those separately)
    buffer b_blur_input("b_blur_input", {GPU_ROWS/GPU_PROCS+(int64_t)2, COLS+(int64_t)2}, p_float32, a_input, &blur);
    buffer b_blur_input_gpu("b_blur_input_gpu", {RESIDENT+(int64_t)2, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
    // the last two rows are junk rows, but we need to make sure we don't give ourselves an
    // out of bound access
    buffer b_bx_gpu("b_bx_gpu", {RESIDENT+(int64_t)2, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_ghost("b_ghost", {(int64_t)4, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
    buffer b_border("b_border", {(int64_t)4, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_by_gpu("b_by_gpu", {RESIDENT, COLS}, p_float32, a_temporary_gpu, &blur);
    buffer b_by("b_by", {GPU_ROWS/GPU_PROCS, COLS}, p_float32, a_output, &blur);
    buffer b_h2d_wait("b_h2d_wait", {GPU_ROWS/GPU_PROCS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_d2h_wait("b_d2h_wait", {GPU_ROWS/GPU_PROCS, 1}, p_wait_ptr, a_input, &blur);
    buffer b_d2h_wait_for_by("b_d2h_wait_for_by", {GPU_ROWS/GPU_PROCS, 1}, p_wait_ptr, a_temporary, &blur);
    buffer b_ghost_wait("b_ghost_wait", {(int64_t)4*GPU_ROWS/GPU_PROCS/RESIDENT, 1}, p_wait_ptr, a_temporary, &blur);
    // buffers that are required for code gen, but we don't directly use
    buffer buff_bx_literals("buff_bx_literals", {GPU_ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_border_literals("buff_border_literals", {GPU_ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_by_literals("buff_by_literals", {GPU_ROWS, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
    buffer buff_null("buff_null", {GPU_ROWS/GPU_PROCS,2}, p_wait_ptr, a_temporary, &blur);

    // access functions

    blur_input->set_access("{blur_input[r,c]->b_blur_input[r,c]}");
    bx->set_access("{bx[r,c]->b_bx_gpu[r%" + sresident + ",c]}");
    by->set_access("{by[r,c]->b_by_gpu[r%" + sresident + ",c]}");
    by->set_wait_access("{by[r,c]->b_d2h_wait_for_by[r,0]}");
    h2d.os->set_access("{h2d[r,c]->b_blur_input_gpu[r%" + sresident + ",c]}");
    h2d.os->set_wait_access("{h2d[r,c]->b_h2d_wait[r,0]}");
    d2h.os->set_access("{d2h[r,c]->b_by[r,c]}");
    d2h.os->set_wait_access("{d2h[r,c]->b_d2h_wait[r%" + sresident + ",0]}");
    ghost.os->set_access("{ghost[r,z,c]->b_ghost[z,c]}");
    ghost.os->set_wait_access("{ghost[r,z,c]->b_ghost_wait[r+z,0]}");
    border.set_access("{border[r,z,c]->b_border[z,c]}");
    d2h_border.os->set_access("{d2h_border[r,z,c]->b_by[r*" + sresident + "+" + sresident + "-2+z,c]}");
    d2h_border.os->set_wait_access("{d2h_border[r,z,c]->buff_null[r,z]}");


    // code generation
    blur.lift_dist_comps();
    blur.set_arguments({&b_blur_input, &b_by, &b_d2h_wait});
    blur.gen_time_space_domain();
    blur.gen_isl_ast();
    blur.gen_halide_stmt();
    blur.gen_halide_obj("/tmp/generated_multi_gpu_blur.o");
    blur.dump_halide_stmt();
    compile_kernels_to_obj();
}

int main() {
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int64);;
#ifdef GPU
    #ifdef DIST
    generate_multi_gpu();
#else
    //    generate_single_gpu();
    generate_single_gpu_big_box();
#endif
#elif defined(COOP)
//    generate_multi_cpu();
    generate_multi_gpu();
#else
#ifdef DIST
    std::cerr << "Generating multi cpu" << std::endl;
  generate_multi_cpu();
#else
    generate_single_cpu_big_box();
#endif
#endif
}
