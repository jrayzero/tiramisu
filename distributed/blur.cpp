//
// Created by Jessica Ray on 11/2/17.
//

// This distribution assumes that the data resides across multiple nodes, so communication is only needed to transfer
// over the ghost zones.

// This splits data evenly

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

#include "blur_params.h"

using namespace tiramisu;

int main() {
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(T_LOOP_ITER_TYPE);

#ifdef CPU_ONLY
    function blur_dist("blur_dist");
#elif defined(GPU_ONLY)
    function blur_dist("blur_dist_gpu");
#endif

    C_LOOP_ITER_TYPE rows = ROWS;
    C_LOOP_ITER_TYPE cols = COLS;
    C_LOOP_ITER_TYPE nodes = NODES;
    C_LOOP_ITER_TYPE procs = PROCS;
    assert(procs % nodes == 0);
    C_LOOP_ITER_TYPE procs_per_node = procs / nodes;
    C_LOOP_ITER_TYPE rows_per_proc = rows / procs;

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    C_LOOP_ITER_TYPE offset = 200;
    var y(T_LOOP_ITER_TYPE, "y"), x(T_LOOP_ITER_TYPE, "x");
    constant rows_const("rows", expr(rows), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant cols_const("cols", expr(cols), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    computation blur_input("[rows, cols]->{blur_input[i1, i0]: 0<=i1<rows and 0<=i0<cols}", expr(), false,
                           T_DATA_TYPE, &blur_dist);
    // compute more than is needed, just to make the comp a little easier
    computation bx("[rows, cols]->{bx[y, x]: 0<=y<rows and 0<=x<cols}",
                   (((blur_input(y, x) + blur_input(y, (x + expr((C_LOOP_ITER_TYPE)1)))) +
                     blur_input(y, (x + expr((C_LOOP_ITER_TYPE)2)))) / expr((C_DATA_TYPE)3)),
                   true, T_DATA_TYPE, &blur_dist);

    computation by("[rows, cols]->{by[y, x]: 0<=y<rows and 0<=x<cols}",
                   (((bx(y, x) + bx((y + expr((C_LOOP_ITER_TYPE)1)), x)) +
                     bx((y + expr((C_LOOP_ITER_TYPE)2)), x)) / expr((C_DATA_TYPE)3)),
                   true, T_DATA_TYPE, &blur_dist);

    // -------------------------------------------------------
    // Layer II common
    // -------------------------------------------------------


    constant procs_const("procs", expr(procs), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant nodes_const("nodes", expr(nodes), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

#ifdef CPU_ONLY
    #ifdef DISTRIBUTE
    var y1("y1"), y2("y2"), q("q");
    bx.split(y, rows_per_proc, y1, y2);
    by.split(y, rows_per_proc, y1, y2);
    xfer_prop sync_block(T_DATA_TYPE, {SYNC, BLOCK, MPI, CPU2CPU});
    xfer_prop async_block(T_DATA_TYPE, {ASYNC, BLOCK, MPI, CPU2CPU});
    // transfer the computed rows from bx
    xfer bx_exchange = computation::create_xfer("[procs, cols]->{bx_exchange_s[q,y,x]: 1<=q<procs and 0<=y<2 and 0<=x<cols-2 and procs>1}",
                                                "[procs, cols]->{bx_exchange_r[q,y,x]: 0<=q<procs-1 and 0<=y<2 and 0<=x<cols-2 and procs>1}",
                                                q-(C_LOOP_ITER_TYPE)1, q+(C_LOOP_ITER_TYPE)1, async_block, sync_block,
                                                bx(y,x), &blur_dist);


    bx.before(*bx_exchange.s, computation::root);
    bx_exchange.s->before(*bx_exchange.r, computation::root);
    bx_exchange.r->before(by, computation::root);

    bx.tag_distribute_level(y1);
    by.tag_distribute_level(y1);

    bx_exchange.s->tag_distribute_level(q, false);
    bx_exchange.r->tag_distribute_level(q, false);

    bx_exchange.s->collapse_many({collapser(2, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)cols), collapser(1, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)2)});
    bx_exchange.r->collapse_many({collapser(2, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)cols), collapser(1, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)2)});

    tiramisu::expr bx_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1, tiramisu::expr(rows_per_proc), tiramisu::expr(rows_per_proc+2));
    tiramisu::expr by_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1, tiramisu::expr(rows_per_proc), tiramisu::expr(rows_per_proc));//-2));

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(rows_per_proc), tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);

    tiramisu::buffer buff_bx("buff_bx", {bx_select_dim0, tiramisu::expr(cols)},// - 2)},
                                 T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    tiramisu::buffer buff_by("buff_by", {by_select_dim0, tiramisu::expr(cols)},// - 2)},
                                 T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");

    bx.set_access("{bx[y, x]->buff_bx[y, x]}");
    by.set_access("{by[y, x]->buff_by[y, x]}");

    bx_exchange.r->set_access("{bx_exchange_r[q,y,x]->buff_bx[" + std::to_string(rows_per_proc) + " + y, x]}");

    blur_dist.lift_dist_comps();
    blur_dist.set_arguments({&buff_input, &buff_bx, &buff_by});
#else
    var y1(T_LOOP_ITER_TYPE, "y1"), y2(T_LOOP_ITER_TYPE, "y2"), x1(T_LOOP_ITER_TYPE, "x1"), x2(T_LOOP_ITER_TYPE, "x2");
    var y3(T_LOOP_ITER_TYPE, "y3"), y4(T_LOOP_ITER_TYPE, "y4");
#ifdef PARALLEL
    bx.split(y, 13500, y3, y4);
    by.split(y, 13500, y3, y4);
    bx.tile(y4, x, (C_LOOP_ITER_TYPE)10, (C_LOOP_ITER_TYPE)8, y1, x1, y2, x2);
    by.tile(y4, x, (C_LOOP_ITER_TYPE)10, (C_LOOP_ITER_TYPE)8, y1, x1, y2, x2);
    bx.tag_parallel_level(y3);
    by.tag_parallel_level(y3);
#else
    bx.tile(y, x, (C_LOOP_ITER_TYPE)10, (C_LOOP_ITER_TYPE)8, y1, x1, y2, x2);
    by.tile(y, x, (C_LOOP_ITER_TYPE)10, (C_LOOP_ITER_TYPE)8, y1, x1, y2, x2);
    bx.tag_vector_level(x2, 8);
    by.tag_vector_level(x2, 8);
#endif

    bx.before(by, x1);//computation::root);

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(rows_per_proc), tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);

    tiramisu::buffer buff_bx("buff_bx", {rows, tiramisu::expr(cols)},// - 2)},
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    tiramisu::buffer buff_by("buff_by", {rows, tiramisu::expr(cols)}, // both were -2 here
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");

    bx.set_access("{bx[y, x]->buff_bx[y, x]}");
    by.set_access("{by[y, x]->buff_by[y, x]}");
    blur_dist.set_arguments({&buff_input, &buff_bx, &buff_by});
#endif
    blur_dist.gen_time_space_domain();
    blur_dist.gen_isl_ast();
    blur_dist.gen_halide_stmt();
    blur_dist.gen_halide_obj("./build/generated_blur_dist.o");
#elif defined(GPU_ONLY)

    var y1("y1"), y2("y2"), y3("y3"), y4("y4"), y5("y5"), y6("y6"), x1("x1"), x2("x2"), q("q");
    bx.split(y, rows_per_proc, y1, y2);
    bx.split(y2, offset, y3, y4);
    bx.split(x, 1000, x1, x2);

    by.split(y, rows_per_proc, y1, y2);
    by.split(y2, offset, y3, y4);
    by.split(x, 1000, x1, x2);

    constant rows_per_proc_const("rows_per_proc", expr(rows_per_proc), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant rows_per_node_const("rows_per_node", expr(rows_per_proc), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant procs_per_node_const("procs_per_node", expr(procs_per_node), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    xfer_prop h2h_mpi_sync(T_DATA_TYPE, {SYNC, BLOCK, MPI, CPU2CPU});
    xfer_prop h2h_mpi_async(T_DATA_TYPE, {ASYNC, BLOCK, MPI, CPU2CPU});
    xfer_prop h2h_mpi_async_nonblock(T_DATA_TYPE, {ASYNC, NONBLOCK, MPI, CPU2CPU});
    xfer_prop h2d_cuda_sync(T_DATA_TYPE, {SYNC, CUDA, CPU2GPU});
    xfer_prop h2d_cuda_async(T_DATA_TYPE, {ASYNC, CUDA, CPU2GPU}, 1);
    xfer_prop kernel_stream(T_DATA_TYPE, {ASYNC, CUDA}, 0);
    xfer_prop d2h_cuda_sync(T_DATA_TYPE, {SYNC, CUDA, GPU2CPU});
    xfer_prop d2h_cuda_async(T_DATA_TYPE, {ASYNC, CUDA, GPU2CPU}, 2);

    xfer_prop kernel(T_DATA_TYPE, {ASYNC, CUDA}, 2);

    // Need an  CPU-GPU transfer for each input
    xfer input_cpu_to_gpu = computation::create_xfer(
            "[procs, rows_per_proc, cols]->{input_cpu_to_gpu_os[y,x]: 0<=y<rows_per_proc and 0<=x<cols}",
            h2d_cuda_async, blur_input(y, x), &blur_dist);
    // Transfer the computed data back to the CPU
    xfer gpu_to_cpu =
            computation::create_xfer("[procs, rows_per_proc, cols]->{gpu_to_cpu_os[y,x]: 0<=y<rows_per_proc and 0<=x<cols}",
                                     d2h_cuda_async, by(y, x), &blur_dist);

    // True because we need to insert a dummy access since the transfer has 3 dims and blur_input only has 2
    generator::update_producer_expr_name(&bx, "blur_input", "input_cpu_to_gpu_os", false);


    input_cpu_to_gpu.os->split(y, rows_per_proc, y1, y2);
    input_cpu_to_gpu.os->split(y2, offset, y3, y4);
    gpu_to_cpu.os->split(y, rows_per_proc, y1, y2);
    gpu_to_cpu.os->split(y2, offset, y3, y4);
    // We want to insert a new computation here that computes bx of the two extra rows. This gives us recomputation
    // instead of communication, which is cheaper for us. The last proc doesn't need to do anything though.
    tiramisu::wait cpu_to_gpu_wait(input_cpu_to_gpu.os->operator()(y, x), kernel_stream, &blur_dist);
    cpu_to_gpu_wait.split(y, rows_per_proc, y1, y2);
    cpu_to_gpu_wait.split(y2, offset, y3, y4);
    tiramisu::wait gpu_to_cpu_wait(gpu_to_cpu.os->operator()(y, x), gpu_to_cpu.os->get_channel(), &blur_dist);
    gpu_to_cpu_wait.split(y, rows_per_proc, y1, y2);
    gpu_to_cpu_wait.split(y2, offset, y3, y4);
    gpu_to_cpu_wait.set_schedule_this_comp(false);// don't actually need this waitopop

    tiramisu::wait kernel_by_wait("[cols, rows]->{by_wait[y]: 0<=y<rows}", by(y, 0), kernel, true, &blur_dist);
    kernel_by_wait.split(y, rows_per_proc, y1, y2);
    kernel_by_wait.split(y2, offset, y3, y4);
    bx.tag_gpu_level2(x1, x2, 0);
    by.tag_gpu_level2(x1, x2, 0);


    bx.tag_distribute_level(y1, true);
    by.tag_distribute_level(y1, true);
    input_cpu_to_gpu.os->tag_distribute_level(y1, true);
    gpu_to_cpu.os->tag_distribute_level(y1, true);
    cpu_to_gpu_wait.tag_distribute_level(y1, true);
    gpu_to_cpu_wait.tag_distribute_level(y1, true);
    kernel_by_wait.tag_distribute_level(y1, true);

    input_cpu_to_gpu.os->before(cpu_to_gpu_wait, y4);//computation::root);//y1);
    cpu_to_gpu_wait.before(bx, y4);//computation::root);//y1);
    bx.before(by, y3);
    by.before(kernel_by_wait, y4);//computation::root);//y1);
    kernel_by_wait.before(*gpu_to_cpu.os, y4);//computation::root);//y1);
    gpu_to_cpu.os->before(gpu_to_cpu_wait, y4);//computation::root); 

    input_cpu_to_gpu.os->collapse_many({collapser(3, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)cols)});
    cpu_to_gpu_wait.collapse_many({collapser(3, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)cols)});
    gpu_to_cpu.os->collapse_many({collapser(3, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)cols)});
    gpu_to_cpu_wait.collapse_many({collapser(3, (C_LOOP_ITER_TYPE)0, (C_LOOP_ITER_TYPE)cols)});

    tiramisu::expr bx_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1,
                                  tiramisu::expr(offset), tiramisu::expr(cols+2));
    tiramisu::expr by_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1,
                                  tiramisu::expr(offset), tiramisu::expr(cols));
#ifdef CHECK_RESULTS
    tiramisu::buffer buff_input("buff_input", {bx_select_dim0, tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);
#else
    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);
#endif
    tiramisu::buffer buff_input_gpu("buff_input_gpu", {bx_select_dim0, tiramisu::expr(cols)}, T_DATA_TYPE,
                                    tiramisu::a_temporary_gpu, &blur_dist);

    tiramisu::buffer buff_bx_gpu("buff_bx_gpu", {bx_select_dim0, tiramisu::expr(cols)},
                                 T_DATA_TYPE, tiramisu::a_temporary_gpu, &blur_dist);

    tiramisu::buffer buff_by_gpu("buff_by_gpu", {by_select_dim0, tiramisu::expr(cols)},
                                 T_DATA_TYPE, tiramisu::a_temporary_gpu, &blur_dist);

#ifdef CHECK_RESULTS
    tiramisu::buffer buff_by("buff_by", {rows_per_proc, tiramisu::expr(cols)},
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);
#else
    tiramisu::buffer buff_by("buff_by", {tiramisu::expr(cols)},
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);
#endif

    tiramisu::buffer buff_cpu_to_gpu_wait("buff_cpu_to_gpu_wait", {rows_per_proc}, tiramisu::p_wait_ptr,
                                          tiramisu::a_temporary, &blur_dist);

    tiramisu::buffer buff_gpu_to_cpu_wait("buff_gpu_to_cpu_wait", {rows_per_proc}, tiramisu::p_wait_ptr,
                                          tiramisu::a_temporary, &blur_dist);

    tiramisu::buffer buff_kernel_by_wait("buff_kernel_by_wait", {rows_per_proc}, tiramisu::p_wait_ptr,
                                         tiramisu::a_temporary, &blur_dist);

    tiramisu::buffer buff_bx_literals("buff_bx_literals", {rows_per_proc, 3}, T_LOOP_ITER_TYPE, tiramisu::a_temporary_gpu,
                                      &blur_dist);
    tiramisu::buffer buff_by_literals("buff_by_literals", {rows_per_proc, 3}, T_LOOP_ITER_TYPE, tiramisu::a_temporary_gpu,
                                      &blur_dist);

#ifdef CHECK_RESULTS
    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");
#else
    blur_input.set_access("{blur_input[i1, i0]->buff_input[i0]}");
#endif

    bx.set_access("{bx[y, x]->buff_bx_gpu[y%" + std::to_string(offset) + ",x]}");

    by.set_access("{by[y, x]->buff_by_gpu[y%" + std::to_string(offset) + ", x]}");

    input_cpu_to_gpu.os->set_access("{input_cpu_to_gpu_os[y,x]->buff_input_gpu[y%" + std::to_string(offset) + ",x]}");

    input_cpu_to_gpu.os->set_wait_access("{input_cpu_to_gpu_os[y,x]->buff_cpu_to_gpu_wait[y]}");
#ifdef CHECK_RESULTS
    gpu_to_cpu.os->set_access("{gpu_to_cpu_os[y,x]->buff_by[y,x]}");
#else
    gpu_to_cpu.os->set_access("{gpu_to_cpu_os[y,x]->buff_by[x]}");
#endif

    gpu_to_cpu.os->set_wait_access("{gpu_to_cpu_os[y,x]->buff_gpu_to_cpu_wait[y]}");

    by.set_wait_access("{by[y,x]->buff_kernel_by_wait[y]}");

    blur_dist.set_arguments({&buff_input, &buff_by});//, &buff_cpu_to_gpu_wait});
    blur_dist.lift_dist_comps();
    blur_dist.gen_time_space_domain();
    blur_dist.gen_isl_ast();
    blur_dist.gen_halide_stmt();
    blur_dist.gen_halide_obj("/tmp/generated_blur_dist.o");//, {Halide::Target::CUDA});//, Halide::Target::Debug});
#endif

    blur_dist.dump_halide_stmt();

    return 0;

}
