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

    function blur_dist("blur_dist");

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

    var y("y"), x("x");
    constant rows_const("rows", expr(rows), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant cols_const("cols", expr(cols), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    computation blur_input("[rows, cols]->{blur_input[i1, i0]: 0<=i1<rows and 0<=i0<cols}", expr(), false,
                           T_DATA_TYPE, &blur_dist);

    computation bx("[rows, cols]->{bx[y, x]: 0<=y<rows and 0<=x<cols-2}",
                   (((blur_input(y, x) + blur_input(y, (x + expr((C_LOOP_ITER_TYPE)1)))) +
                     blur_input(y, (x + expr((C_LOOP_ITER_TYPE)2)))) / expr((C_DATA_TYPE)3)),
                   true, T_DATA_TYPE, &blur_dist);

    computation by("[rows, cols]->{by[y, x]: 0<=y<rows-2 and 0<=x<cols-2}",
                   (((bx(y, x) + bx((y + expr((C_LOOP_ITER_TYPE)1)), x)) +
                     bx((y + expr((C_LOOP_ITER_TYPE)2)), x)) / expr((C_DATA_TYPE)3)),
                   true, T_DATA_TYPE, &blur_dist);

    // -------------------------------------------------------
    // Layer II common
    // -------------------------------------------------------


    constant procs_const("procs", expr(procs), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant nodes_const("nodes", expr(nodes), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    var y1("y1"), y2("y2"), q("q");

    bx.split(y, rows_per_proc, y1, y2);
    by.split(y, rows_per_proc, y1, y2);

#ifdef CPU_ONLY
    communication_prop sync_block("sync_block_MPI", T_DATA_TYPE, {SYNC, BLOCK, MPI, CPU2CPU});
    communication_prop async_block("async_block_MPI", T_DATA_TYPE, {ASYNC, BLOCK, MPI, CPU2CPU});
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

    bx_exchange.s->collapse_many({collapser(2, 0, (C_LOOP_ITER_TYPE)cols), collapser(1, 0, (C_LOOP_ITER_TYPE)2)});
    bx_exchange.r->collapse_many({collapser(2, 0, (C_LOOP_ITER_TYPE)cols), collapser(1, 0, (C_LOOP_ITER_TYPE)2)});

    tiramisu::expr bx_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1, tiramisu::expr(rows_per_proc), tiramisu::expr(rows_per_proc+2));
    tiramisu::expr by_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1, tiramisu::expr(rows_per_proc), tiramisu::expr(rows_per_proc-2));

    tiramisu::buffer buff_input("buff_input", {tiramisu::expr(rows_per_proc), tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);

    tiramisu::buffer buff_bx("buff_bx", {bx_select_dim0, tiramisu::expr(cols - 2)},
                                 T_DATA_TYPE, tiramisu::a_temporary, &blur_dist);

    tiramisu::buffer buff_by("buff_by", {by_select_dim0, tiramisu::expr(cols - 2)},
                                 T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");

    bx.set_access("{bx[y, x]->buff_bx[y, x]}");
    by.set_access("{by[y, x]->buff_by[y, x]}");

    bx_exchange.r->set_access("{bx_exchange_r[q,y,x]->buff_bx[" + std::to_string(rows_per_proc) + " + y, x]}");

    blur_dist.set_arguments({&buff_input, &buff_by});
    blur_dist.lift_dist_comps();
    blur_dist.gen_time_space_domain();
    blur_dist.gen_isl_ast();
    blur_dist.gen_halide_stmt();
    blur_dist.gen_halide_obj("./build/generated_blur_dist.o");
#elif defined(GPU_ONLY)

    constant rows_per_proc_const("rows_per_proc", expr(rows_per_proc), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    // rows_per_proc that we end up with after computation is done (last proc does 2 less rows)
    constant rows_per_proc_after_const("rows_per_proc_after", expr(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1, rows_per_proc - 2, rows_per_proc), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant rows_per_node_const("rows_per_node", expr(rows_per_proc), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);
    constant procs_per_node_const("procs_per_node", expr(procs_per_node), T_LOOP_ITER_TYPE, true, NULL, 0, &blur_dist);

    communication_prop h2h_mpi_sync("h2h_mpi_sync", T_DATA_TYPE, {SYNC, BLOCK, MPI, CPU2CPU});
    communication_prop h2h_mpi_async("h2h_mpi_async", T_DATA_TYPE, {ASYNC, BLOCK, MPI, CPU2CPU});
    communication_prop h2d_cuda("h2d_cuda", T_DATA_TYPE, {SYNC, CUDA, CPU2GPU});
    communication_prop d2h_cuda("d2h_cuda", T_DATA_TYPE, {SYNC, CUDA, GPU2CPU});

    // Minimal communication scheme

    // First, transfer border region between the procs.
    xfer bx_exchange =
            computation::create_xfer(
                    "[cols, nodes]->{bx_exchange_s[q,y,x]: 1<=q<nodes and 0<=y<2 and 0<=x<cols and nodes>1}",
                    "[cols, nodes]->{bx_exchange_r[q,y,x]: nodes<=q<nodes*2-1 and 0<=y<2 and 0<=x<cols and nodes>1}",
                    q + nodes - (C_LOOP_ITER_TYPE)1, q - nodes + (C_LOOP_ITER_TYPE)1, h2h_mpi_async, h2h_mpi_sync,
                    bx(y, x), &blur_dist);

    // Need an  CPU-GPU transfer for each input
    xfer input_cpu_to_gpu =
            computation::create_xfer("[procs, rows_per_proc, cols]->{input_cpu_to_gpu_os[q,y,x]: 0<=q<procs and 0<=y<rows_per_proc+2 and 0<=x<cols}",
                                     h2d_cuda, blur_input(y,x), &blur_dist);

    // True because we need to insert a dummy access since the transfer has 3 dims and blur_input only has 2
    generator::update_producer_expr_name(&bx, "blur_input", "input_cpu_to_gpu_os", true);

    // Transfer the computed data back to the CPU
    xfer gpu_to_cpu =
            computation::create_xfer("[procs, rows_per_proc_after, cols]->{gpu_to_cpu[q,y,x]: 0<=q<procs and 0<=y<rows_per_proc_after and 0<=x<cols-2}",
                                     d2h_cuda, by(y,x), &blur_dist);
    // We want to insert a new computation here that computes bx of the two extra rows. This gives us recomputation
    // instead of communication, which is cheaper for us. The last proc doesn't need to do anything though.
    computation bx_recompute("[rows_per_node, cols, procs]->{bx_recompute[q, y, x]: 0<=q<procs-1 and rows_per_node<=y<rows_per_node+2 and 0<=x<cols-2}",
                             (((blur_input(y, x) + blur_input(y, (x + expr((C_LOOP_ITER_TYPE)1)))) +
                               blur_input(y, (x + expr((C_LOOP_ITER_TYPE)2)))) / expr((C_DATA_TYPE)3)),
                             true, T_DATA_TYPE, &blur_dist);

    bx_exchange.s->before(*bx_exchange.r, computation::root);
    bx_exchange.r->before(*input_cpu_to_gpu.os, computation::root);
    input_cpu_to_gpu.os->before(bx, computation::root);
    bx.before(bx_recompute, computation::root);
    bx_recompute.before(by, computation::root);
    by.before(*gpu_to_cpu.os, computation::root);

    bx_exchange.s->tag_distribute_level(q, false);
    bx_exchange.r->tag_distribute_level(q, false);
    input_cpu_to_gpu.os->tag_distribute_level(q, false);
    bx.tag_distribute_level(y1);
    bx_recompute.tag_distribute_level(q);
    by.tag_distribute_level(y1);
    gpu_to_cpu.os->tag_distribute_level(q, false);

    bx.tag_gpu_level(y2, x);
    bx_recompute.tag_gpu_level(y, x);
    by.tag_gpu_level(y2, x);

    tiramisu::expr bx_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1,
                                  tiramisu::expr(rows_per_proc), tiramisu::expr(rows_per_proc+2));
    tiramisu::expr by_select_dim0(tiramisu::o_select, var(T_LOOP_ITER_TYPE, "rank") == procs-1,
                                  tiramisu::expr(rows_per_proc-2), tiramisu::expr(rows_per_proc));

    tiramisu::buffer buff_input("buff_input", {bx_select_dim0, tiramisu::expr(cols)}, T_DATA_TYPE,
                                tiramisu::a_input, &blur_dist);

    // TODO change this to a temporary one an allocate ourselves
    tiramisu::buffer buff_input_gpu("buff_input_gpu", {tiramisu::expr(rows_per_proc), tiramisu::expr(cols)}, T_DATA_TYPE,
                                    tiramisu::a_output, &blur_dist);

    tiramisu::buffer buff_bx_gpu("buff_bx_gpu", {bx_select_dim0, tiramisu::expr(cols - 2)},
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    tiramisu::buffer buff_by_gpu("buff_by_gpu", {by_select_dim0, tiramisu::expr(cols - 2)},
                             T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    tiramisu::buffer buff_by("buff_by", {by_select_dim0, tiramisu::expr(cols - 2)},
                                 T_DATA_TYPE, tiramisu::a_output, &blur_dist);

    blur_input.set_access("{blur_input[i1, i0]->buff_input[i1, i0]}");

    bx.set_access("{bx[y, x]->buff_bx_gpu[y, x]}");
    bx_recompute.set_access("{bx_recompute[q,y,x]->buff_bx_gpu[y,x]}");
    by.set_access("{by[y, x]->buff_by_gpu[y, x]}");


    bx_exchange.r->set_access("{bx_exchange_r[q,y,x]->buff_bx_gpu[" + std::to_string(rows_per_proc) + " + y, x]}");
    input_cpu_to_gpu.os->set_access("{input_cpu_to_gpu_os[q,y,x]->buff_input_gpu[y,x]}");
    gpu_to_cpu.os->set_access("{gpu_to_cpu[q,y,x]->buff_by[y,x]}");
    blur_dist.set_arguments({&buff_input, &buff_bx_gpu, &buff_input_gpu, &buff_by_gpu, &buff_by});
    blur_dist.lift_dist_comps();
    blur_dist.gen_time_space_domain();
    blur_dist.gen_isl_ast();
    blur_dist.gen_halide_stmt();
    blur_dist.gen_halide_obj("./build/generated_blur_dist.o", {Halide::Target::CUDA});
#endif

    blur_dist.dump_halide_stmt();

    return 0;

}
