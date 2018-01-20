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
  xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 2);
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
  tiramisu::wait d2h_wait_for_by("{d2h_wait_for_by[r,c]: 0<=r<" + srows + " and 0<=c<1}", by->operator()(r, 0), h2d_cuda_async_kernel_stream, true, &blur);
  xfer d2h = computation::create_xfer("{d2h[r,c]: 0<=r<" + srows + " and 0<=c<" + scols + 
                                      "}", d2h_cuda_async, by->operator()(r,c), &blur);
  tiramisu::wait d2h_wait("{d2h_wait[r,c]: " + sresident + "<=r<" + srows + " and 0<=c<1}", d2h.os->operator()(r-RESIDENT,0), h2d_cuda_async, true, &blur);
  d2h.os->split(r, RESIDENT, r0, r1);
  d2h_wait.split(r, RESIDENT, r0, r1);
  d2h_wait_for_by.split(r, RESIDENT, r0, r1);
  d2h.os->collapse_many({collapser(2, (int64_t)0, COLS)});

  // special transfer that copies up just the two extra rows needed to get the boundary region correct
    xfer ghost = computation::create_xfer("{ghost[r,z,c]: 0<=r<" + std::to_string(ROWS/RESIDENT) + " and 0<=z<2 and 0<=c<2+" + scols + 
                                        "}", h2d_cuda_async_kernel_stream, blur_input->operator()((r+(int64_t)1)*(ROWS/RESIDENT)+z,c), &blur);
  //  ghost.os->split(r, 2, r0, r1);
  ghost.os->collapse_many({collapser(2, (int64_t)0, COLS+(int64_t)2)});
  
  // special computation to fix the last two rows of the resident chunk in preparation for computing by
  expr border_expr = (ghost.os->operator()(r,z,c) + ghost.os->operator()(r,z,c+(int64_t)1) + ghost.os->operator()(r,z,c+(int64_t)2)) / 3.0f;
  computation border("{border[r,c]: 0<=r<" + std::to_string(2*ROWS/RESIDENT) + " and 0<=c<" + scols + "}", border_expr, true, p_float32, &blur);
  border.split(r, 2, r0, r1);
  border.split(c, BLOCK_SIZE, c0, c1);
  
  // order
  d2h_wait.before(*h2d.os, r1);
  h2d.os->before(h2d_wait, r1);
  h2d_wait.before(*bx, r1);
    bx->before(*ghost.os, r);
  ghost.os->before(border, r1);
  border.before(*by, r0);
  by->before(d2h_wait_for_by, r1);
  d2h_wait_for_by.before(*d2h.os, r1);

  // tag for the GPU
  bx->tag_gpu_level2(c0, c1, 0);
  border.tag_gpu_level2(c0, c1, 0);
  by->tag_gpu_level2(c0, c1, 0);  

  // buffers
  // add on +2 to the rows in case we are actually needing the next rank's rows (we will need to do the blur on those separately)
  buffer b_blur_input("b_blur_input", {ROWS+(int64_t)2, COLS+(int64_t)2}, p_float32, a_input, &blur);
  buffer b_blur_input_gpu("b_blur_input_gpu", {RESIDENT+(int64_t)2, COLS+(int64_t)2}, p_float32, a_temporary_gpu, &blur);
  // the last two rows are junk rows, but we need to make sure we don't give ourselves an 
  // out of bound access
  buffer b_bx_gpu("b_bx_gpu", {RESIDENT+(int64_t)2, COLS}, p_float32, a_temporary_gpu, &blur);
  buffer b_by_gpu("b_by_gpu", {RESIDENT, COLS}, p_float32, a_temporary_gpu, &blur);
  buffer b_by("b_by", {ROWS, COLS}, p_float32, a_output, &blur);
  buffer b_h2d_wait("b_h2d_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
  buffer b_d2h_wait("b_d2h_wait", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
  buffer b_d2h_wait_for_by("b_d2h_wait_for_by", {ROWS, 1}, p_wait_ptr, a_temporary, &blur);
  buffer b_ghost_wait("b_ghost_wait", {(int64_t)2*ROWS/RESIDENT, 1}, p_wait_ptr, a_temporary, &blur);
  // buffers that are required for code gen, but we don't directly use
  buffer buff_bx_literals("buff_bx_literals", {RESIDENT, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
  buffer buff_border_literals("buff_border_literals", {RESIDENT, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
  buffer buff_by_literals("buff_by_literals", {RESIDENT, (int64_t)1}, p_int64, a_temporary_gpu, &blur);
  
  // access functions

  blur_input->set_access("{blur_input[r,c]->b_blur_input[r,c]}");
  bx->set_access("{bx[r,c]->b_bx_gpu[r%" + sresident + ",c]}");
  by->set_access("{by[r,c]->b_by_gpu[r%" + sresident + ",c]}");  
  by->set_wait_access("{by[r,c]->b_d2h_wait_for_by[r,0]}");
  h2d.os->set_access("{h2d[r,c]->b_blur_input_gpu[r%" + sresident + ",c]}");
  h2d.os->set_wait_access("{h2d[r,c]->b_h2d_wait[r,0]}");
  d2h.os->set_access("{d2h[r,c]->b_by[r,c]}");
  d2h.os->set_wait_access("{d2h[r,c]->b_d2h_wait[r,0]}");
    ghost.os->set_access("{ghost[r,z,c]->b_blur_input_gpu[" + sresident + "+z,c]}");
    ghost.os->set_wait_access("{ghost[r,z,c]->b_ghost_wait[r,0]}");
    border.set_access("{border[r,c]->b_bx_gpu[" + sresident + "+r%2,c]}");


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

int main() {
    global::set_default_tiramisu_options();
    global::set_loop_iterator_type(p_int64);;
    generate_single_gpu();
}
