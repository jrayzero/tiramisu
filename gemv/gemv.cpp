//
// Created by Jessica Ray on 1/17/18.
//
#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>

#include "gemv.h"

using namespace tiramisu;

// use multiple streams for each multiplication with a weight matrix

// one thread sums the entire column
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

std::vector<computation*> make_fwd_pass(std::vector<std::pair<int,int>> layer_sizes, int64_t input_rows, int64_t input_cols, function *f) {
  var c("c"), r("r"), z("z");
    // input data
    std::vector<computation*> comps;
    computation *input = new computation("{input[z,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=c<" + std::to_string(input_cols) + "}", expr(), false, p_float32, f);
    comps.push_back(input);
    int ctr = 0;
    computation *prev_weights = nullptr;
    computation *prev_gemv = nullptr;
    for (auto layer_size : layer_sizes) {
      // weight matrix for this layer
      std::string rows = std::to_string(layer_size.first);
      std::string cols = std::to_string(layer_size.second);
      computation *weights = new computation("{weights_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<" + rows + 
                                           " and 0<=c<" + cols + "}", expr(), false, p_float32, f);
      // Initializes the reduction
      computation *gemv_dummy = new computation("{gemv_dummy_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<" + rows + " and 0<=c<1}", 
                                                expr(0.0f), true, p_float32, f);
      // Does the reduction
      expr gemv_expr;
      if (prev_weights) {
        gemv_expr = expr(weights->operator()(0,r,c) * prev_gemv->operator()(z,r,c) + gemv_dummy->operator()(z,r,0));
          
      } else {
        gemv_expr = expr(weights->operator()(0,r,c) * input->operator()(z,c) + gemv_dummy->operator()(z,r,0));
      }

      computation *gemv = new computation("{gemv_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<" + rows + " and 0<=c<" + 
                                          cols + "}", gemv_expr, true, p_float32, f);
      comps.push_back(weights);
      comps.push_back(gemv_dummy);
      comps.push_back(gemv);
      // apply the activation function
      computation *activation = nullptr;
      if (ctr == layer_sizes.size() - 1) { // softmax
        computation *sum_dummy = new computation("{sum_dummy_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<1 and 0<=c<1}", 
                                                expr(0.0f), true, p_float32, f);
        expr sum_expr(expr(o_expo, gemv->operator()(z,r,c)) + sum_dummy->operator()(z,r,0));
        computation *sum = new computation("{sum_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<" + rows + " and 0<=c<1}", sum_expr, true, p_float32, f);
        expr softmax_expr(expr(o_expo, gemv->operator()(z,r,0)) / sum->operator()(z,r,0));
        activation = new computation("{softmax_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<" + rows + " and 0<=c<1}", softmax_expr, true, p_float32, f);
        comps.push_back(sum_dummy);
        comps.push_back(sum);
        comps.push_back(activation);
      } else { // sigmoid
        expr expo = expr(o_expo, -1.0f * gemv->operator()(z,r,0));
        expr sigmoid_expr = 1.0f / (1.0f + expo);
        activation = new computation("{sigmoid_" + std::to_string(ctr) + "[z,r,c]: 0<=z<" + std::to_string(input_rows) + " and 0<=r<" + rows + " and 0<=c<1}",
                             sigmoid_expr, true, p_float32, f);
        comps.push_back(activation);
      }
      
      prev_weights = weights;
      prev_gemv = gemv;
      ctr++;
    }    
    return comps;
}

void postprocess(function *f, std::string obj_file_name) {
    f->lift_dist_comps();
    f->gen_time_space_domain();
    f->gen_isl_ast();
    f->gen_halide_stmt();
    f->gen_halide_obj(obj_file_name);
    f->dump_halide_stmt();
}

void create_cpu_fwd_pass() {
  var c("c"), r("r"), r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1"), z("z");
    function *gemv_cpu_fwd = new function("gemv_cpu_fwd");
    std::vector<std::pair<int, int>> weights;
    weights.push_back(std::pair<int, int>(WEIGHTS_0, COLS));
    weights.push_back(std::pair<int, int>(WEIGHTS_1, WEIGHTS_0));
    weights.push_back(std::pair<int, int>(WEIGHTS_2, WEIGHTS_1));
    weights.push_back(std::pair<int, int>(WEIGHTS_3, WEIGHTS_2));
    int num_layers = weights.size();
    std::vector<computation*> comps = make_fwd_pass(weights, ROWS, COLS, gemv_cpu_fwd);

    buffer buff_input("buff_input", {ROWS, COLS}, p_float32, a_input, gemv_cpu_fwd);
    std::vector<buffer*> buffs;
    buffs.push_back(&buff_input);
    for (int i = 0; i < num_layers; i++) {
      std::pair<int, int> weight = weights[i];
      buffer *buff_weights = new buffer("buff_weights_" + std::to_string(i), {weight.first, weight.second}, p_float32, a_input, gemv_cpu_fwd);
      buffs.push_back(buff_weights);
      if (i == num_layers - 1) {
        buffer *b = new buffer("buff_gemv_" + std::to_string(i), {ROWS, weight.first}, p_float32, a_output, gemv_cpu_fwd);
        buffs.push_back(b);
      } else {
        new buffer("buff_gemv_" + std::to_string(i), {20, weight.first, 1}, p_float32, a_temporary, gemv_cpu_fwd);
      }
    }

    int i = 0;
    int layer = 0;
    var z1("z1"), z2("z2");
    computation *prev_weights = nullptr;
    computation *prev_gemv = nullptr;
    while (i < comps.size()) {
      if (i == 0) { // this is the input
        computation *input = comps[i];
        input->set_access("{" + input->get_name() + "[z,c]->buff_input[z,c]}");
        i++;
      } else {
        computation *weights = comps[i];
        computation *gemv_dummy = comps[i+1];
        computation *gemv = comps[i+2];
        computation *activation = nullptr;
        computation *sum_dummy = nullptr;
        computation *sum = nullptr;
        if (layer == num_layers - 1) {
          sum_dummy = comps[i+3];
          sum = comps[i+4];
          activation = comps[i+5];
          sum_dummy->split(z, 20, z1, z2);
          sum->split(z, 20, z1, z2);
          sum_dummy->tag_parallel_level(z2);
          sum->tag_parallel_level(z2);
        } else {
          activation = comps[i+3];
        }
        weights->split(z, 20, z1, z2);
        gemv_dummy->split(z, 20, z1, z2);
        gemv->split(z, 20, z1, z2);
        activation->split(z, 20, z1, z2);
        weights->tag_parallel_level(z2);
        gemv_dummy->tag_parallel_level(z2);
        gemv->tag_parallel_level(z2);
        activation->tag_parallel_level(z2);
        // order the operations
        if (prev_weights) {
          prev_gemv->before(*prev_weights, z2);
          prev_weights->before(*gemv_dummy, z2);
          gemv_dummy->before(*gemv, r);
          if (sum_dummy) {
            activation->split(r, 8, r0, r1);
            activation->tag_vector_level(r1, 8);
            gemv->before(*sum_dummy, z2);
            sum_dummy->before(*sum, z2);
            sum->before(*activation, z2);
          } else {
            gemv->before(*activation, r);
          }
        } else {
          gemv_dummy->before(*gemv, r);
          gemv->before(*activation, r);
        }
        prev_weights = weights;
        prev_gemv = activation;//gemv;
        weights->set_access("{" + weights->get_name() + "[z,r,c]->buff_weights_" + std::to_string(layer) + "[r,c]}");
        if (layer == num_layers - 1) {
          gemv_dummy->set_access("{" + gemv_dummy->get_name() + "[z,r,c]->buff_gemv_" + std::to_string(layer) + "[z,r]}");
          gemv->set_access("{" + gemv->get_name() + "[z,r,c]->buff_gemv_" + std::to_string(layer) + "[z,r]}");
          activation->set_access("{" + activation->get_name() + "[z,r,c]->buff_gemv_" + std::to_string(layer) + "[z,r]}");
        } else {
          gemv_dummy->set_access("{" + gemv_dummy->get_name() + "[z,r,c]->buff_gemv_" + std::to_string(layer) + "[z%20,r,0]}");
          gemv->set_access("{" + gemv->get_name() + "[z,r,c]->buff_gemv_" + std::to_string(layer) + "[z%20,r,0]}");
          activation->set_access("{" + activation->get_name() + "[z,r,c]->buff_gemv_" + std::to_string(layer) + "[z%20,r,0]}");
        }
        if (sum_dummy) {
          new buffer("buff_sum_" + std::to_string(layer), {20}, p_float32, a_temporary, gemv_cpu_fwd);
          sum_dummy->set_access("{" + sum_dummy->get_name() + "[z,r,c]->buff_sum_" + std::to_string(layer) + "[z%20]}");
          sum->set_access("{" + sum->get_name() + "[z,r,c]->buff_sum_" + std::to_string(layer) + "[z%20]}");
          i += 6;
        } else {
          i += 4;
        }
        layer++;
      }
    }

    gemv_cpu_fwd->set_arguments(buffs);
    postprocess(gemv_cpu_fwd, "/tmp/generated_gemv_cpu_fwd.o");
}

void create_gpu_fwd_pass() {
  var c("c"), r("r"), r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1"), z("z");
    function *gemv_gpu_fwd = new function("gemv_gpu_fwd");
    std::vector<std::pair<int, int>> weights;
    weights.push_back(std::pair<int, int>(WEIGHTS_0, COLS));
    weights.push_back(std::pair<int, int>(WEIGHTS_1, WEIGHTS_0));
    weights.push_back(std::pair<int, int>(WEIGHTS_2, WEIGHTS_1));
    weights.push_back(std::pair<int, int>(WEIGHTS_3, WEIGHTS_2));
    int num_layers = weights.size();
    std::vector<computation*> comps = make_fwd_pass(weights, ROWS, COLS, gemv_gpu_fwd);

    buffer buff_input("buff_input", {ROWS, COLS}, p_float32, a_input, gemv_gpu_fwd);
    std::vector<buffer*> buffs;
    buffs.push_back(&buff_input);

    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);
    xfer_prop h2d_cuda_sync(p_float32, {SYNC, CUDA, CPU2GPU}, 0);
    xfer_prop d2h_cuda_sync(p_float32, {SYNC, CUDA, GPU2CPU}, 0);

    // transfer weights at the beginning
    send_recv *prev_weight_xfer = nullptr;
    int w = 0;
    int w_idx = 0;
    var q("q");
    while (w < comps.size()) {
      if (w != 0 && (w+4 < comps.size())) {
        computation *_weights = comps[w];
        xfer h2d = computation::create_xfer("{weight_xfer_" + std::to_string(w_idx) + "[q,r,c]: 0<=q<1 and 0<=r<" + 
                                            std::to_string(weights[w_idx].first) + " and 0<=c<" + std::to_string(weights[w_idx].second) +
                                            "}", h2d_cuda_sync, _weights->operator()(0,r,c), gemv_gpu_fwd);        
        if (prev_weight_xfer) {
          prev_weight_xfer->before(*h2d.os, computation::root);
        }
        h2d.os->collapse_many({collapser(2, (int64_t)0, (int64_t)weights[w_idx].second), collapser(1, (int64_t)0, (int64_t)weights[w_idx].first)});

        new buffer("gpu_buff_weights_" + std::to_string(w_idx), {weights[w_idx].first, weights[w_idx].second}, p_float32, a_temporary_gpu, gemv_gpu_fwd);
        h2d.os->set_access("{" + h2d.os->get_name() + "[q,r,c]->gpu_buff_weights_" + std::to_string(w_idx) + "[r,c]}");
        prev_weight_xfer = h2d.os;
        w_idx++;
        w += 4;
      } else {
        w++;
      }
    }

    for (int i = 0; i < num_layers; i++) {
      std::pair<int, int> weight = weights[i];
      buffer *buff_weights = new buffer("buff_weights_" + std::to_string(i), {weight.first, weight.second}, p_float32, a_input, gemv_gpu_fwd);
      buffs.push_back(buff_weights);
      if (i == num_layers - 1) {
        buffer *b = new buffer("buff_gemv_" + std::to_string(i), {ROWS, weight.first}, p_float32, a_output, gemv_gpu_fwd);
        buffs.push_back(b);
        new buffer("gpu_buff_gemv_" + std::to_string(i), {ROWS, weight.first, 1}, p_float32, a_temporary_gpu, gemv_gpu_fwd);
      } else {
        new buffer("gpu_buff_gemv_" + std::to_string(i), {ROWS, weight.first, 1}, p_float32, a_temporary_gpu, gemv_gpu_fwd);
      }
    }

    // make the literal buffers
    new buffer("buff_gemv_0_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_gemv_1_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_gemv_2_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_gemv_3_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_sigmoid_0_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_sigmoid_1_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_sigmoid_2_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_sigmoid_3_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_sum_3_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);
    new buffer("buff_softmax_3_literals", {ROWS,1}, p_int64, a_temporary_gpu, gemv_gpu_fwd);

    int i = 0;
    int layer = 0;
    var z1("z1"), z2("z2");
    computation *prev_weights = nullptr;
    computation *prev_gemv = nullptr;
    send_recv *input_xfer = nullptr;
    while (i < comps.size()) {
      if (i == 0) { // this is the input
        computation *input = comps[i];
        input->set_access("{" + input->get_name() + "[z,c]->buff_input[z,c]}");
        xfer h2d = computation::create_xfer("{input_xfer[z,c]: 0<=z<" + std::to_string(ROWS) + " and 0<=c<" + std::to_string(COLS) + "}", 
                                            h2d_cuda_sync, input->operator()(z,c), gemv_gpu_fwd);        
        h2d.os->collapse_many({collapser(1, (int64_t)0, (int64_t)COLS)});
        new buffer("gpu_buff_input", {ROWS, COLS}, p_float32, a_temporary_gpu, gemv_gpu_fwd);
        h2d.os->set_access("{input_xfer[z,c]->gpu_buff_input[z,c]}");
        input_xfer = h2d.os;
        i++;
      } else {
        computation *weights = comps[i];
        computation *gemv_dummy = comps[i+1];
        gemv_dummy->set_schedule_this_comp(false);
        computation *gemv = comps[i+2];
        computation *activation = nullptr;
        computation *sum_dummy = nullptr;
        computation *sum = nullptr;
        gemv_dummy->split(r, 500, r0, r1);
        gemv->split(r, 500, r0, r1);
        if (layer == num_layers - 1) {
          sum_dummy = comps[i+3];
          sum = comps[i+4];
          activation = comps[i+5];
          sum_dummy->split(r, 500, r0, r1);
          sum->split(r, 500, r0, r1);
        } else {
          activation = comps[i+3];
        }
        activation->split(r, 500, r0, r1);
        // order the operations
        if (prev_weights) {
          prev_gemv->before(*prev_weights, z);
          prev_weights->before(*gemv_dummy, z);
          gemv_dummy->before(*gemv, z);
          generator::update_producer_expr_name(gemv, "weights_" + std::to_string(layer), "weight_xfer_" + std::to_string(layer));
          if (sum_dummy) { // last layer
            gemv->before(*sum_dummy, z);
            sum_dummy->before(*sum, z);
            sum_dummy->set_schedule_this_comp(false);
            sum->before(*activation, z);
            // transfer back the results
            xfer d2h = computation::create_xfer("{result_xfer[z,r]: 0<=z<" + std::to_string(ROWS) + " and 0<=r<" + std::to_string(WEIGHTS_3) + "}", 
                                                d2h_cuda_sync, activation->operator()(z,r,0), gemv_gpu_fwd);        
            d2h.os->set_access("{result_xfer[z,r]->buff_gemv_3[z,r]}");
            activation->before(*d2h.os, z);
            d2h.os->collapse_many({collapser(1, (int64_t)0, (int64_t)WEIGHTS_3)});
            d2h.os->set_schedule_this_comp(false);

          } else {
            gemv->before(*activation, z);
          }
        } else {
          generator::update_producer_expr_name(gemv, "weights_" + std::to_string(layer), "weight_xfer_" + std::to_string(layer));
          generator::update_producer_expr_name(gemv, "input", "input_xfer");
          
          prev_weight_xfer->before(*input_xfer, computation::root);
          input_xfer->before(*gemv_dummy, z);//computation::root);
          gemv_dummy->before(*gemv, r1);
          gemv->before(*activation, z);
        }
        if (prev_weights) {
          if (sum_dummy) {
            gemv->tag_gpu_level2(r0, r1, 0);
            sum->split(r0, 2, r2, r3);
            sum->tag_gpu_level2(r2, r3, 0);
            activation->tag_gpu_level2(r0, r1, 0);          
          } else {
            gemv->tag_gpu_level2(r0, r1, 0);
            activation->tag_gpu_level2(r0, r1, 0);          
          }
        } else {
          gemv->tag_gpu_level2(r0, r1, 0);
          activation->tag_gpu_level2(r0, r1, 0);          
        }

        prev_weights = weights;
        prev_gemv = activation;
        weights->set_access("{" + weights->get_name() + "[z,r,c]->buff_weights_" + std::to_string(layer) + "[r,c]}");
        if (layer == num_layers - 1) {
          gemv_dummy->set_access("{" + gemv_dummy->get_name() + "[z,r,c]->gpu_buff_gemv_" + std::to_string(layer) + "[z,r,0]}");
          gemv->set_access("{" + gemv->get_name() + "[z,r,c]->gpu_buff_gemv_" + std::to_string(layer) + "[z,r,0]}");
          activation->set_access("{" + activation->get_name() + "[z,r,c]->gpu_buff_gemv_" + std::to_string(layer) + "[z,r,0]}");
        } else {
          gemv_dummy->set_access("{" + gemv_dummy->get_name() + "[z,r,c]->gpu_buff_gemv_" + std::to_string(layer) + "[z,r,0]}");
          gemv->set_access("{" + gemv->get_name() + "[z,r,c]->gpu_buff_gemv_" + std::to_string(layer) + "[z,r,0]}");
          activation->set_access("{" + activation->get_name() + "[z,r,c]->gpu_buff_gemv_" + std::to_string(layer) + "[z,r,0]}");
        }
        if (sum_dummy) {
          new buffer("gpu_buff_sum_" + std::to_string(layer), {ROWS}, p_float32, a_temporary_gpu, gemv_gpu_fwd);
          sum_dummy->set_access("{" + sum_dummy->get_name() + "[z,r,c]->gpu_buff_sum_" + std::to_string(layer) + "[z]}");
          sum->set_access("{" + sum->get_name() + "[z,r,c]->gpu_buff_sum_" + std::to_string(layer) + "[z]}");
          i += 6;
        } else {
          i += 4;
        }
        layer++;
      }
    }

    gemv_gpu_fwd->set_arguments(buffs);
    postprocess(gemv_gpu_fwd, "/tmp/generated_gemv_gpu_fwd.o");
    compile_kernels_to_obj();
}

// This does the multiply, then the reduction separately. You could schedule them together to get the original algorithm if you wanted
std::vector<computation *> make_alternate_algorithm(function *f, int64_t rows, int64_t cols) {
    var c("c"), r("r");
    computation *vector = new computation("{vector[r,c]: 0<=r<1 and 0<=c<" + std::to_string(cols) + "}", expr(), false, p_float32, f);
    computation *matrix = new computation("{matrix[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                          expr(), false, p_float32, f);
    computation *multiply = new computation("{multiply[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                            expr(matrix->operator()(r,c) * vector->operator()(0,c)),
                                            true, p_float32, f);
    computation *gemv_dummy = new computation("{gemv_dummy[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<1}", expr(), //multiply->operator()(r,c),
                                              false, p_float32, f);
    computation *sum = new computation("{sum[r,c]: 0<=r<" + std::to_string(rows) + " and 0<=c<" + std::to_string(cols) + "}",
                                       multiply->operator()(r,c) + gemv_dummy->operator()(r,0),
                                       true, p_float32, f);
    return {vector, matrix, multiply, gemv_dummy, sum};
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
    gemv_dummy->before(*gemv, r2);
#endif

    gemv_dummy->before(*gemv, computation::root);

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

    int64_t rows_resident_on_gpu = 2000;
    int64_t threads_per_block = 1000;

    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    xfer_prop h2d_cuda_sync(p_float32, {SYNC, CUDA, CPU2GPU}, -1);
    xfer_prop h2d_cuda_async_alt(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 1);
    xfer_prop d2h_cuda_sync(p_float32, {SYNC, CUDA, GPU2CPU}, 1);



    // just do this all at once at the beginning
    xfer vector_copy = computation::create_xfer("{vector_copy[r,c]: 0<=r<1 and 0<=c<" + std::to_string(COLS) + "}", h2d_cuda_sync,
                                                vector->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "vector", "vector_copy", false);
    // copy up a big chunk, then run the kernel, then copy back. No overlap with kernel, but can have overlap with sending and receiving.
    xfer matrix_row_copy = computation::create_xfer("{matrix_row_copy[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<" +
                                                    std::to_string(COLS) + "}", h2d_cuda_sync,
                                                    matrix->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "matrix", "matrix_row_copy", false);
    xfer init_reduction = computation::create_xfer("{init_reduction[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", h2d_cuda_sync, gemv_dummy->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "gemv_dummy", "init_reduction", false);

    xfer copy_back_results = computation::create_xfer("{copy_back[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", d2h_cuda_sync, gemv->operator()(r,c), gemv_gpu);

    int64_t block_size = 10;
    matrix_row_copy.os->split(r, rows_resident_on_gpu, r0, r1);
    gemv->split(r, rows_resident_on_gpu, r0, r1);
    gemv->split(r1, block_size, r2, r3);
    gemv_dummy->split(r, rows_resident_on_gpu, r0, r1);
    gemv_dummy->split(r1, block_size, r2, r3);
    init_reduction.os->split(r, rows_resident_on_gpu, r0, r1);
    init_reduction.os->split(r1, block_size, r2, r3);
    copy_back_results.os->split(r, rows_resident_on_gpu, r0, r1);
    copy_back_results.os->split(r1, block_size, r2, r3);

    vector_copy.os->before(*matrix_row_copy.os, computation::root);
    matrix_row_copy.os->before(*gemv_dummy, r0);
    gemv_dummy->before(*init_reduction.os, r0);
    init_reduction.os->before(*gemv, r0);
    gemv->before(*copy_back_results.os, r0);

    vector_copy.os->collapse_many({collapser(1, (int64_t)0, COLS)});
    matrix_row_copy.os->collapse_many({collapser(2, (int64_t)0, COLS), collapser(1, (int64_t)0, rows_resident_on_gpu)});
    init_reduction.os->collapse_many({collapser(2, (int64_t)0, block_size), collapser(1, (int64_t)0, rows_resident_on_gpu/block_size)});
    copy_back_results.os->collapse_many({collapser(2, (int64_t)0, block_size), collapser(1, (int64_t)0, rows_resident_on_gpu/block_size)});

    // this has to go after all the other things have been scheduled
    gemv->tag_gpu_level2(r2, r3, -1);

    buffer vector_buff("vector_buff", {1,COLS}, p_float32, a_input, gemv_gpu);
    buffer matrix_buff("matrix_buff", {ROWS, COLS}, p_float32, a_input, gemv_gpu);
    buffer result_buff("result_buff", {ROWS,1}, p_float32, a_output, gemv_gpu);
    buffer zero_buff("zero_buff", {rows_resident_on_gpu,1}, p_float32, a_temporary, gemv_gpu);
    buffer vector_gpu_buff("vector_gpu_buff", {1,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer matrix_gpu_buff("matrix_gpu_buff", {rows_resident_on_gpu,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // copy up one row at a time
    buffer result_gpu_buff("result_gpu_buff", {rows_resident_on_gpu,1}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer buff_bx_literals("buff_gemv_literals", {ROWS, 3}, p_int64, tiramisu::a_temporary_gpu, gemv_gpu);
    buffer null_buffer("null_buffer", {1}, p_wait_ptr, tiramisu::a_temporary, gemv_gpu);

    buffer matrix_gpu_wait_buff("matrix_gpu_wait_buff", {ROWS/rows_resident_on_gpu}, p_wait_ptr, a_temporary, gemv_gpu); //copy up chunks of whole rows (ROWS/rows_resident gives # chunks)
    buffer init_reduc_wait_buff("init_reduc_wait_buff", {ROWS/rows_resident_on_gpu}, p_wait_ptr, a_temporary, gemv_gpu);

    vector->set_access("{vector[r,c]->vector_buff[r,c]}");
    vector_copy.os->set_access("{vector_copy[r,c]->vector_gpu_buff[r,c]}");
    matrix->set_access("{matrix[r,c]->matrix_buff[r,c]}");
    matrix_row_copy.os->set_access("{matrix_row_copy[r,c]->matrix_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    matrix_row_copy.os->set_wait_access("{matrix_row_copy[r,c]->matrix_gpu_wait_buff[r]}");
    gemv_dummy->set_access("{gemv_dummy[r,c]->zero_buff[r%" + std::to_string(rows_resident_on_gpu) + ",0]}");
    init_reduction.os->set_access("{init_reduction[r,c]->result_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    init_reduction.os->set_wait_access("{init_reduction[r,c]->init_reduc_wait_buff[r%" + std::to_string(rows_resident_on_gpu) + "]}");
    gemv->set_access("{gemv[r,c]->result_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",0]}");
    copy_back_results.os->set_access("{copy_back[r,c]->result_buff[r,c]}");
    copy_back_results.os->set_wait_access("{copy_back[r,c]->null_buffer[0]}");

    gemv_gpu->set_arguments({&vector_buff, &matrix_buff, &result_buff});
    postprocess(gemv_gpu, "/tmp/generated_gemv.o");
    print_tiramisu_cuda_runtime();
    compile_kernels_to_obj();
}

void create_gpu_alternate_version() {
    var c("c"), r("r"), r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1");
    function *gemv_gpu = new function("gemv_gpu");
    std::vector<computation *> comps = make_alternate_algorithm(gemv_gpu, ROWS, COLS);
    computation *vector = comps[0];
    computation *matrix = comps[1];
    computation *multiply = comps[2];
    computation *gemv_dummy = comps[3];
    computation *sum = comps[4];

    int64_t rows_resident_on_gpu = 2000;//12500;
    int64_t threads_per_block = 1000;

    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 1);
    xfer_prop h2d_cuda_sync(p_float32, {SYNC, CUDA, CPU2GPU}, -1);
    xfer_prop h2d_cuda_async_alt(p_float32, {ASYNC, CUDA, CPU2GPU}, 0);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 1);
    xfer_prop d2h_cuda_sync(p_float32, {SYNC, CUDA, GPU2CPU}, 1);
    xfer_prop stream0(p_float32, {CUDA}, 0);
    xfer_prop stream1(p_float32, {CUDA}, 1);

    // just do this all at once at the beginning
    computation output_init("{output_init[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", expr(), false, p_float32, gemv_gpu);
    xfer zero_out = computation::create_xfer("{zero_out[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", h2d_cuda_sync, output_init(r,c), gemv_gpu);
    xfer vector_copy = computation::create_xfer("{vector_copy[r,c]: 0<=r<1 and 0<=c<" + std::to_string(COLS) + "}", h2d_cuda_async,
                                                vector->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(multiply, "vector", "vector_copy", false);
    tiramisu::wait vector_copy_wait("{vector_copy_wait[r]: 0<=r<1}", vector_copy.os->operator()(0,0), h2d_cuda_async, true, gemv_gpu);
    // copy up a big chunk, then run the kernel, then copy back. No overlap with kernel, but can have overlap with sending and receiving.
    xfer matrix_row_copy = computation::create_xfer("{matrix_row_copy[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<" +
                                                    std::to_string(COLS) + "}", h2d_cuda_async,
                                                    matrix->operator()(r,c), gemv_gpu);
    tiramisu::wait matrix_row_copy_wait("{matrix_row_copy_wait[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}",
                                        matrix_row_copy.os->operator()(r,c), stream0, true, gemv_gpu);
    generator::update_producer_expr_name(multiply, "matrix", "matrix_row_copy", false);

    tiramisu::wait sum_wait("{sum_wait[r,c]: 0<=r<" + std::to_string(ROWS/rows_resident_on_gpu) + " and 0<=c<1}",
                            sum->operator()(0,0), stream1, true, gemv_gpu); // just wait on the last one
    xfer copy_back_results = computation::create_xfer("{copy_back[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}",
                                                      d2h_cuda_sync, sum->operator()(r,c), gemv_gpu);

    multiply->split(c, threads_per_block, c0, c1);
    sum->split(r, rows_resident_on_gpu, r0, r1);
    sum->split(r1, threads_per_block, r2, r3);

    matrix_row_copy.os->split(r, rows_resident_on_gpu, r0, r1);
    matrix_row_copy_wait.split(r, rows_resident_on_gpu, r0, r1);
    multiply->split(r, rows_resident_on_gpu, r0, r1);
    gemv_dummy->split(r, rows_resident_on_gpu, r0, r1);

    zero_out.os->before(*vector_copy.os, computation::root);
    vector_copy.os->before(vector_copy_wait, computation::root);
    vector_copy_wait.before(*matrix_row_copy.os, computation::root);
    matrix_row_copy.os->before(matrix_row_copy_wait, r1);//computation::root);
    matrix_row_copy_wait.before(*multiply, r1);
    multiply->before(*gemv_dummy, r1);
    gemv_dummy->before(*sum, r0);
    sum->before(sum_wait, r);
    sum_wait.before(*copy_back_results.os, computation::root);

    zero_out.os->collapse_many({collapser(0, (int64_t)0, ROWS)});
    vector_copy.os->collapse_many({collapser(1, (int64_t)0, COLS)});
    matrix_row_copy.os->collapse_many({collapser(2, (int64_t)0, COLS)});//, collapser(1, (int64_t)0, rows_resident_on_gpu)});
    copy_back_results.os->collapse_many({collapser(0, (int64_t)0, ROWS)});//, collapser(1, (int64_t)0, rows_resident_on_gpu/block_size)});

    multiply->tag_gpu_level2(c0, c1, 0);
    sum->tag_gpu_level2(r2, r3, 0);

    buffer vector_buff("vector_buff", {1,COLS}, p_float32, a_input, gemv_gpu);
    buffer matrix_buff("matrix_buff", {ROWS, COLS}, p_float32, a_input, gemv_gpu);
    buffer result_buff("result_buff", {ROWS,1}, p_float32, a_output, gemv_gpu);
    buffer vector_gpu_buff("vector_gpu_buff", {1,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer matrix_gpu_buff("matrix_gpu_buff", {rows_resident_on_gpu,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // copy up one row at a time
    buffer multiply_gpu_buff("multiply_gpu_buff", {rows_resident_on_gpu,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // copy up one row at a time
    buffer result_gpu_buff("result_gpu_buff", {ROWS,1}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer buff_multiply_literals("buff_multiply_literals", {ROWS, 1}, p_int64, tiramisu::a_temporary_gpu, gemv_gpu);
    buffer buff_sum_literals("buff_sum_literals", {ROWS, 3}, p_int64, tiramisu::a_temporary_gpu, gemv_gpu);
    buffer null_buffer("null_buffer", {1}, p_wait_ptr, tiramisu::a_temporary, gemv_gpu);
    buffer sum_wait_buff("sum_wait_buff", {ROWS / rows_resident_on_gpu, 1}, p_wait_ptr, tiramisu::a_temporary, gemv_gpu);
    buffer matrix_gpu_wait_buff("matrix_gpu_wait_buff", {ROWS}, p_wait_ptr, a_temporary, gemv_gpu); //copy up chunks of whole rows (ROWS/rows_resident gives # chunks)
    buffer zero_buff("zero_buff", {ROWS,1}, p_float32, a_input, gemv_gpu);

    output_init.set_access("{output_init[r,c]->zero_buff[r,c]}");
    zero_out.os->set_access("{zero_out[r,c]->result_gpu_buff[r,c]}");
    vector->set_access("{vector[r,c]->vector_buff[r,c]}");
    vector_copy.os->set_access("{vector_copy[r,c]->vector_gpu_buff[r,c]}");
    vector_copy.os->set_wait_access("{vector_copy[r,c]->null_buffer[0]}");
    matrix->set_access("{matrix[r,c]->matrix_buff[r,c]}");
    matrix_row_copy.os->set_access("{matrix_row_copy[r,c]->matrix_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    matrix_row_copy.os->set_wait_access("{matrix_row_copy[r,c]->matrix_gpu_wait_buff[r]}");
    gemv_dummy->set_access("{gemv_dummy[r,c]->result_gpu_buff[r,0]}");
    multiply->set_access("{multiply[r,c]->multiply_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    sum->set_access("{sum[r,c]->result_gpu_buff[r,0]}");
    sum->set_wait_access("{sum[r,c]->sum_wait_buff[0,0]}");
    copy_back_results.os->set_access("{copy_back[r,c]->result_buff[r,c]}");
    copy_back_results.os->set_wait_access("{copy_back[r,c]->null_buffer[0]}");

    gemv_gpu->set_arguments({&vector_buff, &matrix_buff, &result_buff, &zero_buff});
    postprocess(gemv_gpu, "/tmp/generated_gemv.o");
    print_tiramisu_cuda_runtime();
    compile_kernels_to_obj();
}


void create_gpu_version_with_shared() {
    var c("c"), r("r"), r0("r0"), r1("r1"), r2("r2"), r3("r3"), c0("c0"), c1("c1");
    function *gemv_gpu = new function("gemv_gpu");
    std::vector<computation *> comps = make_algorithm(gemv_gpu, ROWS, COLS);
    computation *vector = comps[0];
    computation *matrix = comps[1];
    computation *gemv_dummy = comps[2];
    computation *gemv = comps[3];

    int64_t rows_resident_on_gpu = 5000;
    int64_t threads_per_block = 1024;

    xfer_prop h2d_cuda_async(p_float32, {ASYNC, CUDA, CPU2GPU}, 1);
    xfer_prop d2h_cuda_async(p_float32, {ASYNC, CUDA, GPU2CPU}, 2);
    xfer_prop kernel(p_float32, {ASYNC, CUDA, GPU2CPU}, 0);


    xfer vector_copy = computation::create_xfer("{vector_copy[r,c]: 0<=r<1 and 0<=c<" + std::to_string(COLS) + "}", h2d_cuda_async,
                                                vector->operator()(r,c), gemv_gpu);
    generator::update_producer_expr_name(gemv, "vector", "vector_copy", false);
    // copy up a big chunk, then run the kernel, then copy back. No overlap with kernel, but can have overlap with sending and receiving.
    xfer matrix_row_copy = computation::create_xfer("{matrix_row_copy[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<" +
                                                    std::to_string(COLS) + "}", h2d_cuda_async,
                                                    matrix->operator()(r,c), gemv_gpu);
    tiramisu::wait matrix_row_copy_wait("{matrix_row_copy_wait[r,c]: 0<=r<" + std::to_string(ROWS/rows_resident_on_gpu) + " and 0<=c<1}", matrix_row_copy.os->operator()(r*rows_resident_on_gpu,0), kernel, true, gemv_gpu);
    generator::update_producer_expr_name(gemv, "matrix", "matrix_row_copy", false);
    xfer init_reduction = computation::create_xfer("{init_reduction[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", h2d_cuda_async, gemv_dummy->operator()(r,c), gemv_gpu);
    init_reduction.os->set_schedule_this_comp(false);
    generator::update_producer_expr_name(gemv, "gemv_dummy", "init_reduction", false);

    tiramisu::wait wait_gemv("{wait_gemv[r,c]: 0<=r<" + std::to_string(ROWS/rows_resident_on_gpu) + " and 0<=c<1}", gemv->operator()(r*rows_resident_on_gpu,0), kernel, true, gemv_gpu);

    xfer copy_back_results = computation::create_xfer("{copy_back[r,c]: 0<=r<" + std::to_string(ROWS) + " and 0<=c<1}", d2h_cuda_async, gemv->operator()(r,c), gemv_gpu);
    matrix_row_copy.os->split(r, rows_resident_on_gpu, r0, r1);
    gemv->split(r, rows_resident_on_gpu, r0, r1);
    gemv->split(c, COLS/threads_per_block, c0, c1);
    gemv_dummy->split(r, rows_resident_on_gpu, r0, r1);
    gemv_dummy->split(c, COLS/threads_per_block, c0, c1);
    init_reduction.os->split(r, rows_resident_on_gpu, r0, r1);
    init_reduction.os->split(c, threads_per_block, c0, c1);
    //    matrix_row_copy_wait.split(r, rows_resident_on_gpu, r0, r1);
    matrix_row_copy_wait.split(c, threads_per_block, c0, c1);
    copy_back_results.os->split(r, rows_resident_on_gpu, r0, r1);

    vector_copy.os->before(*matrix_row_copy.os, computation::root);
    matrix_row_copy.os->before(*gemv_dummy, r0);
    gemv_dummy->before(*init_reduction.os, r0);
    init_reduction.os->before(matrix_row_copy_wait, r); 
    matrix_row_copy_wait.before(*gemv, r0);
    gemv->before(wait_gemv, r);
    wait_gemv.before(*copy_back_results.os, r0);

    vector_copy.os->collapse_many({collapser(1, (int64_t)0, COLS)});
    matrix_row_copy.os->collapse_many({collapser(2, (int64_t)0, COLS), collapser(1, (int64_t)0, rows_resident_on_gpu)});
    init_reduction.os->collapse_many({collapser(1, (int64_t)0, rows_resident_on_gpu)});//, collapser(1, (int64_t)0, rows_resident_on_gpu/threads_per_block)});
    copy_back_results.os->collapse_many({collapser(1, (int64_t)0, rows_resident_on_gpu)});//, collapser(1, (int64_t)0, rows_resident_on_gpu)});

    gemv_dummy->set_schedule_this_comp(false);
    // this has to go after all the other things have been scheduled
    gemv->tag_gpu_level2(r1, c0, 0);


    buffer vector_buff("vector_buff", {1,COLS}, p_float32, a_input, gemv_gpu);
    buffer matrix_buff("matrix_buff", {ROWS, COLS}, p_float32, a_input, gemv_gpu);
    buffer result_buff("result_buff", {ROWS,1}, p_float32, a_output, gemv_gpu);
    buffer zero_buff("zero_buff", {rows_resident_on_gpu,1}, p_float32, a_temporary, gemv_gpu);
    buffer vector_gpu_buff("vector_gpu_buff", {1,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer matrix_gpu_buff("matrix_gpu_buff", {rows_resident_on_gpu,COLS}, p_float32, a_temporary_gpu, gemv_gpu); // copy up one row at a time
    buffer result_gpu_buff("result_gpu_buff", {rows_resident_on_gpu,1}, p_float32, a_temporary_gpu, gemv_gpu); // should fully fit on gpu
    buffer buff_bx_literals("buff_gemv_literals", {ROWS, 3}, p_int64, tiramisu::a_temporary_gpu, gemv_gpu);
    buffer null_buffer("null_buffer", {1}, p_wait_ptr, tiramisu::a_temporary, gemv_gpu);

    buffer matrix_gpu_wait_buff("matrix_gpu_wait_buff", {ROWS}, p_wait_ptr, a_temporary, gemv_gpu); //copy up chunks of whole rows (ROWS/rows_resident gives # chunks)
    buffer init_reduc_wait_buff("init_reduc_wait_buff", {ROWS/rows_resident_on_gpu}, p_wait_ptr, a_temporary, gemv_gpu);
    buffer kernel_wait_buff("kernel_wait_buff", {ROWS/rows_resident_on_gpu}, p_wait_ptr, a_temporary, gemv_gpu);

    vector->set_access("{vector[r,c]->vector_buff[r,c]}");
    vector_copy.os->set_access("{vector_copy[r,c]->vector_gpu_buff[r,c]}");
    vector_copy.os->set_wait_access("{vector_copy[r,c]->null_buffer[0]}");
    matrix->set_access("{matrix[r,c]->matrix_buff[r,c]}");
    matrix_row_copy.os->set_access("{matrix_row_copy[r,c]->matrix_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    matrix_row_copy.os->set_wait_access("{matrix_row_copy[r,c]->matrix_gpu_wait_buff[r]}");
    gemv_dummy->set_access("{gemv_dummy[r,c]->zero_buff[r%" + std::to_string(rows_resident_on_gpu) + ",0]}");
    init_reduction.os->set_access("{init_reduction[r,c]->result_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",c]}");
    init_reduction.os->set_wait_access("{init_reduction[r,c]->init_reduc_wait_buff[r%" + std::to_string(rows_resident_on_gpu) + "]}");
    gemv->set_access("{gemv[r,c]->result_gpu_buff[r%" + std::to_string(rows_resident_on_gpu) + ",0]}");
    gemv->set_wait_access("{gemv[r,c]->kernel_wait_buff[0]}");
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
#ifdef FWD_PASS
    create_cpu_fwd_pass();
#else
    create_cpu_version();
#endif
#elif defined(GPU)
#ifdef FWD_PASS
    create_gpu_fwd_pass();
#else
    create_gpu_version_with_shared();
#endif
#endif

    return 0;

}
