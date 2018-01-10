#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/constraint.h>
#include <isl/space.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <string>
#include "../include/tiramisu/expr.h"
#include "../include/tiramisu/debug.h"
#include "../include/tiramisu/core.h"

// TODO
// 0a. Generate kernel code...
// 1. Initialize context and all of that (that would go in the wrapper function that is called from the Halide IR)
// 2. Launch kernel
// make sure having multiple computations in the same loop nest work. Both will need to be converted into kernels
namespace tiramisu {

std::vector<std::string> closure_buffers;
std::vector<std::string> closure_buffers_no_type;
// vars to pass into the wrapper
std::vector<std::string> closure_vars;
std::vector<std::string> closure_vars_no_type;

std::string cuda_expr_from_isl_ast_expr(isl_ast_expr *isl_expr, int kernel_starting_level, int kernel_ending_level,
                                        bool convert_to_loop_type = false, bool map_iterator = false,
                                        bool capture_vars = false);

std::string cuda_headers() {
    std::string includes = "#include <cuda.h>\n";
    //    includes += "#include <iostream>\n";
    //    includes += "#include <cassert>\n";
    includes += "#include <assert.h>\n";
    includes += "#include <stdio.h>\n";
    includes += "#include \"tiramisu/tiramisu_cuda_runtime.h\"\n";
    includes += "#include \"HalideRuntime.h\"\n";
    return includes;
}

tiramisu::expr generator::replace_original_indices_with_gpu_indices(tiramisu::computation *comp, tiramisu::expr exp,
                                                                    std::map<std::string, isl_ast_expr *> iterators_map) {

    tiramisu::expr output_expr;
    if (exp.get_expr_type() == tiramisu::e_val) {
        output_expr = exp;
    } else if (exp.get_expr_type() == tiramisu::e_var) {
        std::map<std::string, isl_ast_expr *>::iterator it;
        it = iterators_map.find(exp.get_name());
        if (it != iterators_map.end()) {
            // figure out how this var maps to the computation level
            //        int dim = comp->get_loop_level_numbers_from_dimension_names({exp.get_name()})[0];
            // this is the transformed name for the iterator
            std::string dim_name = tiramisu_expr_from_isl_ast_expr(iterators_map[exp.get_name()]).get_name();
            dim_name = dim_name.substr(1);
            int dim = std::stoi(dim_name);
            dim /= 2;
            std::pair<int, int> range = comp->fct->gpu_ranges[comp->get_name()];
            int kernel_starting_level = range.first;
            int kernel_ending_level = range.second;
            std::string new_iter_name = "";
            if (dim >= kernel_starting_level && dim <= kernel_ending_level) { // this should be converted to a GPU variable
                if (kernel_ending_level - kernel_starting_level == 1) { // 1 dimension
                    if (dim == kernel_ending_level) {
                        new_iter_name = "thread_x";
                    } else {
                        new_iter_name = "block_x";
                    }
                } else if (kernel_ending_level - kernel_starting_level == 3) { // 2 dimension
                    if (dim == kernel_ending_level) {
                        new_iter_name = "thread_x";
                    } else if (dim == kernel_ending_level - 1) {
                        new_iter_name = "thread_y";
                    } else if (dim == kernel_ending_level - 2) {
                        new_iter_name = "block_x";
                    } else {
                        new_iter_name = "block_y";
                    }
                } else if (kernel_ending_level - kernel_starting_level == 5) { // 3 dimension
                    if (dim == kernel_ending_level) {
                        new_iter_name = "thread_x";
                    } else if (dim == kernel_ending_level - 1) {
                        new_iter_name = "thread_y";
                    } else if (dim == kernel_ending_level - 2) {
                        new_iter_name = "thread_z";
                    } else if (dim == kernel_ending_level - 3) {
                        new_iter_name = "block_x";
                    } else if (dim == kernel_ending_level - 4) {
                        new_iter_name = "block_y";
                    } else {
                        new_iter_name = "block_z";
                    }
                }
                tiramisu::var v(new_iter_name);
                output_expr = tiramisu::expr(tiramisu::o_cast, global::get_loop_iterator_data_type(), v);
            } else {
                assert(false && "Need to add a closure var if this comes up, which this will come up if you have non-GPU loops outside the GPU loops ");
                //          output_expr = tiramisu::expr(tiramisu::o_cast, global::get_loop_iterator_data_type(),
                //                                       tiramisu_expr_from_isl_ast_expr(iterators_map[exp.get_name()], true));
            }
        } else {
            output_expr = exp;
        }
    }
    else if ((exp.get_expr_type() == tiramisu::e_op) && (exp.get_op_type() == tiramisu::o_access || exp.get_op_type() == tiramisu::o_address_of)) {
        DEBUG(10, tiramisu::str_dump("Replacing the occurrences of original iterators in an o_access."));

        for (const auto &access : exp.get_access()) {
            generator::replace_original_indices_with_gpu_indices(comp, access, iterators_map);
        }

        output_expr = exp;
    } else if (exp.get_expr_type() == tiramisu::e_op) {
        DEBUG(10, tiramisu::str_dump("Replacing iterators in an e_op."));

        tiramisu::expr exp2, exp3, exp4;
        std::vector<tiramisu::expr> new_arguments;

        switch (exp.get_op_type()) {
            case tiramisu::o_minus:
            case tiramisu::o_logical_not:
            case tiramisu::o_floor:
            case tiramisu::o_sin:
            case tiramisu::o_cos:
            case tiramisu::o_tan:
            case tiramisu::o_asin:
            case tiramisu::o_acos:
            case tiramisu::o_atan:
            case tiramisu::o_abs:
            case tiramisu::o_sqrt:
            case tiramisu::o_expo:
            case tiramisu::o_log:
            case tiramisu::o_ceil:
            case tiramisu::o_round:
            case tiramisu::o_trunc:
            case tiramisu::o_address:
            case tiramisu::o_is_nan:
            case tiramisu::o_bitwise_not:
                exp2 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(0), iterators_map);
                output_expr = tiramisu::expr(exp.get_op_type(), exp2);
                break;
            case tiramisu::o_cast:
                exp2 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(0), iterators_map);
                output_expr = expr(exp.get_op_type(), exp.get_data_type(), exp2);
                break;
            case tiramisu::o_logical_and:
            case tiramisu::o_logical_or:
            case tiramisu::o_sub:
            case tiramisu::o_add:
            case tiramisu::o_max:
            case tiramisu::o_min:
            case tiramisu::o_mul:
            case tiramisu::o_div:
            case tiramisu::o_mod:
            case tiramisu::o_le:
            case tiramisu::o_lt:
            case tiramisu::o_ge:
            case tiramisu::o_gt:
            case tiramisu::o_eq:
            case tiramisu::o_ne:
            case tiramisu::o_right_shift:
            case tiramisu::o_left_shift:
            case tiramisu::o_bitwise_and:
            case tiramisu::o_bitwise_or:
            case tiramisu::o_bitwise_xor:
            case tiramisu::o_pow:
                exp2 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(0), iterators_map);
                exp3 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(1), iterators_map);
                output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3);
                break;
            case tiramisu::o_select:
            case tiramisu::o_cond:
            case tiramisu::o_lerp:
                exp2 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(0), iterators_map);
                exp3 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(1), iterators_map);
                exp4 = generator::replace_original_indices_with_gpu_indices(comp, exp.get_operand(2), iterators_map);
                output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3, exp4);
                break;
            case tiramisu::o_call:
                for (const auto &e : exp.get_arguments()) {
                    exp2 = generator::replace_original_indices_with_gpu_indices(comp, e, iterators_map);
                    new_arguments.push_back(exp2);
                }
                output_expr = tiramisu::expr(o_call, exp.get_name(), new_arguments, exp.get_data_type());
                break;
            case tiramisu::o_allocate:
            case tiramisu::o_free:
            case tiramisu::o_type:
                output_expr = exp;
                break;
            default:
                tiramisu::error("Unsupported tiramisu expression passed to generator::replace_original_indices_with_gpu_indices().", 1);
        }
    }

    return output_expr;
}


// generate a kernel from the original isl node
std::pair<std::string, std::string> generator::codegen_kernel_body(function &fct, isl_ast_node *node, int current_level,
                                                                   int kernel_starting_level, int kernel_ending_level) {
    if (isl_ast_node_get_type(node) == isl_ast_node_block) {
        // I think blocks would combine multiple computations in the loop
        std::cerr << "WTF is an isl block" << std::endl;
        exit(29);
    } else if (isl_ast_node_get_type(node) == isl_ast_node_for) {
        std::string code = "";
        if (current_level > kernel_ending_level) {
            std::cerr << "Generating for loop in kernel" << std::endl;
            assert(false && "Support for loops in kernels once we actually need them");
        } else { // this is a GPU kernel loop, i.e. a thread iterator (or block iterator, w/e)
            // Figure out the new name for this iterator based on blocks and threads
            // and then convert to the appropriate coordinate
            isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
            char *cstr = isl_ast_expr_to_C_str(iter);
            std::string iterator_str = std::string(cstr);
            std::string idx_computation = "";
            std::string cuda_dim = "";

            isl_ast_expr *init = isl_ast_node_for_get_init(node);
            isl_ast_expr *cond = isl_ast_node_for_get_cond(node);

            isl_ast_expr *cond_upper_bound_isl_format = NULL;
            if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt) {
                cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
            } else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le) {
                // Create an expression of "1".
                isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
                // Add 1 to the ISL ast upper bound to transform it into a strinct bound.
                cond_upper_bound_isl_format = isl_ast_expr_add(
                        isl_ast_expr_get_op_arg(cond, 1),
                        isl_ast_expr_from_val(one));
            } else {
                tiramisu::error("The for loop upper bound is not an isl_est_expr of type le or lt" , 1);
            }
            std::string cuda_extent =
                    cuda_expr_from_isl_ast_expr(cond_upper_bound_isl_format, kernel_starting_level, kernel_ending_level, true, false, true);

            std::string cuda_init = cuda_expr_from_isl_ast_expr(init, kernel_starting_level, kernel_ending_level, true, false, true);
            std::string cuda_loop_extent = "(" + cuda_extent + " - " + cuda_init + ")";

            isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

            if (kernel_ending_level - kernel_starting_level == 1) { // 1 dimension
                if (current_level == kernel_ending_level) {
                    idx_computation = "int thread_x = threadIdx.x;\n";
                    cuda_dim = "\n  int block_width = " + cuda_loop_extent + ";\n  int block_height = 1;\n  int block_depth = 1;\n";
                } else {
                    idx_computation = "int block_x = blockIdx.x;\n";//blockIdx.x;\n";
                    cuda_dim = "\n  int grid_width = " + cuda_loop_extent + ";\n  int grid_height = 1;\n  int grid_depth = 1;\n";
                }
            } else if (kernel_ending_level - kernel_starting_level == 3) { // 2 dimension
                if (current_level == kernel_ending_level) {
                    idx_computation = "int thread_x = threadIdx.x;\n";
                } else if (current_level == kernel_ending_level - 1) {
                    idx_computation = "int thread_y = threadIdx.y;\n";
                } else if (current_level == kernel_ending_level - 2) {
                    idx_computation = "int block_x = blockIdx.x;\n";
                } else {
                    idx_computation = "int block_y = blockIdx.y;\n";
                }
            } else if (kernel_ending_level - kernel_starting_level == 5) { // 3 dimension
                if (current_level == kernel_ending_level) {
                    idx_computation = "int thread_x = threadIdx.x;\n";
                } else if (current_level == kernel_ending_level - 1) {
                    idx_computation = "int thread_y = threadIdx.y;\n";
                } else if (current_level == kernel_ending_level - 2) {
                    idx_computation = "int thread_z = threadIdx.z;\n";
                } else if (current_level == kernel_ending_level - 3) {
                    idx_computation = "int block_x = blockIdx.x;\n";
                } else if (current_level == kernel_ending_level - 4) {
                    idx_computation = "int block_y = blockIdx.y;\n";
                } else {
                    idx_computation = "int block_z = blockIdx.z;\n";
                }
            } else {
                assert(false && "Too many levels tagged for GPU! Must be <= 3 per thread and per block");
            }

            isl_val *inc_val = isl_ast_expr_get_val(inc);
            if (!isl_val_is_one(inc_val)) {
                tiramisu::error("The increment in one of the loops is not +1."
                                        "This is not supported", 1);
            }
            isl_val_free(inc_val);
            isl_ast_node *body = isl_ast_node_for_get_body(node);
            std::pair<std::string, std::string> bodies = tiramisu::generator::codegen_kernel_body(fct, body,
                                                                                                  current_level + 1, kernel_starting_level, kernel_ending_level);
            std::string cuda_body = bodies.first;
            cuda_body = idx_computation + cuda_body;
            return std::pair<std::string, std::string>(cuda_body, bodies.second + cuda_dim);//bodies.second);
        }
    } else if (isl_ast_node_get_type(node) == isl_ast_node_user) {
        // TODO check for let statements!
        isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
        isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
        isl_id *id = isl_ast_expr_get_id(arg);
        isl_ast_expr_free(expr);
        isl_ast_expr_free(arg);
        std::string computation_name(isl_id_get_name(id));
        isl_id *comp_id = isl_ast_node_get_annotation(node);
        tiramisu::computation *comp = (tiramisu::computation *)isl_id_get_user(comp_id);
        isl_id_free(comp_id);
        std::string code = comp->create_kernel_assignment();
        return std::pair<std::string, std::string>(code, "");
    } else {
        std::cerr << "I don't know wtf this isl type is" << std::endl;
        exit(29);
    }
}

// put all the CUDA code for this kernel together
std::pair<std::vector<std::string>, std::vector<std::string>> generate_kernel_file(std::string kernel_name, std::string kernel_fn,
                                              std::string kernel_wrapper_fn,
                                              std::string kernel_body, std::string kernel_wrapper_body,
                                              std::string fatbin_fn) {
    std::ofstream kernel;
    kernel.open(kernel_fn);
    std::ofstream kernel_wrapper;
    kernel_wrapper.open(kernel_wrapper_fn);
    // headers
    kernel << cuda_headers() << std::endl;
    kernel_wrapper << cuda_headers() << std::endl;
    // kernel
    std::string kernel_signature = "void DEVICE_" + kernel_name + "(";
    std::string kernel_wrapper_signature = "extern \"C\" {\nvoid " + kernel_name + "(";
    std::string kernel_params = "  void *kernel_args[] = {";
    std::vector<std::string> buffer_names;
    std::vector<std::string> other_params;
    int idx = 0;
    for (std::string b : closure_buffers) {
        if (idx == 0) {
            kernel_signature += b;
        } else {
            kernel_signature += ", " + b;
        }
        idx++;
    }
    idx = 0;
    for (std::string b : closure_buffers_no_type) {
        buffer_names.push_back(b);
        if (idx == 0) {
            kernel_wrapper_signature += "halide_buffer_t *" + b;
            kernel_params += "&(" + b + "->device)";
        } else {
            kernel_wrapper_signature += ", halide_buffer_t *" + b;
            kernel_params += ", &(" + b + "->device)";
        }
        idx++;
    }
    kernel_params += "};\n";
    int idx2 = idx;
    for (std::string v : closure_vars) {
        if (idx == 0) {
            kernel_wrapper_signature += v;
        } else {
            kernel_wrapper_signature += ", " + v;
        }
        idx++;
    }
    idx = idx2;
    for (std::string v : closure_vars_no_type) {
        other_params.push_back(v);
        idx++;
    }
    kernel_signature += ")";
    kernel_wrapper_signature += ", halide_buffer_t kernel_stream, void *_kernel_event_stream)";
    std::string kernel_code = "extern \"C\" {\n__global__\n";
    kernel_code +=  kernel_signature + " {\n";
    kernel_code += /*"  " + kernel_body +*/ "\n}\n}/*extern \"C\"*/\n";
    kernel << kernel_code << std::endl;
    // wrapper function that the host calls
    std::string kernel_launch = "  fprintf(stderr, \"Launching kernel\\n\");\n  assert(cuLaunchKernel(kernel, grid_width, grid_height, grid_depth, block_width, block_height, block_depth, 0 /*No shmem for now*/, ((CUstream*)(kernel_stream.host))[0], kernel_args, 0) == 0);\n";
    std::string module_mgmt = "  CUmodule mod; CUfunction kernel;\n";
    module_mgmt += "  assert(cuModuleLoad(&mod, \"" + fatbin_fn + "\") == 0);\n";
    module_mgmt += "  CUresult func_err = cuModuleGetFunction(&kernel, mod, \"DEVICE_" + kernel_name + "\");\n";
    module_mgmt += "  if (func_err != CUDA_SUCCESS) { const char *cuda_err; cuGetErrorName(func_err, &cuda_err); fprintf(stderr, \"CUDA error for cuModuleGetFunction: %s\\n\", cuda_err); assert(false); }\n";
    std::string event_check = "  //if (event != NULL) { cuStreamWaitEvent(((CUstream*)kernel_stream.host))[0], event, 0); }\n";
    std::string event_record = "  if(_kernel_event_stream) {\n    fprintf(stderr, \"creating event and recording\\n\");\n    CUevent *kernel_event_stream = (CUevent*)_kernel_event_stream;\n";//"  //if (other_event != NULL) { cuEventRecord(other_event, kernel_stream); }\n";
    event_record += "    CUevent event;\n    assert(cuEventCreate(&event, 0) == 0);\n    assert(cuEventRecord(event, ((CUstream*)(kernel_stream.host))[0]) == 0);\n    kernel_event_stream[0] = event;\n  }\n";
    kernel_wrapper << kernel_wrapper_signature << " {\nfprintf(stderr, \"In kernel wrapper: " << kernel_name << "\\n\");\n" << module_mgmt << kernel_params << kernel_wrapper_body
                   << event_check << kernel_launch << event_record << "fprintf(stderr, \"Leaving kernel wrapper: " << kernel_name << "\\n\");\n" << "}\n}/*extern \"C\"*/\n";
    kernel.close();
    kernel_wrapper.close();
    return std::pair<std::vector<std::string>, std::vector<std::string>>(buffer_names, other_params);
}

// compile the kernel file to an object file that can later be linked in
void compile_kernel_to_obj(std::string kernel_fn, std::string kernel_wrapper_fn) {
    // Generate a fat binary with a cubin file from the kernel
    std::string cmd = "nvcc -I/Users/JRay/ClionProjects/tiramisu/include -I/data/hltemp/jray/tiramisu/include -I/data/hltemp/jray/tiramisu/Halide/include -I/Users/JRay/ClionProjects/tiramisu/Halide/include -ccbin $NVCC_CLANG --compile -g -O3 --std=c++11 " + kernel_fn + " --fatbin -odir /tmp/";
    std::cerr << "cmd: " << cmd << std::endl;
    int ret = system(cmd.c_str());
    assert(ret == 0 && "Non-zero exit code for nvcc invocation");
    // Compile the wrapper into an object file
    cmd = "nvcc -I/Users/JRay/ClionProjects/tiramisu/include -I/Users/JRay/ClionProjects/tiramisu/Halide/include -I/data/hltemp/jray/tiramisu/include -I/data/hltemp/jray/tiramisu/Halide/include -ccbin $NVCC_CLANG --compile -g -O3 --std=c++11 " + kernel_wrapper_fn + " -odir /tmp/";
    std::cerr << "cmd: " << cmd << std::endl;
    ret = system(cmd.c_str());
    assert(ret == 0 && "Non-zero exit code for nvcc invocation");
}

std::tuple<std::string, std::vector<std::string>, std::vector<std::string>> tiramisu::generator::cuda_kernel_from_isl_node(function &fct, isl_ast_node *node,
                                                           int level, std::vector<std::string> &tagged_stmts, std::string kernel_name, int start_kernel_level, int end_kernel_level) {
    std::pair<std::string, std::string> bodies = generator::codegen_kernel_body(fct, node, level, start_kernel_level, end_kernel_level);
    std::string kernel_body = bodies.first;
    std::string wrapper_body = bodies.second;
    std::string kernel_fn = "/tmp/" + kernel_name + ".cu";
    std::string kernel_fatbin_fn = "/tmp/" + kernel_name + ".fatbin";
    std::string kernel_wrapper_fn = "/tmp/" + kernel_name + "_wrapper.cu";
    std::pair<std::vector<std::string>, std::vector<std::string>> params =
            generate_kernel_file(kernel_name, kernel_fn, kernel_wrapper_fn, kernel_body, wrapper_body, kernel_fatbin_fn);
    compile_kernel_to_obj(kernel_fn, kernel_wrapper_fn);
    std::tuple<std::string, std::vector<std::string>, std::vector<std::string>> ret(kernel_fn, params.first, params.second);
    return ret;
}

std::string c_type_from_tiramisu_type(tiramisu::primitive_t type) {
    switch (type) {
        case tiramisu::p_uint8:
            return "unsigned char";
        case tiramisu::p_int8:
            return "char";
        case tiramisu::p_uint16:
            return "unsigned short";
        case tiramisu::p_int16:
            return "short";
        case tiramisu::p_uint32:
            return "unsigned int";
        case tiramisu::p_int32:
            return "int";
        case tiramisu::p_uint64:
            return "unsigned long";
        case tiramisu::p_int64:
            return "long";
        case tiramisu::p_float32:
            return "float";
        case tiramisu::p_float64:
            return "double";
        case tiramisu::p_boolean:
            return "bool";
        default:
            tiramisu::error("Tiramisu type not supported.", true);
            return "";
    }
}

std::string cuda_expr_from_isl_ast_expr(isl_ast_expr *isl_expr, int kernel_starting_level, int kernel_ending_level,
                                        bool convert_to_loop_type, bool map_iterator, bool capture_vars) {
    std::string result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int) {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        if (!convert_to_loop_type) {
            result = "(int)" + std::to_string(isl_val_get_num_si(init_val));
        } else {
            result = "(" + c_type_from_tiramisu_type(global::get_loop_iterator_data_type()) + ")" +
                     std::to_string(isl_val_get_num_si(init_val));
        }
        isl_val_free(init_val);
    } else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id) {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        isl_id_free(identifier);

        if (map_iterator) {
            std::string tmp_name = name_str.substr(1);
            int current_level = std::stoi(tmp_name);
            current_level /= 2;
            if (kernel_ending_level - kernel_starting_level == 1) { // 1 dimension
                if (current_level == kernel_ending_level) {
                    name_str = "thread_x";
                } else {
                    name_str = "block_x";
                }
            } else if (kernel_ending_level - kernel_starting_level == 3) { // 2 dimension
                assert(false);
                if (current_level == kernel_ending_level) {
                    name_str = "thread_y";
                } else if (current_level == kernel_ending_level - 1) {
                    name_str = "thread_x";
                } else if (current_level == kernel_ending_level - 2) {
                    name_str = "block_y";
                } else {
                    name_str = "block_x";
                }
            } else if (kernel_ending_level - kernel_starting_level == 5) { // 3 dimension
                assert(false);
                if (current_level == kernel_ending_level) {
                    name_str = "thread_z";
                } else if (current_level == kernel_ending_level - 1) {
                    name_str = "thread_y";
                } else if (current_level == kernel_ending_level - 2) {
                    name_str = "thread_x";
                } else if (current_level == kernel_ending_level - 3) {
                    name_str = "block_z";
                } else if (current_level == kernel_ending_level - 4) {
                    name_str = "block_y";
                } else {
                    name_str = "block_x";
                }
            }
        }

        if (!convert_to_loop_type) {
            result = "(int)" + name_str;
            if (capture_vars) {
                if (std::find(closure_vars.begin(), closure_vars.end(), "int " + name_str) == closure_vars.end()) {
                    closure_vars.push_back("int " + name_str);
                    closure_vars_no_type.push_back(name_str);
                }
            }
        } else {
            result = "(" + c_type_from_tiramisu_type(global::get_loop_iterator_data_type()) + ")" + name_str;
            if (capture_vars) {
                if (std::find(closure_vars.begin(), closure_vars.end(),
                              c_type_from_tiramisu_type(global::get_loop_iterator_data_type()) + " " + name_str) ==
                    closure_vars.end()) {
                    closure_vars.push_back(
                            c_type_from_tiramisu_type(global::get_loop_iterator_data_type()) + " " + name_str);
                    closure_vars_no_type.push_back(name_str);
                }
            }
        }

    } else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op) {
        std::string op0, op1, op2;

        isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(isl_expr, 0);
        op0 = cuda_expr_from_isl_ast_expr(expr0, kernel_starting_level, kernel_ending_level, convert_to_loop_type, map_iterator, capture_vars);
        isl_ast_expr_free(expr0);

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
        {
            isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(isl_expr, 1);
            op1 = cuda_expr_from_isl_ast_expr(expr1, kernel_starting_level, kernel_ending_level, convert_to_loop_type, map_iterator, capture_vars);
            isl_ast_expr_free(expr1);
        }

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
        {
            isl_ast_expr *expr2 = isl_ast_expr_get_op_arg(isl_expr, 2);
            op2 = cuda_expr_from_isl_ast_expr(expr2, kernel_starting_level, kernel_ending_level, convert_to_loop_type, map_iterator, capture_vars);
            isl_ast_expr_free(expr2);
        }
        switch (isl_ast_expr_get_op_type(isl_expr))
        {
            case isl_ast_op_and:
                result = "(" + op0 + " && " + op1 + ")";
                break;
            case isl_ast_op_and_then:
                assert(false);
                tiramisu::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_or:
                result = "(" + op0 + " || " + op1 + ")";
                break;
            case isl_ast_op_or_else:
                assert(false);
                tiramisu::error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_max:
                assert(false);
                break;
            case isl_ast_op_min:
//                result = "(std::min(" + op0 + ", " + op1 + "))";
                result = "(" + op0 + " < " + op1 + " ? " + op0 + " : " + op1 + ")";
                break;
            case isl_ast_op_minus:
                result = "(-" + op0 + ")";
                break;
            case isl_ast_op_add:
                result = "(" + op0 + " + " + op1 + ")";
                break;
            case isl_ast_op_sub:
                result = "(" + op0 + " - " + op1 + ")";
                break;
            case isl_ast_op_mul:
                result = "(" + op0 + " * " + op1 + ")";
                break;
            case isl_ast_op_div:
                result = "(" + op0 + " / " + op1 + ")";
                break;
            case isl_ast_op_fdiv_q:
            case isl_ast_op_pdiv_q:
                assert(false);
                break;
            case isl_ast_op_zdiv_r:
            case isl_ast_op_pdiv_r:
                assert(false);
                break;
            case isl_ast_op_select:
            case isl_ast_op_cond:
                result = "(" + op0 + " ? " + op1 + " : " + op2 + ")";
                break;
            case isl_ast_op_le:
                result = "(" + op0 + " <= " + op1 + ")";
                break;
            case isl_ast_op_lt:
                result = "(" + op0 + " < " + op1 + ")";
                break;
            case isl_ast_op_ge:
                result = "(" + op0 + " >= " + op1 + ")";
                break;
            case isl_ast_op_gt:
                result = "(" + op0 + " > " + op1 + ")";
                break;
            case isl_ast_op_eq:
                result = "(" + op0 + " == " + op1 + ")";
                break;
            default:
                tiramisu::str_dump("Transforming the following expression",
                                   (const char *)isl_ast_expr_to_C_str(isl_expr));
                tiramisu::str_dump("\n");
                tiramisu::error("Translating an unsupported ISL expression in a Halide expression.", 1);
        }
    }
    else
    {
        tiramisu::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(isl_expr));
        tiramisu::str_dump("\n");
        tiramisu::error("Translating an unsupported ISL expression in a Halide expression.", 1);
    }

    return result;
}

// my bounds on this loop are messed up
std::string generator::linearize_access_cuda(int dims, const shape_t *shape, isl_ast_expr *index_expr, int kernel_starting_level, int kernel_ending_level) {
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);
    std::string index = "";
    for (int i = dims; i >= 1; i--) {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        std::string operand_h = cuda_expr_from_isl_ast_expr(operand, kernel_starting_level, kernel_ending_level, true, true);
        if (i == dims) {
            index = "((" + operand_h + ") * (" + std::to_string(shape[dims-i].stride) + "))";
        } else {
            index = "(" + index + " + " + "((" + operand_h + ") * (" + std::to_string(shape[dims-i].stride) + ")))";
        }
        isl_ast_expr_free(operand);
    }

    return index;
}

std::string generator::linearize_access_cuda(int dims, const shape_t *shape, std::vector<tiramisu::expr> index_expr, int, int) {
    assert(false && "Fix linearize");
    assert(index_expr.size() > 0);
    std::string index = "";
    for (int i = 1; i < dims; i++)
    {
        std::vector<isl_ast_expr *> ie = {};
        std::string operand_h = generator::cuda_expr_from_tiramisu_expr(NULL, ie, index_expr[i-1], nullptr);
        index += operand_h + " * " + std::to_string(shape[i-1].stride);
    }

    DEBUG_INDENT(-4);

    return index;
}

std::string generator::linearize_access_cuda(int dims, std::vector<std::string> &strides, std::vector<tiramisu::expr> index_expr, int, int) {
    assert(false && "Fix linearize");
    assert(index_expr.size() > 0);
    std::string index = "";
    for (int i = 1; i < dims; i++)
    {
        std::vector<isl_ast_expr *> ie = {};
        std::string operand_h = generator::cuda_expr_from_tiramisu_expr(NULL, ie, index_expr[i-1], nullptr);
        index += operand_h + " * " + strides[i-1];
    }
    return index;
}

std::string generator::linearize_access_cuda(int dims, std::vector<std::string> &strides, isl_ast_expr *index_expr, int kernel_starting_level, int kernel_ending_level) {
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);
    std::string index = "";
    for (int i = dims; i >= 1; i--)
    {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i); // skip the first op arg, which is just the name of the buffer
        std::string operand_h = cuda_expr_from_isl_ast_expr(operand, kernel_starting_level, kernel_ending_level, true, true);
        if (i == dims) {
            index = "((" + operand_h + ") * (" + strides[dims-i] + "))";
        } else {
            index = "(" + index + " + ((" + operand_h + ") * (" + strides[dims-i] + ")))";
        }
        isl_ast_expr_free(operand);
    }

    DEBUG_INDENT(-4);

    return index;
}


std::string tiramisu::computation::create_kernel_assignment() {
    assert(!this->is_let_stmt() && "Figure out how to handle a let stmt in the cuda kernel");
    assert(!this->is_library_call() && "Shouldn't have a library call in the kernel");

    std::string c_type = c_type_from_tiramisu_type(this->get_data_type());
    std::string lhs_buffer_name = isl_space_get_tuple_name(
            isl_map_get_space(this->get_access_relation_adapted_to_time_processor_domain()),
            isl_dim_out);

    isl_map *lhs_access = this->get_access_relation_adapted_to_time_processor_domain();
    isl_space *space = isl_map_get_space(lhs_access);
    // Get the number of dimensions of the ISL map representing
    // the lhs_access.
    int num_lhs_access_dims = isl_space_dim(space, isl_dim_out);

    // Fetch the actual buffer.
    const auto &buffer_entry = this->fct->get_buffers().find(lhs_buffer_name);
    assert(buffer_entry != this->fct->get_buffers().end());
    const auto &lhs_tiramisu_buffer = buffer_entry->second;
    int lhs_buf_dims = lhs_tiramisu_buffer->get_dim_sizes().size();

    std::string buffer_sig = c_type_from_tiramisu_type(lhs_tiramisu_buffer->get_elements_type()) + " *" + lhs_buffer_name;
    if (std::find(closure_buffers.begin(), closure_buffers.end(), buffer_sig) == closure_buffers.end()) {
        closure_buffers.push_back(buffer_sig);
        closure_buffers_no_type.push_back(lhs_buffer_name);
    }

    // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is
    // from innermost to outermost; thus, we need to reverse the order
    shape_t *lhs_shape = new shape_t[lhs_tiramisu_buffer->get_dim_sizes().size()];
    int stride = 1;
    std::vector<std::string> strides_vector;

    if (lhs_tiramisu_buffer->has_constant_extents()) {
        for (int i = 0; i < lhs_buf_dims; i++) {
            lhs_shape[i].min = 0;
            int dim_idx = lhs_tiramisu_buffer->get_dim_sizes().size() - i - 1;;
            lhs_shape[i].extent = (int) lhs_tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
            lhs_shape[i].stride = stride;
            stride *= (int) lhs_tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
        }
    } else {
        std::vector<isl_ast_expr *> empty_index_expr;
        std::string stride_expr = "1";
        for (int i = 0; i < lhs_tiramisu_buffer->get_dim_sizes().size(); i++) {
            int dim_idx = lhs_tiramisu_buffer->get_dim_sizes().size() - i - 1;
            strides_vector.push_back(stride_expr);
            if (i == 0) {
                stride_expr = "(" + generator::cuda_expr_from_tiramisu_expr(this->get_function(), empty_index_expr,
                                                                            generator::replace_original_indices_with_gpu_indices(this,
                                                                                                                                 lhs_tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                                                                 this->get_iterators_map()),
                                                                            this) + ")";
            } else {
                stride_expr = "((" + stride_expr + ") * (" +
                              generator::cuda_expr_from_tiramisu_expr(this->get_function(), empty_index_expr,
                                                                      generator::replace_original_indices_with_gpu_indices(this,
                                                                                                                           lhs_tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                                                           this->get_iterators_map()),
                                                                      this) + "))";
            }
        }
    }

    // The number of dimensions in the Halide buffer should be equal to
    // the number of dimensions of the lhs_access function.
    assert(lhs_buf_dims == num_lhs_access_dims);
    assert(this->index_expr[0] != NULL);
    std::string lhs_index = "";
    std::pair<int, int> range = this->fct->gpu_ranges[this->get_name()];

    if (lhs_tiramisu_buffer->has_constant_extents()) {
        lhs_index = generator::linearize_access_cuda(lhs_buf_dims, lhs_shape, this->index_expr[0], range.first, range.second);
    } else {
        lhs_index = tiramisu::generator::linearize_access_cuda(lhs_buf_dims, strides_vector, this->index_expr[0], range.first, range.second);
    }

    this->index_expr.erase(this->index_expr.begin());
    assert(this->lhs_access_type == tiramisu::o_access && "Only o_access allowed for LHS type in gpu kernel generation");

    // Replace the RHS expression to the transformed expressions.
    // We do not need to transform the indices of expression (this->index_expr), because in Tiramisu we assume
    // that an access can only appear when accessing a computation. And that case should be handled in the following transformation
    // so no need to transform this->index_expr separately.
    tiramisu::expr tiramisu_rhs = generator::replace_original_indices_with_gpu_indices(this, this->expression, this->get_iterators_map());
    std::string rhs_expr = generator::cuda_expr_from_tiramisu_expr(this->get_function(), this->index_expr, tiramisu_rhs, this);
    std::string store_expr = lhs_buffer_name + "[" + lhs_index + "] = " + rhs_expr + ";";
    return store_expr;
}

std::string tiramisu::generator::cuda_expr_from_tiramisu_expr(const tiramisu::function *fct, std::vector<isl_ast_expr *> &index_expr,
                                                              const tiramisu::expr &tiramisu_expr, tiramisu::computation *comp, bool map_iterators) {
    std::string result = "";
    if (tiramisu_expr.get_expr_type() == tiramisu::e_val)
    {
        if (tiramisu_expr.get_data_type() == tiramisu::p_uint8)
        {
            result = std::to_string(tiramisu_expr.get_uint8_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int8)
        {
            result = std::to_string(tiramisu_expr.get_int8_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint16)
        {
            result = std::to_string(tiramisu_expr.get_uint16_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int16)
        {
            result = std::to_string(tiramisu_expr.get_int16_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint32)
        {
            result = std::to_string(tiramisu_expr.get_uint32_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int32)
        {
            result = std::to_string(tiramisu_expr.get_int32_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint64)
        {
            result = std::to_string(tiramisu_expr.get_uint64_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int64)
        {
            result = std::to_string(tiramisu_expr.get_int64_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_float32)
        {
            result = std::to_string(tiramisu_expr.get_float32_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_float64)
        {
            result = std::to_string(tiramisu_expr.get_float64_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_string)
        {
            result = tiramisu_expr.get_string_value();
        }
    }
    else if (tiramisu_expr.get_expr_type() == tiramisu::e_op)
    {
        std::string op0, op1, op2;

        if (tiramisu_expr.get_n_arg() > 0)
        {
            tiramisu::expr expr0 = tiramisu_expr.get_operand(0);
            op0 = generator::cuda_expr_from_tiramisu_expr(fct, index_expr, expr0, comp);
        }

        if (tiramisu_expr.get_n_arg() > 1)
        {
            tiramisu::expr expr1 = tiramisu_expr.get_operand(1);
            op1 = generator::cuda_expr_from_tiramisu_expr(fct, index_expr, expr1, comp);
        }

        if (tiramisu_expr.get_n_arg() > 2)
        {
            tiramisu::expr expr2 = tiramisu_expr.get_operand(2);
            op2 = generator::cuda_expr_from_tiramisu_expr(fct, index_expr, expr2, comp);
        }
        assert(tiramisu_expr.get_op_type() != tiramisu::o_buffer && "o_buffer shouldn't survive til here!");
        switch (tiramisu_expr.get_op_type())
        {
            case tiramisu::o_logical_and:
                result = "(" + op0 + " && " + op1 + ")";
                break;
            case tiramisu::o_logical_or:
                result = "(" + op0 + " || " + op1 + ")";
                break;
            case tiramisu::o_max:
                assert(false && "max not supported in kernel");
                break;
            case tiramisu::o_min:
                assert(false && "min not supported in kernel");
                break;
            case tiramisu::o_minus:
                result = "(-" + op0 + ")";
                break;
            case tiramisu::o_add:
                result = "(" + op0 + " + " + op1 + ")";
                break;
            case tiramisu::o_sub:
                result = "(" + op0 + " - " + op1 + ")";
                break;
            case tiramisu::o_mul:
                result = "(" + op0 + " * " + op1 + ")";
                break;
            case tiramisu::o_div:
                result = "(" + op0 + " / " + op1 + ")";
                break;
            case tiramisu::o_mod:
                assert(false && "modulo not supported in kernel");
                break;
            case tiramisu::o_select:
                result = "(" + op0 + " ? " + op1 + " : " + op2 + ")";
                break;
            case tiramisu::o_lerp:
                assert(false && "lerp not supported in kernel");
                break;
            case tiramisu::o_cond:
                tiramisu::error("Code generation for o_cond is not supported yet.", true);
                break;
            case tiramisu::o_le:
                result = "(" + op0 + " <= " + op1 + ")";
                break;
            case tiramisu::o_lt:
                result = "(" + op0 + " < " + op1 + ")";
                break;
            case tiramisu::o_ge:
                result = "(" + op0 + " >= " + op1 + ")";
                break;
            case tiramisu::o_gt:
                result = "(" + op0 + " > " + op1 + ")";
                break;
            case tiramisu::o_logical_not:
                result = "(!" + op0 + ")";
                break;
            case tiramisu::o_eq:
                result = "(" + op0 + " == " + op1 + ")";
                break;
            case tiramisu::o_ne:
                result = "(" + op0 + " != " + op1 + ")";
                break;
            case tiramisu::o_bitwise_and:
                result = "(" + op0 + " & " + op1 + ")";
                break;
            case tiramisu::o_bitwise_or:
                result = "(" + op0 + " | " + op1 + ")";
                break;
            case tiramisu::o_bitwise_xor:
                result = "(" + op0 + " ^ " + op1 + ")";
                break;
            case tiramisu::o_bitwise_not:
                result = "(~" + op0 + ")";
                break;
            case tiramisu::o_is_nan:
                tiramisu::error("Code generation for o_is_nan is not supported yet in CUDA.", true);
                break;
            case tiramisu::o_pow:
                if (tiramisu_expr.get_data_type() == tiramisu::p_float32) {
                    result = "powf(" + op0 + ", " + op1 + ")";
                } else {
                    result = "pow(" + op0 + ", " + op1 + ")";
                }
                break;
            case tiramisu::o_type:
            {
                tiramisu::error("Code generation for o_type is not supported yet in CUDA.", true);
            }
            case tiramisu::o_right_shift:
                result = "(" + op0 + ">>" + op1 + ")";
                break;
            case tiramisu::o_left_shift:
                result = "(" + op0 + " << " + op1 + ")";
                break;
            case tiramisu::o_floor:
                if (tiramisu_expr.get_data_type() == tiramisu::p_float32) {
                    result = "floorf(" + op0 + ")";
                } else {
                    result = "floor(" + op0 + ")";
                }
                break;
            case tiramisu::o_cast:
                result = "((" + c_type_from_tiramisu_type(tiramisu_expr.get_data_type()) + ")" + op0 + ")";
                break;
            case tiramisu::o_sin:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_cos:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_tan:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_asin:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_acos:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_atan:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_abs:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_sqrt:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_expo:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_log:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_ceil:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_round:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_trunc:
                assert(false && "do this cuda func later");
                break;
            case tiramisu::o_access:
            case tiramisu::o_lin_index:
            case tiramisu::o_address:
            case tiramisu::o_address_of:
            {

                const char *access_comp_name = NULL;

                if (tiramisu_expr.get_op_type() == tiramisu::o_access ||
                    tiramisu_expr.get_op_type() == tiramisu::o_lin_index ||
                    tiramisu_expr.get_op_type() == tiramisu::o_address_of)
                {
                    access_comp_name = tiramisu_expr.get_name().c_str();
                }
                else if (tiramisu_expr.get_op_type() == tiramisu::o_address)
                {
                    access_comp_name = tiramisu_expr.get_name().c_str();
                }
                else
                {
                    tiramisu::error("Unsupported operation.", true);
                }

                assert(access_comp_name != NULL);


                // Since we modify the names of update computations but do not modify the
                // expressions.  When accessing the expressions we find the old names, so
                // we need to look for the new names instead of the old names.
                // We do this instead of actually changing the expressions, because changing
                // the expressions will make the semantics of the printed program ambiguous,
                // since we do not have any way to distinguish between which update is the
                // consumer is consuming exactly.
                std::vector<tiramisu::computation *> computations_vector
                        = fct->get_computation_by_name(access_comp_name);
                if (computations_vector.size() == 0 && std::strcmp(access_comp_name, "dummy") != 0)
                {
                    // Search for update computations.
                    computations_vector
                            = fct->get_computation_by_name("_" + std::string(access_comp_name) + "_update_0");
                    assert((computations_vector.size() > 0) && "Computation not found.");
                }

                // We assume that computations that have the same name write all to the same buffer
                // but may have different access relations.
                tiramisu::computation *access_comp = computations_vector[0];

                // A parent partition isn't allowed to be accessed directly, so we need to swap it with the correct child partition
                if (access_comp && access_comp->is_a_parent_partition) {
                    // The child partition to swap with is the one with the same predicate
                    bool found_child = false;
                    int this_ops_pred = comp->get_predicate().get_int_val();
                    for (auto child : access_comp->child_partitions) {
                        if (child->get_predicate().get_int_val() == this_ops_pred) {
                            assert(!found_child && "Found more than one child with the same predicate!");
                            found_child = true;
                            access_comp = child;
                        }
                    }
                }

                assert((access_comp != NULL) && "Accessed computation is NULL.");

                if (comp && comp->is_wait()) {
                    // swap
                    // use operations_vector[0] instead of access_comp because we need it to be non-const
                    isl_map *orig = computations_vector[0]->get_access_relation();
                    computations_vector[0]->set_access(computations_vector[0]->wait_access_map);
                    computations_vector[0]->wait_access_map = orig;
                }
                isl_map *acc = access_comp->get_access_relation_adapted_to_time_processor_domain();
                if (comp && comp->is_wait()) {
                    // swap back
                    isl_map *orig = computations_vector[0]->get_access_relation();
                    computations_vector[0]->set_access(computations_vector[0]->wait_access_map);
                    computations_vector[0]->wait_access_map = orig;
                }

                const char *buffer_name = isl_space_get_tuple_name(isl_map_get_space(acc), isl_dim_out);
                assert(buffer_name != NULL);

                const auto &buffer_entry = fct->get_buffers().find(buffer_name);
                assert(buffer_entry != fct->get_buffers().end());

                const auto &tiramisu_buffer = buffer_entry->second;

                std::string buffer_sig = c_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()) + " *" + buffer_name;
                if (std::find(closure_buffers.begin(), closure_buffers.end(), buffer_sig) == closure_buffers.end()) {
                    closure_buffers.push_back(buffer_sig);
                    closure_buffers_no_type.push_back(buffer_name);
                }

                std::string type = c_type_from_tiramisu_type(tiramisu_buffer->get_elements_type());

                // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is from innermost
                // to outermost; thus, we need to reverse the order
                shape_t *shape = new shape_t[tiramisu_buffer->get_dim_sizes().size()];
                int stride = 1;
                std::vector<std::string> strides_vector;

                if (tiramisu_buffer->has_constant_extents()) {
                    for (size_t i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                        shape[i].min = 0;
                        int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        shape[i].extent = (int)tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                        shape[i].stride = stride;
                        stride *= (int)tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                    }
                } else {
                    std::vector<isl_ast_expr *> empty_index_expr;
                    std::string stride_expr = "1";
                    for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                        int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        strides_vector.push_back(stride_expr);
                        if (i == 0) {
                            stride_expr = "(" + generator::cuda_expr_from_tiramisu_expr(fct, empty_index_expr,
                                                                                        tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                        comp) + ")";
                        } else {
                            stride_expr = "((" + stride_expr + ") * (" + generator::cuda_expr_from_tiramisu_expr(fct, empty_index_expr,
                                                                                                                 tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                                                 comp) + "))";
                        }
                    }
                }
                if (tiramisu_expr.get_op_type() == tiramisu::o_access ||
                    tiramisu_expr.get_op_type() == tiramisu::o_address_of ||
                    tiramisu_expr.get_op_type() == tiramisu::o_lin_index) {
                    std::string index;
                    std::pair<int, int> range = comp->fct->gpu_ranges[comp->get_name()];
                    if (index_expr.size() == 0) {
                        for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                            assert(tiramisu_expr.get_access()[i].is_constant() && "Only constant accesses are supported.");
                        }
                        if (tiramisu_buffer->has_constant_extents()) {
                            index = generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                                                     shape, tiramisu_expr.get_access(), range.first, range.second);
                        } else {
                            index = tiramisu::generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                                                               strides_vector, tiramisu_expr.get_access(), range.first, range.second);
                        }
                    } else {
                        if (tiramisu_buffer->has_constant_extents()) {
                            index = generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                                                     shape, index_expr[0], range.first, range.second);
                        } else {
                            index = tiramisu::generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                                                               strides_vector, index_expr[0], range.first, range.second);
                        }
                        index_expr.erase(index_expr.begin());
                    }
                    if (tiramisu_expr.get_op_type() == tiramisu::o_lin_index) {
                        result = index;
                    } else if (tiramisu_expr.get_op_type() == tiramisu::o_address) {
                        result = tiramisu_buffer->get_name();
                    } else {
                        result = tiramisu_buffer->get_name() + "[" + index + "]";
                        if (tiramisu_expr.get_op_type() == tiramisu::o_address_of) {
                            result = "&(" + result + ")";
                        }
                    }
                }
                delete[] shape;
            }
                break;
            case tiramisu::o_call:
            case tiramisu::o_allocate:
            case tiramisu::o_free:
                tiramisu::error("An expression of type o_allocate or o_free "
                                        "should not be passed to this function", true);
                break;
            default:
                tiramisu::error("Translating an unsupported ISL expression into a Halide expression.", 1);
        }
    } else if (tiramisu_expr.get_expr_type() == tiramisu::e_var) {
        result = tiramisu_expr.get_name();
    } else {
        tiramisu::str_dump("tiramisu type of expr: ",
                           str_from_tiramisu_type_expr(tiramisu_expr.get_expr_type()).c_str());
        tiramisu::error("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
    }
    return result;
}

}
