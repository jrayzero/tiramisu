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
// 0b. Get rid of the GPU tagged loops (it is processing those still)
// 0c. Make sure to convert the outer loop to a conditional!
// 1. Initialize context and all of that (that would go in the wrapper function that is called from the Halide IR)
// 2. Launch kernel
// make sure having multiple computations in the same loop nest work. Both will need to be converted into kernels
namespace tiramisu {

  std::string cuda_headers() {
    std::string includes = "#include <cuda.h>\n";
    includes += "#include <iostream>\n";
    return includes;
  }

  // generate a kernel from the original isl node
  std::string generator::codegen_kernel_body(function &fct, isl_ast_node *node, int current_level, int kernel_starting_level, int kernel_ending_level) {
    if (isl_ast_node_get_type(node) == isl_ast_node_block) {
      std::cerr << "WTF is an isl block" << std::endl;
      exit(29);
      return "";
    } else if (isl_ast_node_get_type(node) == isl_ast_node_for) {
      std::string code = "";
      if (current_level > kernel_ending_level) {
        std::cerr << "Generating for loop in kernel" << std::endl;



      } else { // this is a GPU kernel loop, i.e. a thread iterator (or block iterator, w/e)
        std::cerr << "Generating a thread iterator for the kernel" << std::endl;
        


      }
      isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
      char *cstr = isl_ast_expr_to_C_str(iter);
      std::string iterator_str = std::string(cstr);
      isl_ast_expr *init = isl_ast_node_for_get_init(node);
      isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
      isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);
      
      isl_val *inc_val = isl_ast_expr_get_val(inc);
      if (!isl_val_is_one(inc_val))
        {
          tiramisu::error("The increment in one of the loops is not +1."
                          "This is not supporte", 1);
        }
        isl_val_free(inc_val);
        
        isl_ast_node *body = isl_ast_node_for_get_body(node);
        isl_ast_expr *cond_upper_bound_isl_format = NULL;
        
        std::string cuda_body = tiramisu::generator::codegen_kernel_body(fct, body, current_level + 1, kernel_starting_level, kernel_ending_level);
        return cuda_body;
    } else if (isl_ast_node_get_type(node) == isl_ast_node_user) {
      // TODO check for let statements!
      std::cerr << "User node" << std::endl;
      isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
      isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
      isl_id *id = isl_ast_expr_get_id(arg);
      isl_ast_expr_free(expr);
      isl_ast_expr_free(arg);
      std::string computation_name(isl_id_get_name(id));
      std::cerr << "The user node name is " << computation_name << std::endl;
      isl_id *comp_id = isl_ast_node_get_annotation(node);
      tiramisu::computation *comp = (tiramisu::computation *)isl_id_get_user(comp_id);
      isl_id_free(comp_id);
      std::string code = comp->create_kernel_assignment();
      return code;
    } else {
      std::cerr << "I don't know wtf this isl type is" << std::endl;
      exit(29);
      return "";
    }
  }

  // put all the CUDA code for this kernel together
  void generate_kernel_file(std::string kernel_name, std::string kernel_fn, std::string body) {
    std::ofstream kernel;
    kernel.open(kernel_fn);
    // headers
    kernel << cuda_headers() << std::endl;
    // kernel
    std::string kernel_signature = "void DEVICE_" + kernel_name + "()";
    std::string kernel_code = "__global__\n";    
    kernel_code +=  kernel_signature + " {\n";
    kernel_code += body + "\n}\n\n";    
    // wrapper function that the host calls
    std::string kernel_wrapper_signature = "void " + kernel_name + "()";
    kernel_code += kernel_wrapper_signature + " { }\n\n";
    kernel << kernel_code << std::endl;
    kernel.close();
  }

  // compile the kernel file to an object file that can later be linked in
  void compile_kernel_to_obj(std::string kernel_fn) {
    std::string compiler = "nvcc -IHalide/include -Iinclude/ -ccbin $NVCC_CLANG --compile -g -O3 --std=c++11 " + kernel_fn + " -odir build";
    int ret = system(compiler.c_str());
    assert(ret == 0 && "Non-zero exit code for nvcc invocation");
    
  }

  std::string tiramisu::generator::cuda_kernel_from_isl_node(function &fct, isl_ast_node *node,
                                                             int level, std::vector<std::string> &tagged_stmts, std::string kernel_name, int start_kernel_level, int end_kernel_level) {
    std::string body = generator::codegen_kernel_body(fct, node, level, start_kernel_level, end_kernel_level);
    std::string kernel_fn = "build/" + kernel_name + ".cu";
    generate_kernel_file(kernel_name, kernel_fn, body);
    compile_kernel_to_obj(kernel_fn);
    return kernel_fn;
    
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
  
  std::string cuda_expr_from_isl_ast_expr(isl_ast_expr *isl_expr, bool convert_to_loop_type = false) {
    std::string result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int) {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        if (!convert_to_loop_type) {
          result = "(int)" + std::to_string(isl_val_get_num_si(init_val));//Halide::Expr((int32_t) isl_val_get_num_si(init_val));
        } else {
          result = "(" + c_type_from_tiramisu_type(global::get_loop_iterator_data_type()) + ")" + std::to_string(isl_val_get_num_si(init_val));
        }
        isl_val_free(init_val);
    } else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id) {
      isl_id *identifier = isl_ast_expr_get_id(isl_expr);
      std::string name_str(isl_id_get_name(identifier));
      isl_id_free(identifier);
      if (!convert_to_loop_type) {
        result = "int " + name_str;
      } else {
        result = c_type_from_tiramisu_type(global::get_loop_iterator_data_type()) + " " + name_str;
      }
    } else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op) {
      std::string op0, op1, op2;
      
      isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(isl_expr, 0);
      op0 = cuda_expr_from_isl_ast_expr(expr0, convert_to_loop_type);
      isl_ast_expr_free(expr0);
      
      if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
        {
          isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(isl_expr, 1);
          op1 = cuda_expr_from_isl_ast_expr(expr1, convert_to_loop_type);
          isl_ast_expr_free(expr1);
        }
      
      if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
        {
          isl_ast_expr *expr2 = isl_ast_expr_get_op_arg(isl_expr, 2);
          op2 = cuda_expr_from_isl_ast_expr(expr2, convert_to_loop_type);
          isl_ast_expr_free(expr2);
        }
      
      switch (isl_ast_expr_get_op_type(isl_expr))
        {
        case isl_ast_op_and:
          result = op0 + " && " + op1;
          break;
        case isl_ast_op_and_then:
          assert(false);
          tiramisu::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.",
                          0);
          break;
        case isl_ast_op_or:
          result = op0 + " || " + op1;
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
          assert(false);
          break;
        case isl_ast_op_minus:          
          result = "-" + op0;
          break;
        case isl_ast_op_add:
          result = op0 + " + " + op1;
          break;
        case isl_ast_op_sub:
          result = op0 + " - " + op1;
          break;
        case isl_ast_op_mul:
          result = op0 + " * " + op1;
          break;
        case isl_ast_op_div:
          result = op0 + " / " + op1;
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
          result = op0 + " ? " + op1 + " : " + op2;
          break;
        case isl_ast_op_le:
          result = op0 + " <= " + op1;
          break;
        case isl_ast_op_lt:
          result = op0 + " < " + op1;
          break;
        case isl_ast_op_ge:
          result = op0 + " >= " + op1;
          break;
        case isl_ast_op_gt:
          result = op0 + " > " + op1;
          break;
        case isl_ast_op_eq:
          result = op0 + " == " + op1;
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


  std::string generator::linearize_access_cuda(int dims, const shape_t *shape, isl_ast_expr *index_expr) {
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);
    
    std::string index = "";
    for (int i = 0; i < dims; i++)
      {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        std::string operand_h = cuda_expr_from_isl_ast_expr(operand, true);
        index += operand_h + " * " + std::to_string(shape[i].stride);
        isl_ast_expr_free(operand);
      }
        
    return index;
  }

  std::string generator::linearize_access_cuda(int dims, const shape_t *shape, std::vector<tiramisu::expr> index_expr) {
    assert(index_expr.size() > 0);
    std::string index = "";
    for (int i = 0; i < dims; i++)
    {
        std::vector<isl_ast_expr *> ie = {};
        std::string operand_h = generator::cuda_expr_from_tiramisu_expr(NULL, ie, index_expr[i], nullptr);
        index += operand_h + " * " + std::to_string(shape[i].stride);
    }

    DEBUG_INDENT(-4);

    return index;
  }
  
  std::string generator::linearize_access_cuda(int dims, std::vector<std::string> &strides, std::vector<tiramisu::expr> index_expr) {
    assert(index_expr.size() > 0);
    std::string index = "";
    for (int i = 0; i < dims; i++)
      {
        std::vector<isl_ast_expr *> ie = {};
        std::string operand_h = generator::cuda_expr_from_tiramisu_expr(NULL, ie, index_expr[i - 1], nullptr);
        index += operand_h + " * " + strides[i];
      }
    return index;
  }
  
  std::string generator::linearize_access_cuda(int dims, std::vector<std::string> &strides, isl_ast_expr *index_expr) {
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);    
    std::string index = "";
    for (int i = 0; i < dims; i++)
      {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        std::string operand_h = cuda_expr_from_isl_ast_expr(operand, true);
        index += operand_h + " * " + strides[i];
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
    
    // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is
    // from innermost to outermost; thus, we need to reverse the order
    shape_t *lhs_shape = new shape_t[lhs_tiramisu_buffer->get_dim_sizes().size()];
    int stride = 1;
    std::vector<std::string> strides_vector;

    if (lhs_tiramisu_buffer->has_constant_extents()) {
      for (int i = 0; i < lhs_buf_dims; i++) {
        lhs_shape[i].min = 0;
        int dim_idx = lhs_tiramisu_buffer->get_dim_sizes().size() - i - 1;
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
        stride_expr = stride_expr + " * " + 
          generator::cuda_expr_from_tiramisu_expr(this->get_function(), empty_index_expr,
                                                  replace_original_indices_with_transformed_indices(
                                                                                                    lhs_tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                                    this->get_iterators_map()),
                                                  this);
      }
    }

    // The number of dimensions in the Halide buffer should be equal to
    // the number of dimensions of the lhs_access function.
    assert(lhs_buf_dims == num_lhs_access_dims);
    assert(this->index_expr[0] != NULL);
    std::string lhs_index = "";
    if (lhs_tiramisu_buffer->has_constant_extents()) {
      lhs_index = generator::linearize_access_cuda(lhs_buf_dims, lhs_shape, this->index_expr[0]);
    } else { 
      lhs_index = tiramisu::generator::linearize_access_cuda(lhs_buf_dims, strides_vector, this->index_expr[0]);
    }

    this->index_expr.erase(this->index_expr.begin());
    assert(this->lhs_access_type == tiramisu::o_access && "Only o_access allowed for LHS type in gpu kernel generation");

    // Replace the RHS expression to the transformed expressions.
    // We do not need to transform the indices of expression (this->index_expr), because in Tiramisu we assume
    // that an access can only appear when accessing a computation. And that case should be handled in the following transformation
    // so no need to transform this->index_expr separately.
    tiramisu::expr tiramisu_rhs = replace_original_indices_with_transformed_indices(this->expression,
                                                                                    this->get_iterators_map());
    std::string rhs_expr = generator::cuda_expr_from_tiramisu_expr(this->get_function(), this->index_expr, tiramisu_rhs, this);
    std::string store_expr = lhs_buffer_name + "[" + lhs_index + "] = " + rhs_expr + ";";    
    return store_expr;
  }

  std::string tiramisu::generator::cuda_expr_from_tiramisu_expr(const tiramisu::function *fct, std::vector<isl_ast_expr *> &index_expr,
                                                                const tiramisu::expr &tiramisu_expr, tiramisu::computation *comp) {
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
            result = op0 + " && " +op1;
            break;
          case tiramisu::o_logical_or:
            result = op0 + " || " + op1;
            break;
          case tiramisu::o_max:
            assert(false && "max not supported in kernel");
            break;
          case tiramisu::o_min:
            assert(false && "min not supported in kernel");
            break;
          case tiramisu::o_minus:
            result = "-" + op0;
            break;
          case tiramisu::o_add:
            result = op0 + " + " + op1;
            break;
          case tiramisu::o_sub:
            result = op0 + " - " + op1;
            break;
          case tiramisu::o_mul:
            result = op0 + " * " + op1;
            break;
          case tiramisu::o_div:
            result = op0 + " / " + op1;
            break;
          case tiramisu::o_mod:
            assert(false && "modulo not supported in kernel");
            break;
          case tiramisu::o_select:
            result = op0 + " ? " + op1 + " : " + op2;
            break;
          case tiramisu::o_lerp:
            assert(false && "lerp not supported in kernel");
            break;
          case tiramisu::o_cond:
            tiramisu::error("Code generation for o_cond is not supported yet.", true);
            break;
          case tiramisu::o_le:
            result = op0 + " <= " + op1;
            break;
          case tiramisu::o_lt:
            result = op0 + " < " + op1;
            break;
          case tiramisu::o_ge:
            result = op0 + " >= " + op1;
            break;
          case tiramisu::o_gt:
            result = op0 + " > " + op1;
            break;
          case tiramisu::o_logical_not:
            result = "!" + op0;
            break;
          case tiramisu::o_eq:
            result = op0 + " == " + op1;
            break;
          case tiramisu::o_ne:
            result = op0 + " != " + op1;
            break;
          case tiramisu::o_bitwise_and:
            result = op0 + " & " + op1;
            break;
          case tiramisu::o_bitwise_or:
            result = op0 + " | " + op1;
            break;
          case tiramisu::o_bitwise_xor:
            result = op0 + " ^ " + op1;
            break;
          case tiramisu::o_bitwise_not:
            result = "~" + op0;
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
            result = op0 + ">>" + op1;
            break;
          case tiramisu::o_left_shift:
            result = op0 + " << " + op1;
            break;
          case tiramisu::o_floor:
            if (tiramisu_expr.get_data_type() == tiramisu::p_float32) {
              result = "floorf(" + op0 + ")";
            } else {
              result = "floor(" + op0 + ")";
            }
            break;
          case tiramisu::o_cast:
            result = "(" + c_type_from_tiramisu_type(tiramisu_expr.get_data_type()) + ")" + op0;
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
                  stride_expr = stride_expr + " * " + generator::cuda_expr_from_tiramisu_expr(fct, empty_index_expr,
                                                                                              tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                              comp);
                }
              }
              if (tiramisu_expr.get_op_type() == tiramisu::o_access ||
                  tiramisu_expr.get_op_type() == tiramisu::o_address_of ||
                  tiramisu_expr.get_op_type() == tiramisu::o_lin_index) {
                  std::string index;

                  // If index_expr is empty, and since tiramisu_expr is
                  // an access expression, this means that index_expr was not
                  // computed using the statement generator because this
                  // expression is not an expression that is associated with
                  // a computation. It is rather an expression used by
                  // a computation (for example, as the size of a buffer
                  // dimension). So in this case, we retrieve the indices directly
                  // from tiramisu_expr.
                  // The possible problem in this case, is that the indices
                  // in tiramisu_expr cannot be adapted to the schedule if
                  // these indices are i, j, .... This means that these
                  // indices have to be constant value only. So we check for this.

                  // If the consumer is in a distributed loop and the computation it is accessing is not,
                  // then we need to remove the distributed iterator from the linearized access because
                  // it is just a rank
                  if (index_expr.size() == 0) {
                    for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                      // Actually any access that does not require
                      // scheduling is supported but currently we only
                      // accept literal constants as anything else was not
                      // needed til now.
                      assert(tiramisu_expr.get_access()[i].is_constant() && "Only constant accesses are supported.");
                    }
                    if (tiramisu_buffer->has_constant_extents()) {
                      index = generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                               shape, tiramisu_expr.get_access());
                    } else {
                      index = tiramisu::generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                                                         strides_vector, tiramisu_expr.get_access());
                    }
                  } else {
                    if (tiramisu_buffer->has_constant_extents()) {
                      index = generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                               shape, index_expr[0]);
                    } else {
                      index = tiramisu::generator::linearize_access_cuda(tiramisu_buffer->get_dim_sizes().size(),
                                                                         strides_vector, index_expr[0]);
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
      result = c_type_from_tiramisu_type(tiramisu_expr.get_data_type()) + " " + tiramisu_expr.get_name();
    } else {
      tiramisu::str_dump("tiramisu type of expr: ",
                         str_from_tiramisu_type_expr(tiramisu_expr.get_expr_type()).c_str());
      tiramisu::error("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
    }    
    return result;
  }

}
