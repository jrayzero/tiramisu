#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>


int main(int argc, char **argv)
{
	// Set default coli options.
	coli::global::set_default_coli_options();

	// Declare a function.
	coli::function fct("function0");
	coli::argument buf0(coli::inputarg, "buf0", 2, {10,10}, Halide::Int(8), NULL, &fct);

	// Declare the invariants of the function.  An invariant can be a symbolic
	// constant or a variable that does not change value during the
	// execution of the function.
	coli::invariant p0("N", Halide::Expr((int32_t) 10), &fct);

	// Declare the computations of the function fct.
	// To declare a computation, you need to provide:
	// (1) a Halide expression that represents the computation,
	// (2) an isl set representing the iteration space of the computation, and
	// (3) an isl context (which will be used by the ISL library calls).
	coli::computation computation0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", Halide::Expr((uint8_t) 3), &fct);
	coli::computation computation1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", &buf0, &fct);

	// Map the computations to a buffer (i.e. where each computation
	// should be stored in the buffer).
	// This mapping will be updated automaticall when the schedule
	// is applied.  To disable automatic data mapping updates use
	// coli::global::set_auto_data_mapping(false).
	computation0.set_access("{S0[i,j]->buf0[i,j]}");

	// Dump the iteration domain (input) for the function.
	fct.dump_iteration_domain();

	// Set the schedule of each computation.
	// The identity schedule means that the program order is not modified
	// (i.e. no optimization is applied).
	computation0.tile(0,1,2,2);
	computation0.tag_parallel_dimension(0);

	// Generate the time-processor domain of the computation
	// and dump it on stdout.
	fct.gen_time_processor_domain();
	fct.dump_time_processor_domain();

	// Generate an AST (abstract Syntax Tree)
	fct.gen_isl_ast();

	// Generate Halide statement for the function.
	fct.gen_halide_stmt();

	// If you want to get the generated halide statements, call
	// fct.get_halide_stmts().

	// Dump the Halide stmt generated by gen_halide_stmt()
	// for the function.
	fct.dump_halide_stmt();

	// Generate an object file from the function.
	fct.gen_halide_obj("build/generated_lib_tutorial_01.o");

	return 0;
}
