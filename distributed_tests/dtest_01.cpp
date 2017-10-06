//
// Created by Jessica Ray on 10/3/17.
//

#include <isl/set.h>
#include <tiramisu/debug.h>
#include <tiramisu/core.h>

// Test to distribute addition of a scalar and a matrix. Splits across 2 nodes, A and B. Data is assumed to start
// completely on node A, so the appropriate part needs to be transferred to node B. Results gathered back on node A.
// Uses simplest configurations for everything

using namespace tiramisu;

/*
distpar for (int i=0; i<160; i++)
  for (int j=0; j<160; j++)
     S0[i,j] = x[i,j] + 4;
*/

typedef std::pair<std::string, std::string> ppair;
typedef std::vector<std::vector<computation *>> pred_group;

int main(int argc, char **argv) {

    global::set_default_tiramisu_options();

    function dtest_01("dtest_01");

    // Layer 1

    computation x_input("{x_input[i,j]: 0<=i<160 and 0<=j<160}", expr(), false, p_float32, &dtest_01);
    expr e = expr(x_input(var("i"), var("j")) + expr((float_t) 4));
    computation S("{S[i,j]: 0<=i<160 and 0<=j<160}", e, true, p_float32, &dtest_01);

    // Layer 2

    std::vector<computation *> parts = S.partition<computation>({ppair("SA", "{S[i,j]: 0<=i<80 and 0<=j<160}"),
                                                                 ppair("SB", "{S[i,j]: 80<=i<160 and 0<=j<160}")});

    // Channel attributes
    channel chan_sync_block("chan_sync_block", p_float32, {FIFO, SYNC, BLOCK, MPI});

    // Create the communication pairs
    send_recv nAnB = computation::create_transfer("{[i,j]: 80<=i<160 and 0<=j<160}", "sendAB", "recvAB",
                                                  &chan_sync_block, &chan_sync_block, x_input(var("i"), var("j")),
                                                  {parts[1]}, &dtest_01);
    send_recv nBnA = computation::create_transfer("{[i,j]: 80<=i<160 and 0<=j<160}", "sendBA", "recvBA",
                                                  &chan_sync_block, &chan_sync_block, (*parts[1])(var("i"), var("j")),
                                                  {}, &dtest_01);

    // Schedule everything that's on node A
    nAnB.s->set_low_level_schedule("{sendAB[i,j]->sendAB[0,0,i,0,j,0]}");
    nBnA.r->set_low_level_schedule("{recvBA[i,j]->recvBA[0,2,i,0,j,0]}");
    parts[0]->set_low_level_schedule("{SA[i,j]->SA[0,1,i,0,j,0]}");

    // Now on node B
    nAnB.r->set_low_level_schedule("{recvAB[i,j]->recvAB[0,0,i,0,j,0]}");
    nBnA.s->set_low_level_schedule("{sendBA[i,j]->sendBA[0,2,i,0,j,0]}");
    parts[1]->set_low_level_schedule("{SB[i,j]->SB[0,1,i,0,j,0]}");

    // Assign predicates to break up into the nodes
    pred_group groups = {{&x_input, nAnB.s, nBnA.r, parts[0]}, {nAnB.r, nBnA.s, parts[1]}};
    computation::distribute(groups, {0, 1});

    // Layer 3

    // TODO use collapsing to create the appropriate message size

    // Create buffers
    buffer x_input_buff("x_input_buff", {160, 160}, p_float32, a_input, &dtest_01);
    buffer SA_buff("SA_buff", {160, 160}, p_float32, a_output, &dtest_01);
    buffer SB_buff("SB_buff", {80, 160}, p_float32, a_temporary, &dtest_01);
    buffer nB_recv_buff("nB_recv_buff", {80, 160}, p_float32, a_temporary, &dtest_01);

    // Set the access
    x_input.set_access("{x_input[i,j]->x_input_buff[i,j]}");
    parts[0]->set_access("{SA[i,j]->SA_buff[i,j]}");
    parts[1]->set_access("{SB[i,j]->SB_buff[i-80,j]}");
    nAnB.r->set_access("{recvAB[i,j]->nB_recv_buff[i-80,j]}");
    nBnA.r->set_access("{recvBA[i,j]->SA_buff[i,j]}");

    // Code generation
    dtest_01.gen_time_space_domain();
    dtest_01.set_arguments({&x_input_buff, &SA_buff});
    dtest_01.lift_ops_to_library_calls();
    dtest_01.gen_isl_ast();
    dtest_01.gen_halide_stmt();
    dtest_01.dump_halide_stmt();
    dtest_01.gen_halide_obj("build/generated_fct_dtest_01.o");


}
