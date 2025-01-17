#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <list>
#include "pin.H"

static std::unordered_map<ADDRINT, std::string> str_of_ins_at;

FILE * trace;

// just want a hash table
static std::unordered_map<uint64_t, uint64_t> addrs_list;

static int8_t cache_state[64] = {0};
int timer = 0;

uint64_t func_start = 0x0;
uint64_t func_end = 0x0;
/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE,  "pintool",
    "o", "", "specify file name for MyPinTool output");

KNOB<std::string>   KnobAddrsFile(KNOB_MODE_WRITEONCE,  "pintool",
    "addrs_file", "0x422860", "file path");

KNOB<uint64_t>   KnobFuncStart(KNOB_MODE_WRITEONCE,  "pintool",
    "func_start", "", "target func start addr");

KNOB<uint64_t>   KnobFuncEnd(KNOB_MODE_WRITEONCE,  "pintool",
    "func_end", "", "target func end addr");

/* ===================================================================== */
// Utilities
/* ===================================================================== */
int debug_count = 0;
VOID RecordAddr(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    uint32_t cache_idx = ((long)mem_addr & 0x0FC0ll) >> 6;
    cache_state [cache_idx] = 1;

    timer ++;
    if ((uint64_t)ip == func_end){  // if (timer == 260){
        timer = 0;
        for (int i=0; i<64; i++){
            fprintf(trace, "%d ", cache_state[i]);
        }
        fprintf(trace, "\n");
        memset(cache_state, 0, 64);
    }
    // fprintf(trace,"%p\n", ip);
    // fprintf(trace,"%llx\n", (uint64_t)mem_addr&0x0FC0ll);

    //std::string ins_str = str_of_ins_at[(ADDRINT)ip];
    //fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
    //fprintf(trace,"R:\t%p:\t%lu\n", mem_addr, mem_size);
    //fprintf(trace,"%p: R %p\n", ip, addr);
}

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID *v)
{
    ADDRINT ins_addr = INS_Address(ins);

    if (func_start > ins_addr || ins_addr > func_end){
        return;
    }

    str_of_ins_at[INS_Address(ins)] = INS_Disassemble(ins);
    std::string ins_asm = INS_Disassemble(ins);
    /*
    if (!(ins_asm.find("xmm")!=ins_asm.npos || ins_asm.find("ymm")!=ins_asm.npos)){
        return;
    }
    */
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.
    //
    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);

//    if (memOperands == 0){
//        INS_InsertPredicatedCall(
//            ins, IPOINT_BEFORE, (AFUNPTR)RecordInst,
//            IARG_INST_PTR,
//            IARG_END);
//    }

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++)
    {
        if (INS_MemoryOperandIsRead(ins, memOp))
        {
            // USIZE mem_size = INS_MemoryReadSize(ins); // DEPRECATED
            USIZE mem_size = INS_MemoryOperandSize(ins, memOp);
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordAddr,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_UINT64, mem_size,
                IARG_END);
        }
        // Note that in some architectures a single memory operand can be
        // both read and written (for instance incl (%eax) on IA-32)
        // In that case we instrument it once for read and once for write.
        if (INS_MemoryOperandIsWritten(ins, memOp))
        {
            // USIZE mem_size = INS_MemoryWriteSize(ins);  // DEPRECATED
            USIZE mem_size = INS_MemoryOperandSize(ins, memOp);
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordAddr,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_UINT64, mem_size,
                IARG_END);
        }
    }
}

VOID Fini(INT32 code, VOID *v)
{
    fprintf(trace, "#eof\n");
    fclose(trace);
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage()
{
    PIN_ERROR( "This Pintool prints a trace of memory addresses\n"
              + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

int ReadAddrList(){
    std::string addrs_file = KnobAddrsFile.Value();
    FILE *fp = fopen(addrs_file.c_str(),"r");
    //int count = 0;
    while(!feof(fp)){
        uint64_t current_addr;
        fscanf(fp, "%lx\n", &current_addr);
        addrs_list[current_addr] = current_addr;
        //printf("insert 0x%lx\n", current_addr); // debug
        //count += 1;
        //printf("%d\n", count);
    }
    return 0;
}


/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char *argv[])
{
    if (PIN_Init(argc, argv)) return Usage();

    std::string fileName = KnobOutputFile.Value();
    trace = fopen(fileName.c_str(), "w");
    //trace = fopen("pinatrace.out", "w");

    func_start = KnobFuncStart.Value();
    func_end = KnobFuncEnd.Value();
    // ReadAddrList();

    // debug
    //printf("output: %s, start: %p, end: %p\n", fileName.c_str(), (void *)start_addr, (void *)end_addr);

    /*
    std::unordered_map<uint64_t, uint64_t>::iterator iter;
    iter = addrs_list.begin();
    int count = 0;
    while(iter != addrs_list.end()) {
        printf("0x%lx\n", iter->second);
        iter++;
        count += 1;
        printf("%d\n", count);
    }
    return 0;
    */

    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();

    return 0;
}