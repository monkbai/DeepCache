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
static int8_t profile_cache_state[64] = {0};
int timer = 0;
int mulps_count = 0;

int64_t cache_accesses = 0;
int64_t extra_accesses = 0;

uint64_t func_start = 0x0;
uint64_t func_end = 0x0;
uint64_t insert_point = 0x0;
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

KNOB<uint64_t>   KnobInsPnt(KNOB_MODE_WRITEONCE,  "pintool",
    "insert_point", "", "location to insert obfuscation code");

/* ===================================================================== */
// Utilities
/* ===================================================================== */
int debug_count = 0;
int stop_flag = 0;
VOID StopFlag(VOID * ip)
{
    if ((uint64_t)ip == func_end){
        stop_flag = 1;
        fprintf(trace, "stop at %p\n", ip);
    }
}

VOID RecordAddr(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (stop_flag)
        return;
    // debug
//    if (str_of_ins_at[(ADDRINT)ip].find("movaps") != std::string::npos && debug_count < 20){
//        printf("%p: %s\n%p, %ld\n", ip, str_of_ins_at[(ADDRINT)ip].c_str(), mem_addr, mem_size);
//        debug_count ++;
//    }

    if (cache_accesses % 512 == 0){
        int i = 0;
        for (; i < 64; i++){
            if (profile_cache_state [i] == 0){
                // extra_count ++;
                extra_accesses ++;
                cache_state[i] = 1;
            }
            else{
                profile_cache_state[i] = 0;
            }
        }
    }


    cache_accesses ++;

    uint32_t cache_idx = ((long)mem_addr & 0x0FC0ll) >> 6;
    cache_state [cache_idx] = 1;
    profile_cache_state [cache_idx] = 1;
    // TODO: recheck this for glow binary
//    uint32_t cache_end_idx = ( ((long)mem_addr+mem_size-1) & 0x0FC0ll) >> 6;
//    if (cache_idx != cache_end_idx){
//        printf("%p, %ld\n", mem_addr, mem_size);
//    }

    timer ++;
    if (timer == 2000){  // used to be 260
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

VOID Obfuscate(VOID * ip)
{
    if (stop_flag)
        return;

    int extra_count = 0;
    int i = 0;
    for (; i < 64; i++){
        if (profile_cache_state [i] == 0){
            extra_count ++;
            extra_accesses ++;
            cache_state[i] = 1;
        }
        else{
            profile_cache_state[i] = 0;
        }
    }
}

VOID Mulps(VOID * ip){
    return;
    mulps_count ++;
    
    if (mulps_count == 24){
        int i = 0;
        for (; i < 64; i++){
            if (profile_cache_state [i] == 0){
                // extra_count ++;
                extra_accesses ++;
                cache_state[i] = 1;
            }
            else{
                profile_cache_state[i] = 0;
            }
        }
        mulps_count = 0;
    }
   
}

VOID RecordReadAddr(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (stop_flag)
        return;
    uint32_t cache_idx = ((long)mem_addr & 0x0FC0ll) >> 6;
    
    if (timer >= 100000){ 
        return;
    }
    timer ++;
    fprintf(trace, "R %d\n", cache_idx);
}
VOID RecordWriteAddr(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (stop_flag)
        return;
    uint32_t cache_idx = ((long)mem_addr & 0x0FC0ll) >> 6;
    
    if (timer >= 100000){ 
        return;
    }
    timer ++;
    fprintf(trace, "W %d\n", cache_idx);
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
    // printf("%s\n", ins_asm.c_str());

    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    UINT32 memOp = 0;
    USIZE mem_size = 0;
    for (; memOp < memOperands; memOp++)
    {
        if (INS_MemoryOperandIsRead(ins, memOp))
        {
            // USIZE mem_size = INS_MemoryReadSize(ins); // DEPRECATED
            mem_size = INS_MemoryOperandSize(ins, memOp);
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
            mem_size = INS_MemoryOperandSize(ins, memOp);
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordAddr,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_UINT64, mem_size,
                IARG_END);
        }
    }
    if (ins_addr == func_end){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)StopFlag,
            IARG_INST_PTR,
            IARG_END);

    }

    // if (ins_addr == insert_point){
    //     INS_InsertPredicatedCall(
    //         ins, IPOINT_BEFORE, (AFUNPTR)Obfuscate,
    //         IARG_INST_PTR,
    //         IARG_END);

    // }

    if (ins_asm.find("mulps") != std::string::npos){
        INS_InsertPredicatedCall(
        ins, IPOINT_BEFORE, (AFUNPTR)Mulps,
        IARG_INST_PTR,
        IARG_END);
    }
}

VOID Fini(INT32 code, VOID *v)
{
    fprintf(trace, "#eof\n");
    fclose(trace);
    printf("cache_accesses: %ld\nextra_accesses: %ld\n", cache_accesses, extra_accesses);
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
    insert_point = KnobInsPnt.Value();

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