import os

from .mte_base import MteGraph,MteTensor,BasicOp

c_type_dict={
    "int32":"int32_t",
    "uint32":"uint32_t",
    "int8":"int8_t",
    "int16":"int16_t",
    "uint16":"uint16_t",
    "float32":"float",
    "uint8":"uint8_t",
}
def gen_model_call(model_name,mte_graph:MteGraph,file_path):
    func="#include \"mte_models.h\"\n"
    func+=f"#include \"{model_name}_params.h\"\n"
    func+=f"void {model_name}(){{\n"
    for run_idx in mte_graph.run_seq:
        mte_op:BasicOp=mte_graph.get_op(run_idx)
        func+=f"    /*op idx:{mte_op.op_idx},op_name:{mte_op.__class__.__name__}*/\n"
        func+="    "+mte_op.get_call_func()
        func+=";\n"
    func+="}"
    f=open(file_path,"w")
    f.write(func)
    f.close()

def gen_models_header_file(model_infos,file_path):
    with open(os.path.join(file_path,f"mte_models.h"),"w") as f0:
        f0.write("#include \"mte_ops.h\"\n")
        peak_mems=[]
        for mte_graph,model_name,peak_mem in model_infos:
            mte_graph:MteGraph=mte_graph
            with open(os.path.join(file_path,f"{model_name}_params.h"),"w") as f1:
                tensor_idxes=mte_graph.get_tensor_idxes()
                f1.write("#include \"mte_ops.h\"\n")
                for tensor_idx in tensor_idxes:
                    tensor:MteTensor=mte_graph.get_tensor(tensor_idx)
                    if tensor.data is not None:
                        params_str="static const "+c_type_dict[tensor.dtype]+" "+tensor.var_symbol+f"[{tensor.size}]={{"
                        for i,ele in enumerate(tensor.data.flatten()):
                            params_str+=f"{ele}"
                            if i+1<len(tensor.data.flatten()):
                                params_str+=","
                        params_str+="};\n"
                        f1.write(params_str)
            input_tensor:MteTensor=mte_graph.get_model_input_tensors()[0]
            output_tensor=mte_graph.get_model_output_tensors()[0]
            f0.write(f"void {model_name}();\n")
            f0.write(f"{c_type_dict[input_tensor.dtype]} *get_{model_name}_input_addr();\n")
            f0.write(f"{c_type_dict[output_tensor.dtype]} *get_{model_name}_output_addr();\n")
            peak_mems.append(peak_mem)
        max_peak_mem=max(peak_mems)
        f0.write(f"#define MAX_MEM_SIZE {max_peak_mem}\n")
        f0.write("extern int8_t *mte_mem;\n")
        f0.write("void set_mte_mem_addr(int8_t *addr);\n")


def gen_models_c_file(model_infos,file_path):
    with open(os.path.join(file_path,f"mte_models.c"),"w") as f0:
        f0.write("#include \"mte_models.h\"\n"
                 "int8_t *mte_mem;")
        f0.write("void set_mte_mem_addr(int8_t *addr){\n"
                 "    mte_mem=addr;\n"
                 "}\n")
        for mte_graph,model_name,peak_mem in model_infos:
            mte_graph:MteGraph=mte_graph
            input_tensor:MteTensor=mte_graph.get_model_input_tensors()[0]
            output_tensor=mte_graph.get_model_output_tensors()[0]
            f0.write((f"{c_type_dict[input_tensor.dtype]} *get_{model_name}_input_addr(){{\n"
                      f"    return {input_tensor.mem_symbol};\n"
                      f"}}\n"))
            f0.write((f"{c_type_dict[output_tensor.dtype]} *get_{model_name}_output_addr(){{\n"
                      f"    return {output_tensor.mem_symbol};\n"
                      f"}}\n"))
