import os
import sys
import multiprocessing

from .gen_c_code import gen_model_call, gen_models_header_file, gen_models_c_file
from .graph_optimize.optimize_ops import optimize_reshape, remove_redundant_ops
from .graph_optimize.fuse_ops import fuse_zero_pad_with_conv
from .memory.memory_utils import allocate_tensor_memory
from .op_parse.parse_tools import MteGraphFromTflite, MteGraph


def model_parse(model_path):
    if model_path.endswith('.tflite'):
        mte_graph=MteGraphFromTflite(model_path)
    else:
        raise NotImplementedError
    return mte_graph

def gen_codes_from_model(model_path,codes_path):
    model_name=os.path.split(model_path)[1].split('.')[0]
    mte_graph=model_parse(model_path)
    fuse_zero_pad_with_conv(mte_graph)
    remove_redundant_ops(mte_graph)
    mte_graph.fix_inplace()
    peak_mem=allocate_tensor_memory(mte_graph,vis_name=model_name)
    os.makedirs(codes_path, exist_ok=True)
    call_model_file_path=os.path.join(codes_path, f"{model_name}.c")
    gen_model_call(model_name,mte_graph,call_model_file_path)
    return mte_graph,model_name,peak_mem


def gen_codes_from_models(model_paths,codes_path,workers=8):
    if isinstance(model_paths,str):
        model_paths=[model_paths]
    model_infos=[]

    # process = multiprocessing.Pool()
    # results=[]
    # for i in range(0, 3):
    #     result=process.apply_async(gen_codes_from_model, args=(model_paths[i],codes_path))
    #     results.append(result)
    # process.close()
    # process.join()
    # for result in results:
    #     model_infos.append(result.get())

    for model_path in model_paths:
        model_info=gen_codes_from_model(model_path,codes_path)
        model_infos.append(model_info)
    gen_models_header_file(model_infos,codes_path)
    gen_models_c_file(model_infos,codes_path)




if __name__ == '__main__':
    model_paths=[
        "../temp/yolov10.tflite",
        "../temp/model_front.tflite",
        "../temp/model_post.tflite",
    ]
    gen_codes_from_models(model_paths,"../temp/c_codes")


