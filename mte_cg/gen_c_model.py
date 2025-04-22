import os

from .gen_c_code import gen_model_call, gen_models_header_file, gen_models_c_file, gen_ops_c_file
from .memory import allocate_tensor_memory
from .base import MteGraphFromTflite


def model_parse(model_path):
    if model_path.endswith('.tflite'):
        mte_graph=MteGraphFromTflite(model_path)
    else:
        raise NotImplementedError
    return mte_graph

def gen_codes_from_model(model_path,model_name,codes_path,work_path,fit_method="first_fit"):
    if model_name is None:
        model_name=os.path.split(model_path)[1].split('.')[0]
    mte_graph=model_parse(model_path)
    peak_mem=allocate_tensor_memory(mte_graph,fit_method,vis_name=model_name,vis_path=work_path)
    os.makedirs(codes_path, exist_ok=True)
    call_model_file_path=os.path.join(codes_path, f"{model_name}.c")
    gen_model_call(model_name,mte_graph,call_model_file_path)
    return mte_graph,model_name,peak_mem


def gen_codes_from_models(model_paths,codes_path,work_path,model_names=None,workers=8):
    if isinstance(model_paths,str):
        model_paths=[model_paths]
    if isinstance(model_names,str):
        model_names=[model_names]
    if model_names is not None:
        assert len(model_paths)==len(model_names)
    model_infos=[]
    os.makedirs(codes_path, exist_ok=True)
    os.makedirs(work_path, exist_ok=True)
    # process = multiprocessing.Pool()
    # results=[]
    # for i in range(0, 3):
    #     result=process.apply_async(gen_codes_from_model, args=(model_paths[i],codes_path))
    #     results.append(result)
    # process.close()
    # process.join()
    # for result in results:
    #     model_infos.append(result.get())

    for i,model_path in enumerate(model_paths):
        model_info=gen_codes_from_model(model_path,model_names[i],codes_path,work_path)
        model_infos.append(model_info)
    gen_models_header_file(model_infos,codes_path)
    gen_models_c_file(model_infos,codes_path)
    gen_ops_c_file(model_infos,codes_path)




if __name__ == '__main__':
    model_paths=[
        "../temp/yolov10t.tflite",
        "../temp/model_front.tflite",
        "../temp/model_post.tflite",
        "../temp/yolo_fastestv2.tflite",
        "../temp/ghost.tflite",
    ]
    model_names=[
        "network_1",
        "network_2",
        "network_3",
        "network_4",
        "network_5",
    ]
    gen_codes_from_models(model_paths,"../temp/c_codes","../temp",model_names=model_names)


