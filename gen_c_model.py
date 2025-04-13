from mte_cg.gen_c_model import gen_codes_from_models
# If you have defined op, import the py file of your custom op
import examples.custom_op

if __name__ == '__main__':
    model_paths=[
        "./temp/yolov10t.tflite",
    ]
    model_names=[
        "network_1",
    ]
    gen_codes_from_models(model_paths,"./temp/c_codes","./temp",model_names=model_names)