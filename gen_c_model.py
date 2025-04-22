from mte_cg.gen_c_model import gen_codes_from_models
# If you have defined op, import the py file of your custom op
import examples.custom_op

if __name__ == '__main__':
    model_paths=[
        "./temp/yolov10t.tflite",
        "./temp/model_front.tflite",
        "./temp/model_post.tflite",
        "./temp/yolo_fastestv2.tflite",
        "./temp/ghost.tflite",
    ]
    model_names=[
        "network_1",
        "network_2",
        "network_3",
        "network_4",
        "network_5",
    ]
    gen_codes_from_models(model_paths,"./temp/c_codes","./temp",model_names=model_names)