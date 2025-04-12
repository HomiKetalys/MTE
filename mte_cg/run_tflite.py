import onnxruntime as ort
import tensorflow as tf
import numpy as np

class tfOrtModelRuner():
    def __init__(self, model_path: str):
        if model_path.endswith(".tflite"):
            self.model_interpreter = tf.lite.Interpreter(model_path=model_path,experimental_preserve_all_tensors=True)
            self.model_interpreter.allocate_tensors()
            self.model_input_details = self.model_interpreter.get_input_details()[0]
            self.model_output_details = self.model_interpreter.get_output_details()
            self.model_type = 1
        else:
            self.ort_sess = ort.InferenceSession(model_path)
            self.model_type = 0

    def __call__(self, x):
        if self.model_type == 0:
            return self.ort_sess.run(None, {'input.1': x})[0]
        else:
            self.model_interpreter.set_tensor(self.model_input_details['index'], x)
            self.model_interpreter.invoke()
            out_list = []
            for output_details in self.model_output_details:
                out_list.append(self.model_interpreter.get_tensor(output_details['index']))
            if len(out_list) == 1:
                return out_list[0]
            else:
                return out_list

if __name__ == '__main__':
    inp=np.zeros((256,256,3),dtype="int8")
    inp[:,:,0]=-70
    inp[:,:,1]=10
    inp[:,:,2]=80
    inp=inp[None,:,:,:]
    model=tfOrtModelRuner(model_path="../temp/yolo_fastestv2.tflite")
    oup=model(inp)
    print(oup)


