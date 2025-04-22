from ..base import OPERATOR, MteOp, ModelReader, TFLiteReader


def get_op_options(op_idx,model_reader:ModelReader):
    mte_options={
        "mode":"nearest",
    }
    if isinstance(model_reader,TFLiteReader):
        if model_reader.get_op_name(op_idx)=="RESIZE_NEAREST_NEIGHBOR":
            mte_options["mode"]="nearest"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return mte_options

@OPERATOR.register_operator("RESIZE_NEAREST_NEIGHBOR")
def resize_nearest_parse_func(op_idx, model_reader:ModelReader):
    if model_reader.get_op_name(op_idx)=="RESIZE_NEAREST_NEIGHBOR":
        op = Interpolate(op_idx,mode="resize_nearest")
    else:
        raise NotImplementedError
    return op

class Interpolate(MteOp):
    _inplace=False
    def __init__(self, op_idx,mode):
        super().__init__(op_idx)
        self.op_idx = op_idx
        self.mode = mode

    def op_post_process(self):
        self.input_tensors[1].in_ram=False
        return []

    def get_call_func(self):
        input_tensor=self.input_tensors[0]
        size_tensor=self.input_tensors[1]
        output_tensor=self.output_tensors[0]
        if self.mode=="resize_nearest":
            func=(f"resize_nearest("
                  f"{input_tensor.mem_symbol},{input_tensor.shape[1]},{input_tensor.shape[2]},{input_tensor.shape[3]},"
                  f"{output_tensor.mem_symbol},{output_tensor.shape[2]},{output_tensor.shape[3]}"
                  f")")
            return func
        else:
            raise NotImplementedError


