from ..mte_base import OPERATOR, BasicOp, ModelReader, TFLiteReader


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
    options=get_op_options(op_idx,model_reader)
    op = Interpolate(op_idx,mode=options["mode"])
    return op

class Interpolate(BasicOp):
    _inplace=False
    def __init__(self, op_idx,mode):
        super().__init__(op_idx)
        self.op_idx = op_idx
        self.mode = mode

    def get_call_func(self):
        input_tensor=self.input_tensors[0]
        size_tensor=self.input_tensors[1]
        output_tensor=self.output_tensors[0]
        if self.mode=="nearest":
            func=(f"interpolate_nearest("
                  f"{input_tensor.mem_symbol},{size_tensor.var_symbol},{output_tensor.mem_symbol},{output_tensor.mem_symbol}"
                  f")")
            return func
        else:
            raise NotImplementedError


