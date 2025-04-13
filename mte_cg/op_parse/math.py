import numpy as np
from ..base import OPERATOR, MteOp, ModelReader,MteTensor


@OPERATOR.register_operator("LOGISTIC")
def resize_nearest_parse_func(op_idx, model_reader:ModelReader):
    op = Sigmoid(op_idx)
    return op


@OPERATOR.register_operator("QUANTIZE")
def quantize_parse_func(op_idx, model_reader:ModelReader):
    op = Quantize(op_idx)
    return op


@OPERATOR.register_operator("TANH")
def quantize_parse_func(op_idx, model_reader:ModelReader):
    op = Tanh(op_idx)
    return op


@OPERATOR.register_operator("DEQUANTIZE")
def quantize_parse_func(op_idx, model_reader:ModelReader):
    op = DeQuantize(op_idx)
    return op

@OPERATOR.register_operator("SOFTMAX")
def softmax_parse_func(op_idx, model_reader:ModelReader):
    op = Softmax(op_idx)
    return op


@OPERATOR.register_operator("ADD")
def add_parse_func(op_idx, model_reader:ModelReader):
    op = Add(op_idx)
    return op

@OPERATOR.register_operator("SUB")
def sub_parse_func(op_idx, model_reader:ModelReader):
    op = Sub(op_idx)
    return op

@OPERATOR.register_operator("MUL")
def mul_parse_func(op_idx, model_reader:ModelReader):
    op = Mul(op_idx)
    return op

class Add(MteOp):
    _inplace=False
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.op_idx = op_idx

    def get_call_func(self):
        input_tensor0=self.input_tensors[0]
        input_tensor1=self.input_tensors[1]
        output_tensor=self.output_tensors[0]
        scale0=input_tensor0.scale/output_tensor.scale
        scale1=input_tensor1.scale/output_tensor.scale
        offset=output_tensor.offset-scale0*input_tensor0.offset-scale1*input_tensor1.offset
        scale0=int(np.round(scale0*2**14).astype("int32"))
        assert scale0<2**15
        scale1=int(np.round(scale1*2**14).astype("int32"))
        assert scale1<2**15
        offset=int(np.round(offset*2**14).astype("int32"))+2**13
        func=(f"add("
              f"{input_tensor0.mem_symbol},{scale0},"
              f"{input_tensor1.mem_symbol},{scale1},"
              f"{input_tensor0.size},"
              f"{output_tensor.mem_symbol},{offset}"
              f")")
        return func

class Sub(MteOp):
    _inplace=False
    def __init__(self, op_idx):
        super().__init__(op_idx)

class Mul(MteOp):
    _inplace=False
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.op_idx = op_idx



class Softmax(MteOp):
    _inplace=False
    def __init__(self, op_idx=None):
        super().__init__(op_idx)

    def op_post_process(self):
        input_tensor=self.input_tensors[0]
        output_vars0=[]
        for i in range(0,256):
            input_var=np.array(i).astype("int8")
            input_var=input_var.astype("int32")
            input_var_fp=input_tensor.scale*(input_var-input_tensor.offset)
            input_var_fp_exp=np.exp(input_var_fp-input_tensor.scale*(127-input_tensor.offset))
            output_vars0.append(input_var_fp_exp)
        output_vars1=[]
        for var in output_vars0:
            output_vars1.append(var/self.output_tensors[0].scale)
        map0_tensor=MteTensor(
            tensor_idx=None,
            dtype="float32",
            shape=(256,),
            tensor_type="extra_weight",
            data=np.array(output_vars0,dtype="float32"),
        )
        map1_tensor=MteTensor(
            tensor_idx=None,
            dtype="float32",
            shape=(256,),
            tensor_type="extra_weight",
            data=np.array(output_vars1,dtype="float32"),
        )
        return [map0_tensor,map1_tensor]

    def get_call_func(self):
        nums0=self.input_tensors[0].size//self.input_tensors[0].shape[-1]
        n=self.input_tensors[0].shape[-1]
        func=(f"mte_softmax("
              f"{self.input_tensors[0].mem_symbol},{nums0},{n},1,"
              f"{self.input_tensors[1].var_symbol},{self.input_tensors[1].mem_symbol},"
              f"{self.input_tensors[2].var_symbol},{self.input_tensors[2].mem_symbol},"
              f"{self.output_tensors[0].mem_symbol},{int(self.output_tensors[0].offset)}"
              f")")
        return func




def get_pre_weight_tensor(input_tensor,output_tensor,kernel_func):
    output_vars=[]
    for i in range(256):
        input_var=np.array(i).astype("int8")
        input_var=input_var.astype("int32")
        input_var_fp=input_tensor.scale*(input_var-input_tensor.offset)
        output_var_fp=kernel_func(input_var_fp)
        output_var=output_var_fp/output_tensor.scale+output_tensor.offset
        output_var=np.round(output_var).astype("int32")
        output_var=np.clip(output_var,-128,127).astype("int8")
        output_vars.append(output_var)
    data=np.array(output_vars,dtype="int8")
    tensor=MteTensor(
        tensor_idx=None,
        dtype="int8",
        shape=(256,),
        tensor_type="extra_weight",
        data=data,
    )
    return tensor

class Tanh(MteOp):
    _inplace=True
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.op_idx = op_idx
        self.ins_inplace=True

    def op_post_process(self):
        tensor=get_pre_weight_tensor(self.input_tensors[0],self.output_tensors[0],np.tanh)
        return [tensor]

    def get_call_func(self):
        return (f"mte_tanh("
                f"{self.input_tensors[0].mem_symbol},{self.input_tensors[0].size},"
                f"{self.input_tensors[1].var_symbol},{self.input_tensors[1].mem_symbol},"
                f"{self.output_tensors[0].mem_symbol}"
                f")")

class Quantize(MteOp):
    _inplace=True
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.op_idx = op_idx
        self.ins_inplace=True

    def op_post_process(self):
        tensor=get_pre_weight_tensor(self.input_tensors[0],self.output_tensors[0],lambda x:x)
        return [tensor]

    def get_call_func(self):
        return (f"mte_quantize("
                f"{self.input_tensors[0].mem_symbol},{self.input_tensors[0].size},"
                f"{self.input_tensors[1].var_symbol},{self.input_tensors[1].mem_symbol},"
                f"{self.output_tensors[0].mem_symbol}"
                f")")

class DeQuantize(MteOp):
    _inplace=False
    def __init__(self, op_idx):
        super().__init__(op_idx)

    def op_post_process(self):
        input_tensor=self.input_tensors[0]
        output_vars=[]
        for i in range(256):
            input_var=np.array(i).astype("int8")
            input_var=input_var.astype("int32")
            input_var_fp=input_tensor.scale*(input_var-input_tensor.offset)
            output_var_fp=input_var_fp
            output_vars.append(output_var_fp)
        data=np.array(output_vars,dtype="float32")
        tensor=MteTensor(
            tensor_idx=None,
            dtype="float32",
            shape=(256,),
            tensor_type="extra_weight",
            data=data,
        )
        return [tensor]

    def get_call_func(self):
        return (f"mte_dequantize("
                f"{self.input_tensors[0].mem_symbol},{self.input_tensors[0].size},"
                f"{self.input_tensors[1].var_symbol},{self.input_tensors[1].mem_symbol},"
                f"{self.output_tensors[0].mem_symbol}"
                f")")

class Sigmoid(MteOp):
    _inplace=True
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.ins_inplace=True

    def op_post_process(self):
        tensor=get_pre_weight_tensor(self.input_tensors[0],self.output_tensors[0],lambda x:1/(1+np.exp(-x)))
        return [tensor]

    def get_call_func(self):
        return (f"mte_sigmoid("
                f"{self.input_tensors[0].mem_symbol},{self.input_tensors[0].size},"
                f"{self.input_tensors[1].var_symbol},{self.input_tensors[1].mem_symbol},"
                f"{self.output_tensors[0].mem_symbol}"
                f")")

class Identity(MteOp):
    _inplace=True
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.ins_inplace=True

    def get_call_func(self):
        return f"/*identity do not change data*/"