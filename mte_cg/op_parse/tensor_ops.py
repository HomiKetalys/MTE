import numpy as np

from ..mte_base import OPERATOR, BasicOp, ModelReader,TFLiteReader, MteTensor

def get_concat_op_options(op_idx, model_reader:ModelReader):
    mte_options={
        "dim":-1,
    }
    if isinstance(model_reader,TFLiteReader):
        options=model_reader.operators[op_idx]["builtin_options"]
        mte_options["dim"]=options["axis"]
    else:
        raise NotImplementedError
    return mte_options

@OPERATOR.register_operator("CONCATENATION")
def concat_parse_func(op_idx, model_reader:ModelReader):
    options=get_concat_op_options(op_idx, model_reader)
    op = Concat(op_idx,options["dim"])
    return op


def get_pack_op_options(op_idx, model_reader:ModelReader):
    mte_options={
        "dim":-1,
        "repeat_nums":1,
    }
    if isinstance(model_reader,TFLiteReader):
        options=model_reader.operators[op_idx]["builtin_options"]
        mte_options["dim"]=options["axis"]
        mte_options["repeat_nums"]=options["values_count"]
    else:
        raise NotImplementedError
    return mte_options

@OPERATOR.register_operator("PACK")
def pack_parse_func(op_idx, model_reader:ModelReader):
    mte_options=get_pack_op_options(op_idx, model_reader)
    op = Pack(op_idx,mte_options["dim"],mte_options["repeat_nums"])
    return op

def get_gather_op_options(op_idx, model_reader:ModelReader):
    mte_options={
        "dim":-1,
    }
    if isinstance(model_reader,TFLiteReader):
        options=model_reader.operators[op_idx]["builtin_options"]
        mte_options["dim"]=options["axis"]
    else:
        raise NotImplementedError
    return mte_options


@OPERATOR.register_operator("GATHER")
def gather_parse_func(op_idx, model_reader:ModelReader):
    options=get_gather_op_options(op_idx, model_reader)
    op = Gather(op_idx,dim=options["dim"])
    return op

@OPERATOR.register_operator("RESHAPE")
def gather_parse_func(op_idx, model_reader:ModelReader):
    op = Reshape(op_idx)
    return op



@OPERATOR.register_operator("TRANSPOSE")
def pad_parse_func(op_idx, model_reader:ModelReader):
    op = Transpose(op_idx)
    return op

class Transpose(BasicOp):
    _inplace=False
    def __init__(self, op_idx=None):
        super().__init__(op_idx)

    def op_post_process(self):
        self.input_tensors[1].in_ram=False

        input_shape_tensor=MteTensor(
            tensor_idx=None,
            dtype="int32",
            shape=(len(self.input_tensors[0].shape),),
            tensor_type="extra_weight",
            data=np.array(self.input_tensors[0].shape, dtype="int32"),
            in_ram=False,
        )
        output_shape_tensor=MteTensor(
            tensor_idx=None,
            dtype="int32",
            shape=(len(self.output_tensors[0].shape),),
            tensor_type="extra_weight",
            data=np.array(self.output_tensors[0].shape, dtype="int32"),
            in_ram=False,
        )

        output_idx_buffer=MteTensor(
            tensor_idx=None,
            dtype="int32",
            shape=(len(self.output_tensors[0].shape),),
            tensor_type="buffer",
        )

        return [input_shape_tensor,output_shape_tensor,output_idx_buffer]

    def get_call_func(self):
        func=(f"transpose("
              f"{self.input_tensors[0].mem_symbol},{self.input_tensors[0].size},{self.input_tensors[1].var_symbol},{self.input_tensors[2].var_symbol},{self.input_tensors[3].var_symbol},{self.input_tensors[2].size},"
              f"{self.input_tensors[4].mem_symbol},"
              f"{self.output_tensors[0].mem_symbol}"
              f")")
        return func

class Concat(BasicOp):
    _inplace=False
    def __init__(self, op_idx=None,dim=None):
        super().__init__(op_idx)
        self.dim=dim

    def op_post_process(self):
        addr_tensor=MteTensor(
            tensor_idx=None,
            dtype="int32",
            shape=(len(self.input_tensors),),
            tensor_type="extra_weight",
            data=self.input_mem_addrs,
        )
        addr_tensor.in_ram=False
        shape=self.input_tensors[0].shape
        if self.dim==-1:
            dim=len(shape)
        else:
            dim=self.dim+1
        block_shape=shape[dim:]
        block_size=1
        for s in block_shape:
            block_size*=s
        block_size_tensor=MteTensor(
            tensor_idx=None,
            dtype="int32",
            shape=(len(self.input_tensors),),
            tensor_type="extra_weight",
            data=np.array([t.shape[self.dim]*block_size for t in self.input_tensors])
        )
        block_size_tensor.in_ram=False
        return [addr_tensor,block_size_tensor]

    def input_mem_addrs(self):
        mem_addrs=[t.mem_addr for t in self.input_tensors[:self.input_tensors[-1].size]]
        mem_addrs=np.array(mem_addrs,dtype="int32")
        return mem_addrs

    def get_call_func(self):
        shape=self.input_tensors[0].shape
        if self.dim==-1:
            dim=len(shape)-1
        else:
            dim=self.dim
        block_shape=shape[:dim]
        concat_nums=1
        for s in block_shape:
            concat_nums*=s
        func=(f"concat("
              f"{self.input_tensors[-2].var_symbol},{self.input_tensors[-1].var_symbol},{len(self.input_tensors)-2},{concat_nums},{self.output_tensors[0].mem_symbol}"
              f")")
        return func

class Pack(BasicOp):
    _inplace=False
    def __init__(self, op_idx=None,dim=0,repeat_nums=1):
        super().__init__(op_idx)
        self.dim=dim
        self.repeat_nums=repeat_nums

    def get_call_func(self):
        shape=self.input_tensors[0].shape
        if self.dim==-1:
            dim=len(shape)-1
        else:
            dim=self.dim
        block_shape=shape[dim:]
        block_size=1
        for s in block_shape:
            block_size*=s
        pack_nums=self.input_tensors[0].size//block_size
        func=(f"pack("
              f"{self.input_tensors[0].mem_symbol},{block_size},{pack_nums},{self.repeat_nums},{self.output_tensors[0].mem_symbol}"
              f")")
        return func

class Gather(BasicOp):
    _inplace=False
    def __init__(self, op_idx=None,dim=None):
        super().__init__(op_idx)
        self.dim=dim

    def op_post_process(self):
        self.input_tensors[1].in_ram=False
        return []

    def get_call_func(self):
        shape=self.input_tensors[0].shape
        if self.dim==-1:
            dim=len(shape)
        else:
            dim=self.dim+1
        block_shape=shape[dim:]
        block_size=1
        for s in block_shape:
            block_size*=s
        gather_block_size=block_size*self.input_tensors[0].shape[self.dim]
        gather_nums=self.input_tensors[0].size//gather_block_size
        func=(f"gather("
              f"{self.input_tensors[0].mem_symbol},{gather_nums},{gather_block_size},{block_size},"
              f"{self.input_tensors[1].var_symbol},{self.input_tensors[1].size},"
              f"{self.output_tensors[0].mem_symbol}"
              f")")
        return func

class Reshape(BasicOp):
    _inplace=True
    def __init__(self, op_idx=None):
        super().__init__(op_idx)
        self.ins_inplace=True

    def get_call_func(self):
        func="/*reshape op:reshape operator do not change data layout*/"
        return func

class Identity(BasicOp):
    _inplace=True
    def __init__(self, op_idx=None):
        super().__init__(op_idx)
        self.ins_inplace=True

    def get_call_func(self):
        func="/*identity op:identity operator do not change data layout*/"
        return func