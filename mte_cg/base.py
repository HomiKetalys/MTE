import numpy as np
import tensorflow as tf
from tensorflow.lite.tools.visualize import CreateDictFromFlatbuffer, BuiltinCodeToName
from tensorflow.lite.tools.visualize import BuiltinCodeToName

graph_optimizers=[]
class MteGraph:
    def __init__(self,model_reader):
        self.model_reader = model_reader
        self.tensors_table:dict[int:MteTensor]=self.gen_tensor_table()
        self.ops_table: dict[int:MteOp]=self.gen_op_table()
        self.run_seq=model_reader.run_seq
        self.connected=False
        self.connect_table()
        self.add_extra_input_tensors()
        graph_optimizers.sort(key=lambda x:x[0])
        for graph_optimizer in graph_optimizers:
            graph_optimizer[1](self)
        self.fix_inplace()


    def get_model_input_tensors(self):
        return [self.tensors_table[idx] for idx in self.model_reader.get_model_inputs_idx()]

    def get_model_output_tensors(self):
        return [self.tensors_table[idx] for idx in self.model_reader.get_model_outputs_idx()]

    def get_op(self,op_idx):
        if op_idx in self.ops_table.keys():
            return self.ops_table[op_idx]
        else:
            return None

    def get_tensor(self,tensor_idx):
        if tensor_idx in self.tensors_table.keys():
            return self.tensors_table[tensor_idx]
        else:
            return None

    def get_tensor_idxes(self):
        return self.tensors_table.keys()

    def remove_op_input_tensor(self,op,tensor):
        op.remove_input_tensor(tensor)
        tensor.remove_to_op(op)
        self.check_and_remove_leaf_tensor(tensor)

    def replace_op(self,op0,op1):
        input_tensors:list[MteTensor]=op0.input_tensors
        assert len(op1.input_tensors)==0
        assert len(op1.output_tensors)==0
        for tensor in input_tensors:
            op1.input_tensors.append(tensor)
            op0.remove_input_tensor(tensor)
            tensor.add_to_op(op1)
            tensor.remove_to_op(op0)
        output_tensors:list[MteTensor]=op0.output_tensors
        for tensor in output_tensors:
            op1.output_tensors.append(tensor)
            op0.remove_output_tensor(tensor)
            tensor.add_from_op(op1)
            tensor.remove_from_op(op0)
        run_idx=self.run_seq.index(op0.op_idx)
        self.run_seq[run_idx]=op1.op_idx
        self.ops_table.pop(op0.op_idx)
        self.ops_table[op1.op_idx]=op1

    def check_and_remove_leaf_tensor(self,tensor):
        if len(tensor.to_ops)==0:
            self.tensors_table.pop(tensor.tensor_idx)
            if len(tensor.from_ops)>0:
                from_op=tensor.from_ops[0]
                from_op_input_tensors=from_op.input_tensors
                for from_op_input_tensor in from_op_input_tensors:
                    from_op_input_tensor.remove_to_op(from_op)
                    self.check_and_remove_leaf_tensor(from_op_input_tensor)
                self.ops_table.pop(from_op.op_idx)
                self.run_seq.remove(from_op.op_idx)


    def replace_op_input_tensor(self,op,tensor0,tensor1):
        op.replace_input_tensor(tensor0,tensor1)
        tensor1.add_to_op(op)
        tensor0.remove_to_op(op)
        self.check_and_remove_leaf_tensor(tensor0)


    def get_tensor_size(self,tensor_idx):
        return self.tensors_table[tensor_idx]["tensor"].aligned_size

    def get_tensor_mem_size(self,tensor_idx):
        return self.tensors_table[tensor_idx].mem_size

    def get_tensor_shape(self,tensor_idx):
        return self.tensors_table[tensor_idx].shape

    def get_tensor_type(self,tensor_idx):
        return self.tensors_table[tensor_idx].tensor_type

    def gen_tensor_table(self):
        tensors_table:dict={}
        for tensor_idx in self.model_reader.tensors_idx:
            tensors_table[tensor_idx]=MteTensor(
                tensor_idx=tensor_idx,
                dtype=self.model_reader.get_tensor_dtype(tensor_idx),
                shape=self.model_reader.get_tensor_shape(tensor_idx),
                tensor_type=self.model_reader.get_tensor_type(tensor_idx),
                data=self.model_reader.get_tensor_data(tensor_idx),
                scale=self.model_reader.get_tensor_scale(tensor_idx),
                offset=self.model_reader.get_tensor_offset(tensor_idx),
            )
        return tensors_table

    def gen_op_table(self):
        ops_table:dict={}
        for op_idx in self.model_reader.ops_idx:
            op_name = self.model_reader.get_op_name(op_idx)
            assert op_name in OPERATOR.operator_dict,f"unsupported operator: {op_name}"
            ops_table[op_idx]=OPERATOR.operator_dict[op_name](op_idx,self.model_reader)
        return ops_table

    def fix_inplace(self):
        for run_idx in self.run_seq:
            mte_op:MteOp=self.ops_table[run_idx]
            input_tensor:MteTensor=mte_op.input_tensors[0]
            for to_op in input_tensor.to_ops:
                if to_op.op_idx !=mte_op.op_idx:
                    to_op.ins_inplace=False


    def connect_table(self):
        for op_idx in self.model_reader.ops_idx:
            inputs_idx = self.model_reader.get_inputs_idx(op_idx)
            outputs_idx = self.model_reader.get_outputs_idx(op_idx)
            for idx in inputs_idx:
                assert idx in self.tensors_table.keys()
                self.tensors_table[idx].to_ops.append(self.ops_table[op_idx])
                self.ops_table[op_idx].input_tensors.append(self.tensors_table[idx])

            for idx in outputs_idx:
                assert idx in self.tensors_table.keys()
                self.tensors_table[idx].from_ops.append(self.ops_table[op_idx])
                self.ops_table[op_idx].output_tensors.append(self.tensors_table[idx])

    def add_extra_input_tensors(self):
        for op_idx in self.ops_table.keys():
            op=self.ops_table[op_idx]
            op.graph_post_process(self)

    def add_op_input_tensor(self,op,tensor):
        self.tensors_table[tensor.tensor_idx]=tensor
        op.input_tensors.append(tensor)
        tensor.to_ops.append(op)



class ModelReader:
    def get_model_inputs_idx(self, op_idx):
        raise NotImplementedError

    def get_model_outputs_idx(self, op_idx):
        raise NotImplementedError

    @property
    def ops_idx(self):
        raise NotImplementedError

    @property
    def run_seq(self):
        raise NotImplementedError

    @property
    def tensors_idx(self):
        raise NotImplementedError

    def get_tensor_shape(self,tensor_idx):
        raise NotImplementedError

    def get_tensor_type(self,tensor_idx):
        raise NotImplementedError

    def get_tensor_dtype(self,tensor_idx):
        raise NotImplementedError

    def get_op_name(self,op_idx):
        raise NotImplementedError

    def get_tensor_data(self,tensor_idx):
        raise NotImplementedError

    def get_options(self,op_idx):
        raise NotImplementedError

    def get_model_input_idxes(self):
        raise NotImplementedError

    def get_model_output_idxes(self):
        raise NotImplementedError


class TFLiteReader(ModelReader):
    def __init__(self,tflite_path):
        with open(tflite_path, 'rb') as f:
            model_buffer = f.read()
        # interpreter = tf.lite.Interpreter(model_content=model_buffer,experimental_preserve_all_tensors=True)
        self.interpreter = tf.lite.Interpreter(model_content=model_buffer)
        # interpreter.allocate_tensors()

        data = CreateDictFromFlatbuffer(model_buffer)
        self.op_codes = data['operator_codes']
        self.operators=data['subgraphs'][0]['operators']
        self.tensor_details=self.interpreter.get_tensor_details()

    def get_model_inputs_idx(self):
        return [inp["index"] for inp in self.interpreter.get_input_details()]

    def get_model_outputs_idx(self):
        return [oup["index"] for oup in self.interpreter.get_output_details()]


    def get_inputs_idx(self, op_idx):
        return self.operators[op_idx]['inputs']

    def get_outputs_idx(self, op_idx):
        return self.operators[op_idx]['outputs']

    @property
    def ops_idx(self):
        return list(range(len(self.operators)))

    @property
    def run_seq(self):
        return list(range(len(self.operators)))

    @property
    def tensors_idx(self):
        return list(range(len(self.tensor_details)))

    def get_tensor_shape(self,tensor_idx):
        return self.tensor_details[tensor_idx]['shape']

    def get_tensor_scale(self,tensor_idx):
        scale=self.tensor_details[tensor_idx]['quantization_parameters']['scales']
        if len(scale)>0:
            return scale
        return None

    def get_tensor_offset(self,tensor_idx):
        offset=self.tensor_details[tensor_idx]['quantization_parameters']['zero_points']
        if len(offset)>0:
            return offset
        return None

    def get_tensor_type(self,tensor_idx):
        try:
            self.interpreter.get_tensor(tensor_idx)
        except:
            tensor_type="activation"
        else:
            tensor_type="weight"
        return tensor_type

    def get_tensor_dtype(self,tensor_idx):
        dtype=self.tensor_details[tensor_idx]['dtype'].__name__
        if dtype=="int64":
            dtype="int32"
        return dtype

    def get_tensor_data(self,tensor_idx):
        try:
            data=self.interpreter.get_tensor(tensor_idx)
        except:
            data=None
        return data

    def get_op_name(self,op_idx):
        op=self.operators[op_idx]
        opcode_idx = op['opcode_index']
        op_code = self.op_codes[opcode_idx]['builtin_code']
        op_name = BuiltinCodeToName(op_code)
        return op_name




class MteGraphFromTflite(MteGraph):
    def __init__(self,tflite_path):
        super().__init__(TFLiteReader(tflite_path))


class MteOp:
    _inplace=True
    _extra_op_id=10000
    def __init__(self,op_idx):
        if op_idx is None:
            op_idx=MteOp._extra_op_id
            MteOp._extra_op_id+=1
        self.op_idx=op_idx
        self.ins_inplace=False
        self.input_tensors:list[MteTensor]=[]
        self.output_tensors:list[MteTensor]=[]

    @property
    def inplace_offset(self):
        return 0

    @property
    def inplace(self):
        return MteOp._inplace and self._inplace and self.ins_inplace

    def remove_input_tensor(self,tensor):
        is_in_tensors=False
        for i,input_tensor in enumerate(self.input_tensors):
            if input_tensor.tensor_idx==tensor.tensor_idx:
                self.input_tensors.pop(i)
                is_in_tensors=True
                break
        assert is_in_tensors

    def remove_output_tensor(self,tensor):
        is_in_tensors=False
        for i,output_tensor in enumerate(self.output_tensors):
            if output_tensor.tensor_idx==tensor.tensor_idx:
                self.output_tensors.pop(i)
                is_in_tensors=True
                break
        assert is_in_tensors

    def replace_input_tensor(self,tensor0,tensor1):
        is_in_tensors=False
        for i,input_tensor in enumerate(self.input_tensors):
            if input_tensor.tensor_idx==tensor0.tensor_idx:
                self.input_tensors[i]=tensor1
                is_in_tensors=True
                break
        assert is_in_tensors

    def replace_output_tensor(self,tensor0,tensor1):
        is_in_tensors=False
        for i,output_tensor in enumerate(self.output_tensors):
            if output_tensor.tensor_idx==tensor0.tensor_idx:
                self.output_tensors[i]=tensor1
                is_in_tensors=True
                break
        assert is_in_tensors

    def get_activation_inputs(self):
        activation_inputs=[]
        for tensor in self.input_tensors:
            if tensor.tensor_type=="activation":
                activation_inputs.append(tensor)
        return activation_inputs

    def get_activation_outputs(self):
        activation_outputs=[]
        for tensor in self.output_tensors:
            if tensor.tensor_type=="activation":
                activation_outputs.append(tensor)
        return activation_outputs

    def graph_post_process(self, mte_graph:MteGraph):
        for tensor in self.op_post_process():
            mte_graph.add_op_input_tensor(self,tensor)

    def op_post_process(self):
        return []

    def get_call_func(self):
        raise NotImplementedError
    
    def get_c_file_paths(self):
        return []


class MteTensor:
    weight_cache=True
    # weight_filter=lambda x:x>=4*1024
    weight_filter=None
    extra_tensor_idx=10000
    def __init__(self, tensor_idx, dtype, shape, tensor_type,data=None,scale=None,offset=None,in_ram=None):
        if tensor_idx is None:
            tensor_idx=MteTensor.extra_tensor_idx
            MteTensor.extra_tensor_idx+=1
        self.tensor_idx=tensor_idx
        self.dtype=dtype
        self.shape=shape
        self.tensor_type=tensor_type
        self.to_ops:list[MteOp]=[]
        self.from_ops:list[MteOp]=[]
        self.data_=data
        self.mem_addr=None
        self.offset=offset
        self.scale=scale
        self.in_ram=in_ram

    @property
    def data(self):
        if callable(self.data_):
            return self.data_()
        else:
            return self.data_

    @property
    def size(self):
        size=1
        for s in self.shape:
            size*=s
        return size

    @property
    def used_in_ram(self):
        if self.in_ram is not None:
            return self.in_ram
        if self.tensor_type in ["weight","extra_weight"] and MteTensor.weight_cache:
            if MteTensor.weight_filter is not None:
                return MteTensor.weight_filter(self.mem_size)
            else:
                return True
        elif self.tensor_type=="buffer" or self.tensor_type=="activation":
            return True
        else:
            return False

    @property
    def mem_size(self):
        mem_size=self.size
        if self.dtype in ["float32","int32","uint32"]:
            mem_size=mem_size*4
        elif self.dtype in ["int16","uint16"]:
            mem_size=mem_size*2
        elif self.dtype in ["int8","uint8"]:
            mem_size=mem_size
        else:
            raise NotImplementedError
        return mem_size

    def remove_to_op(self,op):
        self.to_ops.remove(op)

    def remove_from_op(self,op):
        self.from_ops.remove(op)

    def add_to_op(self,op):
        self.to_ops.append(op)

    def add_from_op(self,op):
        self.from_ops.append(op)

    @property
    def var_symbol(self):
        return f"{self.tensor_type}_{self.tensor_idx:04d}"

    @property
    def mem_symbol(self):
        assert self.used_in_ram
        return f"&mte_mem[{self.mem_addr}]"

class opRegistry(object):
    def __init__(self, name) -> None:
        self._name = name
        self._operator_dict = dict()

    def __len__(self):
        return len(self._operator_dict)

    @property
    def name(self):
        return self._name

    @property
    def operator_dict(self):
        return self._operator_dict

    def get(self, key):
        return self._operator_dict.get(key, None)

    def _register_operator(self, op_func, op_name=None):
        if (not isinstance(op_name, str)) or op_name is None:
            op_name = op_func.__name__

        op_func_=self._operator_dict.get(op_name, None)
        if op_func_:
            if op_func_!=op_func:
                raise KeyError(f'{op_name} is already registered in {self._name}')

        self._operator_dict[op_name] = op_func

    def register_operator(self, names):
        if isinstance(names, str):
            names=[names]
        # if op_func is not None:
        #     self._register_operator(op_func, name)
        #     return op_func

        def _register(func):
            for name in names:
                self._register_operator(func, name)
            return func

        return _register

OPERATOR = opRegistry("MteOP")