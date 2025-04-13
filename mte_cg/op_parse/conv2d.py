import numpy as np
from toolz import isiterable

from ..base import OPERATOR, MteOp, ModelReader, TFLiteReader, MteTensor


def get_conv2d_op_options(op_idx, model_reader:ModelReader):
    mte_options={
        "stride":None,
        "dilation":None,
        "padding":None,
        "act":None,
        "group":1,
    }
    if isinstance(model_reader,TFLiteReader):
        op=model_reader.operators[op_idx]
        options=op["builtin_options"]
        mte_options["stride"]=(options["stride_h"],options["stride_w"])
        mte_options["dilation"]=(options["dilation_h_factor"],options["dilation_w_factor"])
        if options["padding"]==1:
            mte_options["padding"]="valid"
        elif options["padding"]==0:
            mte_options["padding"]="same"
        else:
            raise NotImplementedError
        if options["fused_activation_function"]==0:
            mte_options["act"]=None
        elif options["fused_activation_function"]==1:
            mte_options["act"]="relu"
        elif options["fused_activation_function"]==3:
            mte_options["act"]="relu6"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return mte_options



@OPERATOR.register_operator(["CONV_2D","DEPTHWISE_CONV_2D"])
def conv2d_parse_func(op_idx, model_reader:ModelReader):
    input_tensor_idxes = model_reader.get_inputs_idx(op_idx)
    out_channels,kernel_h,kernel_w,in_channels=model_reader.get_tensor_shape(input_tensor_idxes[1])
    options=get_conv2d_op_options(op_idx, model_reader)
    stride = options['stride']
    dilation = options['dilation']
    act = options['act']
    padding = options['padding']
    if isinstance(padding,str):
        if padding=="valid":
            padding=((0,0),(0,0))
        elif padding=="same":
            padding=((kernel_h//2,kernel_h//2),(kernel_w//2,kernel_w//2))
    op_name=model_reader.get_op_name(op_idx)
    if op_name in ["CONV_2D","DEPTHWISE_CONV_2D"]:
        group=options["group"] if op_name== "CONV_2D" else in_channels
        out_channels=out_channels*group if op_name== "DEPTHWISE_CONV_2D" else out_channels
    else:
        group=options["group"]

    op = Conv2d(op_idx,
                in_channels, out_channels,
                (kernel_h, kernel_w),
                stride,
                padding,
                dilation,
                group,
                act)
    return op

implemented_conv_configs=[
    [3,-1,((3,3),),((1,1),(2,2)),((1,1),),1],
    [-1,-1,((1,1),),-1,((1,1),),-1],
]

implemented_dwconv_configs=[
    [-1,-1,((3,3),(5,5),(7,7)),-1,((1,1),),-1],
]

class Conv2d(MteOp):
    _inplace=True
    def __init__(self, op_idx, in_channels, out_channels,kernel_size=(1, 1), stride=(1, 1), padding=((0,0),(0,0)), dilate=(1, 1), group=1, activation=None):
        super().__init__(op_idx)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilate = dilate
        self.group = group
        self.activation = activation
        self.conv_config=[self.in_channels,self.out_channels,self.kernel_size,self.stride,self.dilate,self.group]
        self.pad_value=0
        self.dwconv=self.group==self.in_channels and self.in_channels==self.out_channels
        self.ins_inplace=True

    @property
    def inplace_offset(self):
        if self.dwconv:
            inplace_offset=0
        elif self.group==1:
            input_h,input_w,input_ch=self.input_tensors[0].shape[1:]
            output_h,output_w,output_ch=self.output_tensors[0].shape[1:]
            inplace_offset=0
            for h in range(output_h):
                for w in range(output_w):
                    in_h=self.stride[0]*h
                    in_w=self.stride[1]*w
                    h_begin=max(in_h-self.padding[0][0],-1)
                    w_begin=max(in_w-self.padding[1][0],-1)
                    if h_begin<0:
                        spare_size=0
                    elif w_begin<0:
                        spare_size=(h_begin*input_w)*input_ch
                    else:
                        spare_size=(h_begin*input_w+w_begin+1)*input_ch
                    output_size=(h*output_w+w+1)*output_ch
                    inplace_offset=min(inplace_offset,spare_size-output_size)
            inplace_offset=inplace_offset
        else:
            raise NotImplementedError
        return inplace_offset


    def op_post_process(self):

        input_scale=self.input_tensors[0].scale
        weight_scale=self.input_tensors[1].scale
        output_scale=self.output_tensors[0].scale

        scale=np.round(weight_scale*(input_scale/output_scale)*2**32).astype("int64")
        assert np.sum(scale>=2**31)==0
        scale_tensor=MteTensor(
            tensor_idx=None,
            dtype="int32",
            shape=weight_scale.shape,
            tensor_type="extra_weight",
            data=scale,
        )
        h,w,c=self.input_tensors[0].shape[1:]
        if self.dwconv:
            buffer_tensor=MteTensor(
                tensor_idx=None,
                dtype="int8",
                shape=(h+self.padding[0][0]+self.padding[0][1],w+self.padding[1][0]+self.padding[1][1]),
                tensor_type="buffer",
            )
            self.input_tensors[2].data_-=np.sum(self.input_tensors[0].offset*self.input_tensors[1].data_,axis=(0,1,2))
            self.input_tensors[1].data_=np.transpose(self.input_tensors[1].data_,(0,3,1,2))

        else:
            if self.get_matched_config()==implemented_conv_configs[0]:
                self.input_tensors[1].dtype="int16"
            buffer_tensor=MteTensor(
                tensor_idx=None,
                dtype="int16",
                shape=(2,c),
                tensor_type="buffer",
            )
        return [scale_tensor,buffer_tensor]

    def get_matched_config(self):
        if self.dwconv:
            target_configs=implemented_dwconv_configs
        else:
            target_configs=implemented_conv_configs
        is_in_target_configs=False
        matched_target_config=None
        for conv_config in target_configs:
            is_target_config=True
            for source_config,target_config in zip(self.conv_config,conv_config):
                if target_config!=-1:
                    if isiterable(target_config):
                        if source_config not in target_config:
                            is_target_config=False
                            break
                    else:
                        if source_config!=target_config:
                            is_target_config=False
                            break
            if is_target_config:
                is_in_target_configs=True
                matched_target_config=conv_config
                break
        assert is_in_target_configs
        return matched_target_config

    def get_call_func(self):
        matched_target_config=self.get_matched_config()
        func=""
        if self.dwconv:
            func+="dw_"
        func+="conv2d_"
        input_tensor=self.input_tensors[0]
        weight_tensor=self.input_tensors[1]
        bias_tensor=self.input_tensors[2]
        scale_tensor=self.input_tensors[3]
        buffer_tensor=self.input_tensors[4]
        output_tensor=self.output_tensors[0]

        if self.activation is None:
            min_act=-128
        else:
            min_act=int(output_tensor.offset)
        if self.activation=="relu6":
            max_act=int(min(127,round(6/float(output_tensor.scale)+min_act)))
        else:
            max_act=127

        if self.dwconv:
            func+=(f"{self.kernel_size[0]}x{self.kernel_size[1]}_stride_{self.stride[0]}_{self.stride[1]}_dilate_{self.dilate[0]}_{self.dilate[1]}_s8("
                   f"{input_tensor.mem_symbol},"
                   f"{input_tensor.shape[1]},{input_tensor.shape[2]},{input_tensor.shape[3]},{int(input_tensor.offset)},"
                   f"{self.padding[0][0]},{self.padding[0][1]},{self.padding[1][0]},{self.padding[1][1]},"
                   f"{weight_tensor.var_symbol},{weight_tensor.mem_symbol},"
                   f"{bias_tensor.var_symbol},{bias_tensor.mem_symbol},"
                   f"{scale_tensor.var_symbol},{scale_tensor.mem_symbol},"
                   f"{buffer_tensor.mem_symbol},"
                   f"{min_act},{max_act},"
                   f"{output_tensor.mem_symbol},"
                   f"{output_tensor.shape[1]},{output_tensor.shape[2]},{output_tensor.shape[3]},{int(output_tensor.offset)}"
                   f")")
        else:
            if matched_target_config==implemented_conv_configs[0]:

                func+=(f"input_3_{self.kernel_size[0]}x{self.kernel_size[1]}_stride_{self.stride[0]}_{self.stride[1]}_dilate_{self.dilate[0]}_{self.dilate[1]}_s8("
                       f"{input_tensor.mem_symbol},"
                       f"{input_tensor.shape[1]},{input_tensor.shape[2]},{input_tensor.shape[3]},{int(input_tensor.offset)},"
                       f"{weight_tensor.var_symbol},{weight_tensor.mem_symbol},"
                       f"{bias_tensor.var_symbol},{bias_tensor.mem_symbol},"
                       f"{scale_tensor.var_symbol},{scale_tensor.mem_symbol},"
                       f"{buffer_tensor.mem_symbol},"
                       f"{min_act},{max_act},"
                       f"{output_tensor.mem_symbol},"
                       f"{output_tensor.shape[1]},{output_tensor.shape[2]},{output_tensor.shape[3]},{int(output_tensor.offset)}"
                       f")")
            else:
                func+=(f"{self.kernel_size[0]}x{self.kernel_size[1]}_s8("
                       f"{input_tensor.mem_symbol},"
                       f"{input_tensor.shape[1]},{input_tensor.shape[2]},{input_tensor.shape[3]},{int(input_tensor.offset)},"
                       f"{weight_tensor.var_symbol},{weight_tensor.mem_symbol},"
                       f"{bias_tensor.var_symbol},{bias_tensor.mem_symbol},"
                       f"{scale_tensor.var_symbol},{scale_tensor.mem_symbol},"
                       f"{buffer_tensor.mem_symbol},"
                       f"{self.stride[0]},{self.stride[1]},"
                       f"{min_act},{max_act},"
                       f"{output_tensor.mem_symbol},"
                       f"{output_tensor.shape[1]},{output_tensor.shape[2]},{output_tensor.shape[3]},{int(output_tensor.offset)}"
                       f")")
        return func







