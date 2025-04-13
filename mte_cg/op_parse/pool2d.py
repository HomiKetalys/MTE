from ..base import OPERATOR, MteOp, ModelReader, TFLiteReader


# def get_max_pool2d_op_options(op_idx, model_reader:ModelReader):
#     mte_options={
#         "kernel_size":None,
#         "stride":None,
#         "padding":None,
#     }
#     if isinstance(model_reader,TFLiteReader):
#         options=model_reader.operators[op_idx]["builtin_options"]
#         mte_options["kernel_size"]=(options["filter_height"],options["filter_width"])
#         mte_options["stride"]=(options["stride_h"],options["stride_w"])
#         if options["padding"]==0:
#             mte_options["padding"]=((options["filter_height"]//2,(options["filter_height"]-1)//2),
#                                     (options["filter_width"]//2,(options["filter_width"]-1)//2))
#         elif options["padding"]==1:
#             mte_options["padding"]=((0,0),
#                                     (0,0))
#         else:
#             raise NotImplementedError
#     else:
#         raise NotImplementedError
#     return mte_options
#
# @OPERATOR.register_operator("MAX_POOL_2D")
# def max_pood2d_parse_func(op_idx, model_reader:ModelReader):
#     mte_options=get_max_pool2d_op_options(op_idx, model_reader)
#     op = MaxPool2d(op_idx,mte_options["kernel_size"],mte_options["stride"],mte_options["padding"])
#     return op
#
# class MaxPool2d(MteOp):
#     _inplace=False
#     def __init__(self, op_idx=None,kernel_size=(1,1),stride=(1,1),padding=((0,0),(0,0))):
#         super().__init__(op_idx)
#         self.kernel_size=kernel_size
#         self.stride=stride
#         self.padding=padding
#
#     def get_call_func(self):
#         shape=self.input_tensors[0].shape
#         return (f"max_pool2d("
#                 f"{self.input_tensors[0].mem_symbol},{shape[1]},{shape[2]},{shape[3]},{int(self.input_tensors[0].offset)},"
#                 f"{self.padding[0][0]},{self.padding[0][1]},{self.padding[1][0]},{self.padding[1][1]},"
#                 f"{self.kernel_size[0]},{self.kernel_size[1]},{self.stride[0]},{self.stride[1]},"
#                 f"{-128},{127},"
#                 f"{self.output_tensors[0].mem_symbol},{self.output_tensors[0].shape[1]},{self.output_tensors[0].shape[2]}"
#                 f")")

@OPERATOR.register_operator("AVERAGE_POOL_2D")
def avg_pood2d_parse_func(op_idx, model_reader:ModelReader):
    op = AvgPool2d(op_idx)
    return op

class AvgPool2d(MteOp):
    _inplace=False
    def __init__(self, op_idx):
        super().__init__(op_idx)

    def get_call_func(self):
        return "avg_pool2d()"