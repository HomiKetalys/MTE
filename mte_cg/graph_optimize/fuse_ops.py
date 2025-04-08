from ..op_parse import Pad, Conv2d, Reshape
from ..op_parse.parse_tools import MteGraph


def fuse_zero_pad_with_conv(mte_graph:MteGraph):
    for run_idx in mte_graph.run_seq:
        mte_op=mte_graph.get_op(run_idx)
        if isinstance(mte_op,Pad):
            pad_op:Pad=mte_op
            output_tensor=pad_op.output_tensors[0]
            to_ops=output_tensor.to_ops
            to_op_idxes=[op.op_idx for op in to_ops]
            if len(to_op_idxes)>0:
                padding_data=pad_op.input_tensors[1].data
                input_tensor=pad_op.input_tensors[0]
                if padding_data[0][0]==0 and padding_data[0][1]==0 and padding_data[3][0]==0 and padding_data[3][1]==0:
                    for to_op_idx in to_op_idxes:
                        to_op=mte_graph.get_op(to_op_idx)
                        if isinstance(to_op,Conv2d):
                            conv_op:Conv2d=to_op
                            conv_op.padding=((padding_data[1][0]+conv_op.padding[0][0],padding_data[1][1]+conv_op.padding[0][1]),
                                             (padding_data[2][0]+conv_op.padding[1][0],padding_data[2][1]+conv_op.padding[1][1]))
                            mte_graph.replace_op_input_tensor(conv_op,output_tensor,input_tensor)
