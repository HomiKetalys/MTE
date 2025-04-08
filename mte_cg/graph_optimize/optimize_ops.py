import numpy as np

from ..op_parse import Pad, Conv2d, Reshape, Transpose, MteTensor, Identity
from ..op_parse.parse_tools import MteGraph

def optimize_reshape(mte_graph:MteGraph):
    for op_idx in mte_graph.run_seq:
        mte_op=mte_graph.get_op(op_idx)
        if isinstance(mte_op,Reshape):
            mte_graph.remove_op_input_tensor(mte_op,mte_op.input_tensors[1])
            mte_graph.replace_op(mte_op,Identity())

# def find_no_change_layout_op_link(op:MteTensor):
#     final_tensors=[]
#     input_shape=op.input_tensors[0].shape



def remove_redundant_ops(mte_graph:MteGraph):
    for run_idx in mte_graph.run_seq:
        mte_op=mte_graph.get_op(run_idx)
        if isinstance(mte_op,Transpose):
            mte_op:Transpose=mte_op
            output_tensor:MteTensor=mte_op.output_tensors[0]
            if len(output_tensor.to_ops)==1:
                next_op=output_tensor.to_ops[0]
                if isinstance(next_op,Reshape):
                    next_op:Reshape=next_op
                    output_tensor:MteTensor=next_op.output_tensors[0]
                    if len(output_tensor.to_ops)==1:
                        nnext_op=output_tensor.to_ops[0]
                        if isinstance(nnext_op,Transpose):
                            nnext_op:Transpose=nnext_op
                            transpose_data1=mte_op.input_tensors[1].data
                            reshape_data=next_op.input_tensors[1].data
                            transpose_data2=nnext_op.input_tensors[1].data
                            input_data0=np.array(list(range(mte_op.input_tensors[0].size)),dtype="int32").reshape(mte_op.input_tensors[0].shape)
                            input_data1=np.transpose(input_data0,transpose_data1)
                            input_data1=np.reshape(input_data1,reshape_data)
                            input_data1=np.transpose(input_data1,transpose_data2)
                            if np.sum(np.abs(input_data1.flatten()-input_data0.flatten()))==0:
                                mte_graph.remove_op_input_tensor(mte_op,mte_op.input_tensors[1])
                                mte_graph.replace_op(mte_op,Identity())
                                mte_graph.remove_op_input_tensor(next_op,next_op.input_tensors[1])
                                mte_graph.replace_op(next_op,Identity())
                                mte_graph.remove_op_input_tensor(nnext_op,nnext_op.input_tensors[1])
                                mte_graph.replace_op(nnext_op,Identity())
                                break
