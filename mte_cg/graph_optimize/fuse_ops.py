import copy

from ..base import MteGraph
from ..op_parse import Transpose,Pad,MteOp,Identity

from .utils import register_graph_optimizer

@register_graph_optimizer(2)
def fuse_zero_pad(mte_graph:MteGraph):
    run_seq=copy.deepcopy(mte_graph.run_seq)
    for run_idx in run_seq:
        mte_op=mte_graph.get_op(run_idx)
        if mte_op is None:
            continue
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
                        if hasattr(to_op,"padding"):
                            to_op.padding=((padding_data[1][0]+to_op.padding[0][0],padding_data[1][1]+to_op.padding[0][1]),
                                             (padding_data[2][0]+to_op.padding[1][0],padding_data[2][1]+to_op.padding[1][1]))
                            mte_graph.replace_op_input_tensor(to_op,output_tensor,input_tensor)


def remove_continued_transpose(mte_op:MteOp, perm0, mte_graph:MteGraph):
    assert isinstance(mte_op,Transpose)
    to_ops=mte_op.output_tensors[0].to_ops
    if len(to_ops)>1:
        return False
    else:
        to_op=to_ops[0]
    if isinstance(to_op,Transpose):
        perm1=to_op.input_tensors[1].data
        new_perm=[None]*len(perm0)
        for i in range(len(perm0)):
            new_perm[i]=perm0[perm1[i]]
        for i in range(len(perm0)):
            if int(new_perm[i])!=i:
                if remove_continued_transpose(to_op,new_perm,mte_graph):
                    mte_graph.remove_op_input_tensor(mte_op,mte_op.input_tensors[1])
                    mte_graph.replace_op(mte_op,Identity())
                    return True
                else:
                    return False
        mte_graph.remove_op_input_tensor(to_op,to_op.input_tensors[1])
        mte_graph.replace_op(to_op,Identity())
        mte_graph.remove_op_input_tensor(mte_op,mte_op.input_tensors[1])
        mte_graph.replace_op(mte_op,Identity())
        return True
    return False

@register_graph_optimizer(1)
def fuse_transpose(mte_graph:MteGraph):
    run_seq=copy.deepcopy(mte_graph.run_seq)
    for run_idx in run_seq:
        mte_op=mte_graph.get_op(run_idx)
        if mte_op is None:
            continue
        if isinstance(mte_op,Transpose):
            remove_continued_transpose(mte_op,mte_op.input_tensors[1].data,mte_graph)
