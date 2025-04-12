from ..mte_base import OPERATOR, BasicOp, ModelReader


@OPERATOR.register_operator("PAD")
def pad_parse_func(op_idx, model_reader:ModelReader):
    op = Pad(op_idx)
    return op

class Pad(BasicOp):
    _inplace=False
    def __init__(self, op_idx):
        super().__init__(op_idx)
        self.op_idx = op_idx