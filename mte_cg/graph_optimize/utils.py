
from ..mte_base import graph_optimizers
def register_graph_optimizer(priority=0):
    def wrapper(func):
        graph_optimizers.append((priority, func))
    return wrapper