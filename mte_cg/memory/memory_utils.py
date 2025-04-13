import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..base import MteGraph, MteOp, MteTensor

class InplaceMemBlock:
    align_size=4
    ins_nums=0
    def __init__(self, tensor:MteTensor, begin):
        self.aligned_addr=-self.align(tensor.mem_size)
        self.aligned_size=self.align(tensor.mem_size)
        self.aligned_min_addr=-self.align(tensor.mem_size)
        self.tensors=[tensor]
        self.rel_aligned_addr=[self.aligned_addr]
        self.begin=begin
        self.end=begin+1
        self.aligned_mem_addr=None
        self.mem_type=tensor.tensor_type
        self.idx=self.ins_nums
        InplaceMemBlock.ins_nums+=1
        self.tensor_begin=[begin]
        self.tensor_end=[begin+1]
        self.patial_released=False

    def set_aligned_mem_addr(self, aligned_addr):
        self.aligned_mem_addr=aligned_addr
        for tensor,tensor_aligned_addr in zip(self.tensors,self.rel_aligned_addr):
            tensor.mem_addr=(self.aligned_mem_addr+self.aligned_mem_size+tensor_aligned_addr)*InplaceMemBlock.align_size

    @property
    def last_tensor_idx(self):
        return self.tensors[-1].tensor_idx

    def align(self,size):
        if size>=0:
            return (size+self.align_size-1)//self.align_size
        else:
            return -(-size+self.align_size-1)//self.align_size

    @property
    def mem_size(self):
        return self.aligned_mem_size*self.align_size

    @property
    def aligned_mem_size(self):
        return -self.aligned_min_addr

    @property
    def mem_addr(self):
        return self.aligned_mem_addr*self.align_size

    @property
    def last_tensor_begin(self):
        return self.tensor_begin[-1]

    def add_mem(self,tensor,run_id,inplace=None):
        aligned_size=self.align(tensor.mem_size)
        inplace=self.align(inplace)
        if self.aligned_addr+inplace>=self.aligned_min_addr:
            self.aligned_addr=self.aligned_addr+inplace
        else:
            if aligned_size<=-(self.aligned_addr + self.aligned_size):
                self.aligned_addr=self.aligned_addr + self.aligned_size
            else:
                if inplace is None:
                    self.aligned_addr= self.aligned_addr - aligned_size
                else:
                    self.aligned_addr=min(self.aligned_addr + inplace, -aligned_size)
        self.aligned_size=aligned_size
        if self.aligned_addr<self.aligned_min_addr:
            self.aligned_min_addr=self.aligned_addr
        self.tensors.append(tensor)
        self.rel_aligned_addr.append(self.aligned_addr)
        self.tensor_begin.append(run_id)
        self.tensor_end.append(run_id+1)
        self.end=run_id+1

    def set_end(self,end):
        self.tensor_end[-1]=end
        self.end=end

def check_slots(mem_slots):
    for i in range(len(mem_slots)-1):
        if mem_slots[i][1]<=mem_slots[i+1][0]:
            return False
    return True

class MemSlots:
    def __init__(self,ram_size,method="first_fit"):
        self.mem_slots=[[0,ram_size//InplaceMemBlock.align_size]]
        self.release_time=[0]
        self.release_addr=[0]
        self.released_idx=set()
        self.aligned_peak_mem=0
        self.method=method

    def __len__(self):
        return len(self.mem_slots)

    def __getitem__(self, idx):
        return self.mem_slots[idx]

    @property
    def peak_mem(self):
        return self.aligned_peak_mem*InplaceMemBlock.align_size

    def fit(self,idx,mem_space):
        if self.method=="first_fit":
            idx=self.first_fit(mem_space)
        elif self.method=="best_fit":
            idx=self.best_fit(mem_space)
        elif self.method=="recent_fit":
            idx=self.recent_fit(mem_space)
        else:
            raise NotImplementedError
        assert idx!=-1
        if self.method=="recent_fit":
            if self.mem_slots[idx][1]-self.mem_slots[idx][0]>mem_space.aligned_mem_size:
                if True:
                    if self.mem_slots[idx][1]-self.release_addr[idx]>mem_space.aligned_mem_size:
                        aligned_mem_addr=self.release_addr[idx]
                        if self.release_addr[idx]>self.mem_slots[idx][0]:
                            self.mem_slots.insert(idx+1,[self.release_addr[idx]+mem_space.aligned_mem_size,self.mem_slots[idx][1]])
                            self.release_time.insert(idx+1,self.release_time[idx])
                            self.release_addr.insert(idx+1,self.release_addr[idx]+mem_space.aligned_mem_size)
                            self.mem_slots[idx][1]=self.release_addr[idx]
                        else:
                            self.mem_slots[idx][0]=self.release_addr[idx]+mem_space.aligned_mem_size
                        self.release_addr[idx]=self.mem_slots[idx][0]
                    elif self.mem_slots[idx][1]-self.release_addr[idx]==mem_space.aligned_mem_size:
                        aligned_mem_addr= self.release_addr[idx]
                        self.mem_slots[idx][1]=self.release_addr[idx]
                        self.release_addr[idx]=self.mem_slots[idx][0]
                    else:
                        aligned_mem_addr= self.mem_slots[idx][1]-mem_space.aligned_mem_size
                        self.mem_slots[idx][1]-=mem_space.aligned_mem_size
                        self.release_addr[idx]=self.mem_slots[idx][0]
                else:
                    if self.aligned_peak_mem-self.release_addr[idx]>=mem_space.aligned_mem_size:
                        aligned_mem_addr= self.release_addr[idx]
                        if self.release_addr[idx]>self.mem_slots[idx][0]:
                            self.mem_slots.insert(idx+1,[self.release_addr[idx]+mem_space.aligned_mem_size,self.mem_slots[idx][1]])
                            self.release_time.insert(idx+1,self.release_time[idx])
                            self.release_addr.insert(idx+1,self.release_addr[idx]+mem_space.aligned_mem_size)
                            self.mem_slots[idx][1]=self.release_addr[idx]
                            self.release_addr[idx]=self.mem_slots[idx][0]
                        else:
                            self.mem_slots[idx][0]=self.release_addr[idx]+mem_space.aligned_mem_size
                            self.release_addr[idx]=self.mem_slots[idx][0]
                    elif self.aligned_peak_mem-self.mem_slots[idx][0]>mem_space.aligned_mem_size:
                        aligned_mem_addr= self.aligned_peak_mem - mem_space.aligned_mem_size
                        self.mem_slots.insert(idx + 1, [self.aligned_peak_mem, self.mem_slots[idx][1]])
                        self.release_time.insert(idx+1,self.release_time[idx])
                        self.release_addr.insert(idx + 1, self.aligned_peak_mem)
                        self.mem_slots[idx][1]= self.aligned_peak_mem - mem_space.aligned_mem_size
                        self.release_addr[idx]=self.mem_slots[idx][0]
                    else:
                        aligned_mem_addr= self.mem_slots[idx][0]
                        self.mem_slots[idx][0]+=mem_space.aligned_mem_size
                        self.release_addr[idx]=self.mem_slots[idx][0]
            else:
                aligned_mem_addr= self.mem_slots[idx][0]
                self.mem_slots.pop(idx)
                self.release_time.pop(idx)
                self.release_addr.pop(idx)
        else:
            aligned_mem_addr= self.mem_slots[idx][0]
            if self.mem_slots[idx][1]-self.mem_slots[idx][0]>mem_space.aligned_mem_size:
                self.mem_slots[idx][0]+=mem_space.aligned_mem_size
            else:
                self.mem_slots.pop(idx)
                self.release_time.pop(idx)
                self.release_addr.pop(idx)
        mem_space.set_aligned_mem_addr(aligned_mem_addr)
        self.aligned_peak_mem=max(self.aligned_peak_mem, self.mem_slots[-1][0])

    def first_fit(self,mem_space):
        idx=-1
        for i in range(len(self.mem_slots)):
            if self.mem_slots[i][1]-self.mem_slots[i][0]>=mem_space.aligned_mem_size:
                idx=i
                break
        return idx

    def best_fit(self,mem_space):
        min_d_idx=-1
        min_d=100000000
        for i in range(0,len(self.mem_slots)):
            if self.mem_slots[i][1]-self.mem_slots[i][0]>=mem_space.aligned_mem_size:
                d=abs(self.mem_slots[i][1]-self.mem_slots[i][0]-mem_space.aligned_mem_size)
                if d<min_d:
                    min_d_idx=i
                    min_d=d
        return min_d_idx

    def recent_fit(self,mem_space):

        idx=-1
        time=-1
        for i in range(0,len(self.mem_slots)):
            if self.mem_slots[i][1]-self.mem_slots[i][0]>=mem_space.aligned_mem_size:
                if self.release_time[i]>time:
                    idx=i
                    time=time+self.release_time[i]
        if idx==len(self.mem_slots)-1:
            if self.aligned_peak_mem-self.mem_slots[idx][0]<mem_space.aligned_mem_size:
                time=-1
                for i in range(0,len(self.mem_slots)-1):
                    if self.mem_slots[i][1]-self.mem_slots[i][0]>=mem_space.aligned_mem_size:
                        if self.release_time[i]>time:
                            idx=i
                            time=time+self.release_time[i]
        return idx

    def release_mem(self,mem_addr,mem_size,release_time):
        pos=0
        for j in range(len(self.mem_slots)):
            if self.mem_slots[j][0]<=mem_addr:
                pos=j+1
        if pos==0:
            if self.mem_slots[0][0]==mem_addr+mem_size:
                self.mem_slots[0][0]= mem_addr
                self.release_time[0]=release_time
                self.release_addr[0]=mem_addr
            else:
                self.mem_slots.insert(0, [mem_addr, mem_addr +mem_size])
                self.release_time.insert(0, release_time)
                self.release_addr.insert(0, mem_addr)
        else:
            if self.mem_slots[pos-1][1]==mem_addr:
                if pos<len(self.mem_slots) and self.mem_slots[pos][0]==mem_addr+mem_size:
                    self.mem_slots[pos-1][1]=self.mem_slots[pos][1]
                    self.release_time[pos-1]=release_time
                    self.release_addr[pos-1]=mem_addr
                    self.mem_slots.pop(pos)
                    self.release_time.pop(pos)
                    self.release_addr.pop(pos)
                else:
                    self.mem_slots[pos-1][1]= mem_addr + mem_size
                    self.release_time[pos-1]=release_time
                    self.release_addr[pos-1]=mem_addr
            else:
                if pos<len(self.mem_slots) and self.mem_slots[pos][0]==mem_addr+mem_size:
                    self.mem_slots[pos][0]= mem_addr
                    self.release_time[pos]=release_time
                    self.release_addr[pos]=mem_addr
                else:
                    self.mem_slots.insert(pos, [mem_addr , mem_addr + mem_size])
                    self.release_time.insert(pos, release_time)
                    self.release_addr.insert(pos, mem_addr)
    def recycle(self,mem_spaces,recycle_time):
        for mem_space in mem_spaces:
            if mem_space.end<=recycle_time and mem_space.idx not in self.released_idx:
                self.release_mem(mem_space.aligned_mem_addr+mem_space.aligned_mem_size+mem_space.rel_aligned_addr[-1],mem_space.align(mem_space.tensors[-1].mem_size),recycle_time)
                # assert check_slots(mem_slots)
                self.released_idx.add(mem_space.idx)
            if mem_space.last_tensor_begin+1==recycle_time and mem_space.patial_released is False and len(mem_space.tensors)>1:
                if mem_space.aligned_mem_size+mem_space.rel_aligned_addr[-1]!=0:
                    self.release_mem(mem_space.aligned_mem_addr,mem_space.aligned_mem_size+mem_space.rel_aligned_addr[-1],recycle_time)
                if mem_space.rel_aligned_addr[-1]+mem_space.align(mem_space.tensors[-1].mem_size)!=0:
                    self.release_mem(mem_space.aligned_mem_addr+mem_space.aligned_mem_size+mem_space.rel_aligned_addr[-1]+mem_space.align(mem_space.tensors[-1].mem_size),
                                     -(mem_space.rel_aligned_addr[-1]+mem_space.align(mem_space.tensors[-1].mem_size)),recycle_time)
                mem_space.patial_released=True


def create_mem_spaces(mte_graph:MteGraph):
    mem_spaces:list[InplaceMemBlock]=[]
    for run_id,op_idx in enumerate(mte_graph.run_seq):
        mte_op:MteOp=mte_graph.get_op(op_idx)
        input_tensors=mte_op.input_tensors
        for input_tensor in input_tensors:
            input_idx=input_tensor.tensor_idx
            is_in_mem_spaces=False
            for mem_space in mem_spaces:
                if mem_space.last_tensor_idx==input_idx:
                    is_in_mem_spaces=True
                    mem_space.set_end(run_id+1)
            if not is_in_mem_spaces:
                if input_tensor.used_in_ram:
                    mem_spaces.append(
                        InplaceMemBlock(input_tensor,run_id)
                    )
        # output_idxes=mte_graph.get_output_idxes(op_idx)
        output_tensors=mte_op.output_tensors
        assert len(output_tensors)==1
        output_tensor=output_tensors[0]
        if mte_op.inplace:
            first_input_idx=input_tensors[0].tensor_idx
            is_in_mem_spaces=False
            for mem_space in mem_spaces:
                if mem_space.last_tensor_idx==first_input_idx:
                    mem_space.add_mem(output_tensor,run_id,mte_op.inplace_offset)
                    is_in_mem_spaces=True
            assert is_in_mem_spaces
        else:
            mem_spaces.append(InplaceMemBlock(output_tensor, run_id))
    return mem_spaces

def allocate_memory(mem_spaces:list[InplaceMemBlock],fit_method):
    max_sram_size=20*1024*1024
    mem_slots=MemSlots(max_sram_size,method=fit_method)
    for mem_idx,mem_space in enumerate(mem_spaces):
        mem_slots.recycle(mem_spaces[:mem_idx],mem_space.begin)
        mem_slots.fit(mem_slots,mem_space)
        assert mem_space.aligned_mem_addr is not None
        # for tensor_idx in mem_space.tensor_idxes:
        #     allocated_tensor_mems[tensor_idx]= mem_space.mem_addr-mem_space.min_addr + mem_space.tensor_idxes[tensor_idx]
    return mem_spaces,mem_slots.peak_mem

color_map={
    "weight":"yellow",
    "extra_weight":"goldenrod",
    "buffer":"green",
    "activation":"blue",
}

def show_memory_distribution(mem_spaces:list[InplaceMemBlock],peak_mem,vis_name=None,vis_path=None):
    fig, ax = plt.subplots(figsize=(16, 12), dpi=500)
    rects=[]
    for mem_space in mem_spaces:
        # assert mem_space.mem_type!="buffer"
        color=color_map[mem_space.mem_type]
        for begin,end,tensor in zip(mem_space.tensor_begin,mem_space.tensor_end,mem_space.tensors):
            rects.append(
                patches.Rectangle((begin, tensor.mem_addr), end - begin-0.5, tensor.mem_size, edgecolor='red', facecolor=color, linewidth=0.1,alpha=0.5)
            )
    for rect in rects:
        ax.add_patch(rect)
    end=-1
    for mem_space in mem_spaces:
        end=max(end,mem_space.end)
    ax.set_ylim([0,peak_mem])
    ax.set_xlim([0,end])
    # plt.show()
    if vis_path is not None:
        if vis_name is not None:
            plt.savefig(os.path.join(vis_path,f'{vis_name}.png'))
        else:
            plt.savefig(os.path.join(vis_path,'memory.png'))


def static_allocate_memory(mem_spaces):
    from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, LpInteger, LpStatus, value,HiGHS_CMD

    # 定义内存块，每个内存块包含 id, size, start, end
    blocks = []
    for id,mem_space in enumerate(mem_spaces):
        blocks.append({"id": id, "size": mem_space.aligned_mem_size, "start": mem_space.begin, "end":mem_space.end})

    n = len(blocks)
    # Big-M 常量选取为所有块大小之和（足够大）
    B = sum(block["size"] for block in blocks)

    # 创建一个最小化问题
    prob = LpProblem("StaticMemoryAllocation", LpMinimize)

    # 为每个内存块定义起始地址变量 x_i (设为整数)
    x_vars = {
        block["id"]: LpVariable(f"x_{block['id']}", lowBound=0, cat=LpInteger)
        for block in blocks
    }

    # 定义 M 变量，表示总内存使用上界（即所有块分配后最高地址）
    M_var = LpVariable("M", lowBound=0, cat=LpInteger)

    # 目标：最小化 M
    prob += M_var, "Minimize overall memory usage"

    # 对于每个内存块，保证 x_i + size_i <= M
    for block in blocks:
        prob += x_vars[block["id"]] + block["size"] <= M_var, f"block_{block['id']}_fits_in_M"

    # 对于重叠的内存块添加“不重叠”的约束
    # 如果两个内存块 i 和 j 的生命周期重叠，则需要保证它们分配的内存区域不冲突
    # 利用二进制变量 d_{ij} 决定先后顺序，仅对 i < j 添加约束
    d_vars = {}  # 存储各对重叠块的二进制变量
    for i in range(n):
        for j in range(i + 1, n):
            block_i = blocks[i]
            block_j = blocks[j]
            # 判断生命周期是否重叠：即 block_i.start < block_j.end 且 block_j.start < block_i.end
            if block_i["start"] < block_j["end"] and block_j["start"] < block_i["end"]:
                # 定义二进制变量 d_{ij}：若为 1，表示块 i 在块 j 前面；否则块 j 在 i 前面
                d_var = LpVariable(f"d_{block_i['id']}_{block_j['id']}", cat=LpBinary)
                d_vars[(block_i["id"], block_j["id"])] = d_var

                # 若 d == 1, 则保证 block_i 在 block_j 之前： x_i + size_i <= x_j + B*(1-d)
                prob += (
                    x_vars[block_i["id"]] + block_i["size"] <= x_vars[block_j["id"]] + B * (1 - d_var),
                    f"order_{block_i['id']}_before_{block_j['id']}"
                )
                # 若 d == 0, 则保证 block_j 在 block_i 之前： x_j + size_j <= x_i + B*d
                prob += (
                    x_vars[block_j["id"]] + block_j["size"] <= x_vars[block_i["id"]] + B * d_var,
                    f"order_{block_j['id']}_before_{block_i['id']}"
                )

    # 求解问题（建议使用默认的 HiGHS 求解器）
    prob.solve(solver=HiGHS_CMD(msg=True))

    print("求解状态:", LpStatus[prob.status])
    if LpStatus[prob.status] == "Optimal":
        print("最小总内存使用量 M =", value(M_var))
        for block in blocks:
            alloc_addr = value(x_vars[block["id"]])
            mem_spaces[block["id"]].aligned_mem_size = alloc_addr
        return mem_spaces,None,value(M_var)
    else:
        return None,None,None


def allocate_tensor_memory(mte_graph:MteGraph,vis_name=None,vis_path=None):

    mem_spaces=create_mem_spaces(mte_graph)
    mem_spaces.sort(key=lambda x:x.begin*100000-x.aligned_mem_size)

    mem_spaces,peak_mem=allocate_memory(mem_spaces,"first_fit")
    # mem_spaces,allocated_tensor_mems,peak_mem=static_allocate_memory(mem_spaces)
    # mem_spaces[-1].aligned_mem_addr-=mem_spaces[-2].aligned_mem_addr
    # mem_spaces[-2].aligned_mem_addr=00
    show_memory_distribution(mem_spaces,peak_mem,vis_name,vis_path)
    print(f" Peak Memory:{peak_mem/1024:.2f}KB")
    return peak_mem


