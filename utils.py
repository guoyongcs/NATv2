import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from genotypes import LOOSE_END_PRIMITIVES, FULLY_CONCAT_PRIMITIVES, Genotype
from graphviz import Digraph
from collections import defaultdict
import scipy.sparse as sp
import torch.nn as nn
import difflib


def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def compute_flops(module: nn.Module, size, skip_pattern):
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size))
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

    
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    if isinstance(model, nn.DataParallel):
        return np.sum(np.prod(v.size()) for v in model.module.model_parameters()) / 1e6
    else:
        return np.sum(np.prod(v.size()) for v in model.model_parameters()) / 1e6


def count_parameters_woaux_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def draw_genotype(genotype, n_nodes, filename, concat=None):
    """

    :param genotype: 
    :param filename: 
    :return: 
    """
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("-2", fillcolor='darkseagreen2')
    g.node("-1", fillcolor='darkseagreen2')
    steps = n_nodes

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for op, source, target in genotype:
        if source == 0:
            u = "-2"
        elif source == 1:
            u = "-1"
        else:
            u = str(source - 2)
        v = str(target-2)
        op = 'null' if op == 'none' else op
        op = op.replace('dil_conv', 'dil_sep_conv') if 'dil_conv' in op else op
        g.edge(u, v, label=op, fillcolor="gray")


    g.node("out", fillcolor='palegoldenrod')
    if concat is not None:
        for i in concat:
            if i-2>=0:
                g.edge(str(i-2), "out", fillcolor="gray")
    else:
        for i in range(steps):
            g.edge(str(i), "out", fillcolor="gray")

    g.render(filename, view=False)


def arch_to_genotype(arch_normal, arch_reduce, n_nodes, cell_type, normal_concat=None, reduce_concat=None):
    try:
        primitives = eval(cell_type)
    except:
        assert False, 'not supported op type %s' % (cell_type)

    gene_normal = [(primitives[op], f, t) for op, f, t in arch_normal]
    gene_reduce = [(primitives[op], f, t) for op, f, t in arch_reduce]
    if normal_concat is not None:
        _normal_concat = normal_concat
    else:
        _normal_concat = range(2, 2 + n_nodes)
    if reduce_concat is not None:
        _reduce_concat = reduce_concat
    else:
        _reduce_concat = range(2, 2 + n_nodes)
    genotype = Genotype(normal=gene_normal, normal_concat=_normal_concat,
                        reduce=gene_reduce, reduce_concat=_reduce_concat)
    return genotype


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def arch_to_matrix(arch):
    f_list = []
    t_list = []
    for _, f, t in arch:
        f_list.append(f)
        t_list.append(t)
    return np.array(f_list), np.array(t_list)


def parse_arch(arch, num_node):
    f_list, t_list = arch_to_matrix(arch)
    adj = sp.coo_matrix((np.ones(f_list.shape[0]), (t_list, f_list)),
                        shape=(num_node, num_node),
                        dtype=np.float32)
    adj = adj.multiply(adj>0)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def sum_normalize(input):
    return input/torch.sum(input, -1, keepdim=True)


def convert_lstm_output(n_nodes, prev_nodes, prev_ops):
    """

    :param n_nodes: number of nodes
    :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
    :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
    :return: arch list, (op, f, t) is the elements
    """
    assert len(prev_nodes) == 2 * n_nodes
    assert len(prev_ops) == 2 * n_nodes
    arch_list = []
    for i in range(n_nodes):
        t_node = i + 2
        f1_node = prev_nodes[i * 2].item()
        f2_node = prev_nodes[i * 2 + 1].item()
        f1_op = prev_ops[i * 2].item()
        f2_op = prev_ops[i * 2 + 1].item()
        arch_list.append((f1_op, f1_node, t_node))
        arch_list.append((f2_op, f2_node, t_node))
    return arch_list


def genotype_to_arch(genotype, op_type='NOT_LOOSE_END_PRIMITIVES'):
    try:
        COMPACT_PRIMITIVES = eval(op_type)
    except:
        assert False, 'not supported op type %s' % (op_type)
    arch_normal = [(COMPACT_PRIMITIVES.index(op), f, t) for op, f, t in genotype.normal]
    arch_reduce = [(COMPACT_PRIMITIVES.index(op), f, t) for op, f, t in genotype.reduce]
    return arch_normal, arch_reduce


def str_diff_num(a,b):
    counter = 0
    for i,s in enumerate(difflib.ndiff(a, b)):
        if s[0]==' ': continue
        elif s[0]=='-' or s[0]=='+':
            counter += 1
    return int(counter/2)
