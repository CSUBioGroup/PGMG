from pathlib import Path
from typing import List, Optional

import dgl
import numpy as np
import torch

from .smiles2ppgraph import MAX_NUM_PP_GRAPHS

idx2phar = {0: 'AROM',
            1: 'HYBL',
            2: 'POSC',
            3: 'HACC',
            4: 'HDON',
            5: 'LHYBL',
            6: 'UNKONWN',  # any other types
            }

idx2size = {0: ((5, 6), (0.5, 0.5)),  # (n1,n2),(p_n1,p_n2)
            1: ((3,), (1,)),
            2: ((1,), (1,)),
            3: ((1,), (1,)),
            4: ((1,), (1,)),
            5: ((6,), (1,)),
            6: ((1,), (1,)),
            }

phar2idx = {v: k for k, v in idx2phar.items()}


def pos2edis(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def edis2sdis(edis):
    # Euclidian distances to shortest-path distances
    return edis * 1.06068655 - 0.43105129  # linear mapping


def build_dgl_graph(u, v, dis, node_type: np.array, node_size: np.array):
    g = dgl.graph((u, v))
    g.edata['h'] = torch.FloatTensor(dis).reshape(-1, 1)
    g.ndata['h'] = torch.FloatTensor(np.concatenate((node_type, node_size.reshape(-1, 1)), axis=1))
    return g


def format_type(types: List[str]):
    tp = [0] * 7
    size = -1
    for t in types:
        # t_idx = phar2idx.get(t, 6)
        t_idx = phar2idx[t]
        tp[t_idx] = 1
        sizes, probs = idx2size[t_idx]
        c_size = np.random.choice(sizes, p=probs)
        size = max(c_size, size)
    return tp, size


def load_phar_file(file_path: Path):
    load_file_fn = {'.posp': load_pp_file, '.edgep': load_ep_file}.get(file_path.suffix, None)

    if load_file_fn is None:
        raise ValueError(f'Invalid file path: "{file_path}"!')

    return load_file_fn(file_path)


def load_pp_file(file_path: Path):
    node_type = []
    node_size = []
    node_pos = []  # [(x,y,z)]

    lines = file_path.read_text().strip().split('\n')

    n_nodes = len(lines)

    assert n_nodes <= MAX_NUM_PP_GRAPHS

    for line in lines:
        types, x, y, z = line.strip().split()

        tp, size = format_type(types.strip().split(','))

        node_type.append(tp)
        node_size.append(size)
        node_pos.append(tuple(float(i) for i in (x, y, z)))

    node_type = np.array(node_type)
    node_size = np.array(node_size)
    node_pos = np.array(node_pos)

    n_nodes = len(node_type)

    u = []
    v = []
    dis = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            u.append(i)
            v.append(j)

            dis.append(edis2sdis(pos2edis(node_pos[i], node_pos[j])))

    u, v = u + v, v + u
    dis = dis + dis

    g = build_dgl_graph(u, v, dis, node_type, node_size)

    return g


def load_ep_file(file_path: Path):
    node_type = []
    node_size = []

    lines = file_path.read_text().strip().split('\n')

    n_nodes = int(lines[0].strip())

    assert n_nodes <= MAX_NUM_PP_GRAPHS

    for i in range(1, 1 + n_nodes):
        idx, types = lines[i].strip().split()
        assert int(idx) == i

        tp, size = format_type(types.strip().split(','))

        node_type.append(tp)
        node_size.append(size)

    node_type = np.array(node_type)
    node_size = np.array(node_size)

    u = []
    v = []
    dis = []
    for i in range(1 + n_nodes, 1 + n_nodes + n_nodes * (n_nodes - 1) // 2):
        ui, vi, disi = lines[i].strip().split(' ')
        ui, vi = int(ui) - 1, int(vi) - 1
        disi = float(disi)

        u.append(ui)
        v.append(vi)
        dis.append(disi)

    u, v = u + v, v + u
    dis = dis + dis

    g = build_dgl_graph(u, v, dis, node_type, node_size)

    return g


def export_ep_text(g: dgl.DGLGraph, file_path: Optional[Path] = None):
    """
    transfer a given DGLGraph into text using .edgep format
    """

    g = g.cpu()

    assert g.batch_size == 1

    edgep_text = ''

    n = g.num_nodes()
    edgep_text += f'{n}\n'

    type_vec = g.ndata['h'][:, :-1] if 'h' in g.ndata.keys() else g.ndata['type']

    for i in range(n):
        types = type_vec[i].nonzero().numpy().flatten()
        t = ','.join([idx2phar[i] for i in types])
        edgep_text += f'{i + 1} {t}\n'

    dist_vec = g.edata['h'] if 'h' in g.edata.keys() else g.edata['dist']

    dist_mapping = {}
    for i, (u, v) in enumerate(torch.stack(g.edges()).T.numpy()):
        u, v = (u, v) if u < v else (v, u)
        dist_mapping[(u, v)] = dist_vec.numpy().flatten()[i]

    for (u, v), dist in dist_mapping.items():
        edgep_text += f'{u + 1} {v + 1} {dist}\n'

    if file_path is not None:
        assert file_path.suffix == '.edgep'
        file_path.write_text(edgep_text)

    return edgep_text
