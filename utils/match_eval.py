import os
import random

import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit import RDLogger
from rdkit.Chem import ChemicalFeatures

from utils.smiles2ppgraph import cal_dist

RDLogger.DisableLog('rdApp.*')

from itertools import product, permutations
from collections import Counter


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = random.randint(0, n - 1)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])
    cresult = Counter(Psample)
    psam = [cresult[x] for x in plist]
    pe = [x * N for x in plist]
    return Psample


def cal_dist_all(mol, phco_list_i, phco_list_j):
    for phco_elment_i in phco_list_i:
        for phco_elment_j in phco_list_j:
            if phco_elment_i == phco_elment_j:
                if len(phco_list_i) == 1 and len(phco_list_j) == 1:
                    dist = 0
                else:
                    dist = max(len(phco_list_i), len(phco_list_j)) * 0.2
        if not set(phco_list_i).intersection(set(phco_list_j)):
            dist_set = []
            for atom_i in phco_list_i:
                for atom_j in phco_list_j:
                    dist_ = cal_dist(mol, atom_i, atom_j)
                    dist_set.append(dist_)
            min_dist = min(dist_set)
            if max(len(phco_list_i), len(phco_list_j)) == 1:
                dist = min_dist
            else:
                dist = min_dist + max(len(phco_list_i), len(phco_list_j)) * 0.2
    return dist


def extract_dgl_info(g):
    node_type = g.ndata.get('type', g.ndata['h'][:, :-1])  # a temporary fix
    dist = g.edata.get('dist', g.edata['h'])

    ref_dist_list = []
    value = []
    for i in range(len(g.edges()[0])):
        ref_dist_name = '{}{}'.format(int(g.edges()[0][i]), int(g.edges()[1][i]))  ##取参考药效团的距离
        ref_dist_list.append(ref_dist_name)
        value.append(float(dist[i]))
    dist_dict = dict(zip(ref_dist_list, value))
    type_list = []
    for n in range(len(node_type)):
        list_0 = [0]
        nonzoro_list = node_type[n].numpy().tolist()
        list_0.extend(nonzoro_list)
        aa = np.nonzero(list_0)
        type_list.append(tuple(aa[0]))
    return dist_dict, type_list


__FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
__MAPPING = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}


def match_score(smiles, g):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1

    feats = __FACTORY.GetFeaturesForMol(mol)
    dist, ref_type = extract_dgl_info(g)

    all_phar_types = {i for j in ref_type for i in j}

    phar_filter = [[] for _ in range(len(ref_type))]

    phar_mapping = {i: [] for i in ref_type}
    for i in range(len(ref_type)):
        phar_mapping[ref_type[i]].append(i)

    mol_phco_candidate = []
    for f in feats:
        phar = f.GetFamily()
        phar_index = __MAPPING.setdefault(phar, 7)
        if phar_index not in all_phar_types:
            continue
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        phar_info = ((phar_index,), atom_index)
        mol_phco_candidate.append(phar_info)

    tmp_n = len(mol_phco_candidate)
    for i in range(tmp_n):
        phar_i, atom_i = mol_phco_candidate[i]
        for j in range(i + 1, tmp_n):
            phar_j, atom_j = mol_phco_candidate[j]
            if atom_i == atom_j and phar_i != phar_j:
                phars = tuple(sorted((phar_i[0], phar_j[0])))
                mol_phco_candidate.append([phars, atom_i])

    for phar, atoms in mol_phco_candidate:
        if phar in phar_mapping:
            for idx in phar_mapping[phar]:
                phar_filter[idx].append(atoms)

    match_score = max_match(smiles, g, phar_filter, phar_mapping.values())
    return match_score


def __iter_product(phco, phar_grouped):
    group_elements = [None for _ in range(len(phar_grouped))]
    n_places = []
    for i in range(len(phar_grouped)):
        group_elements[i] = list(range(len(phco[phar_grouped[i][0]])))
        l_elements = len(group_elements[i])
        l_places = len(phar_grouped[i])
        n_places.append(l_places)

        if l_elements < l_places:
            group_elements[i].extend([None] * (l_places - l_elements))

    for i in product(*[permutations(i, n) for i, n in zip(group_elements, n_places)]):
        res = [None] * len(phco)

        for g_ele, g_idx in zip(i, phar_grouped):
            for a, b in zip(g_ele, g_idx):
                res[b] = a

        yield res


def max_match(smiles, g, phco, phar_mapping):
    # will modify phar_filter

    mol = Chem.MolFromSmiles(smiles)
    ref_dist, ref_type = extract_dgl_info(g)

    length = len(phco)

    dist_dict = {}
    for i in range(length - 1):
        for j in range(i + 1, length):
            for elment_len1 in range(len(phco[i])):
                for elment_len2 in range(len(phco[j])):
                    if phco[i][elment_len1] is None or phco[j][elment_len2] is None:
                        dist = 100
                    else:
                        dist = cal_dist_all(mol, phco[i][elment_len1], phco[j][elment_len2])  ##

                    dist_name = (i, elment_len1, j, elment_len2)

                    dist_dict[dist_name] = dist

    match_score_max = 0
    for phco_elment_list in __iter_product(phco, list(phar_mapping)):

        error_count = 0
        correct_count = 0

        for p in range(len(phco_elment_list)):
            for q in range(p + 1, len(phco_elment_list)):

                key_ = (p, phco_elment_list[p], q, phco_elment_list[q])

                if phco_elment_list[p] is None or phco_elment_list[q] is None:
                    dist_ref_candidate = 100
                else:
                    dist_ref_candidate = abs(dist_dict[key_] - ref_dist['{}''{}'.format(p, q)])
                if dist_ref_candidate < 1.21:
                    correct_count += 1
                else:
                    error_count += 1
        match_score = correct_count / (correct_count + error_count)

        match_score_max = max(match_score, match_score_max)

        if match_score_max == 1:
            return match_score_max

    return match_score_max


from tqdm.auto import tqdm
from multiprocessing import Pool, TimeoutError
from multiprocessing.dummy import Pool as ThreadPool


__timeout = None
__pp_graph = None
__smiles_list = None


def foo(idx):
    g = __pp_graph[idx]
    smiles = __smiles_list[idx]

    s = match_score(smiles, g)

    return s


def foo_timeout(idx):
    with ThreadPool(1) as p:
        res = p.apply_async(foo, args=(idx,))
        score = -3
        try:
            score = res.get(__timeout)  # Wait timeout seconds for func to complete.
        except TimeoutError:
            score = -2
    return score


def get_match_score(phar_graphs, smiles_list, n_workers=8, timeout=20):
    """
    meaning of return value:
        0~1: normal match score;
        -1: invalid molecule;
        -2: timeout
    """

    assert len(phar_graphs) == len(smiles_list)
    global __timeout
    global __pp_graph
    global __smiles_list

    __timeout = timeout
    __pp_graph = phar_graphs
    __smiles_list = smiles_list

    N = len(smiles_list)
    with Pool(n_workers, maxtasksperchild=32) as pool:
        match_score = list(tqdm(pool.imap(foo_timeout, list(range(N))), total=N))

    return match_score
