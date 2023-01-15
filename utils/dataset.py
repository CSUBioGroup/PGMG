import re
from typing import List

import dgl
import numpy as np
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.smiles2ppgraph import smiles2ppgraph

MAX_NUM_PP_GRAPHS = 8  # same with smiles2ppgraph.py

# from logging import getLogger
# LOGGER = getLogger('main')

class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>')
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use

    PATTEN = re.compile(r'\[[^\]]+\]'
                        # only some B|C|N|O|P|S|F|Cl|Br|I atoms can omit square brackets
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a))

        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx=[]
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3  # 3 for <mask> !!!!!!
            else:
                a = self.s2i[s]
            l.append(a)
            
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTEN, s) is not None:
                atom_idx.append(i)
        if return_atom_idx:
            return l, atom_idx
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s[i]
                if c == '<eos>':
                    break
                s.append(c)
            smiles.append(''.join(s))

        return smiles


def run_test_tokenizer():
    smiles = ['CCNC(=O)NInc1%225cpppcc2nc@@nc(N@c3ccc(O[C@@H+5]c4cccc(F)c4)c(Cl)c3)c2c1']
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    print(tokenizer.parse(smiles[0]))
    print(tokenizer.get_text([tokenizer.parse(smiles[0])]))


def _corrupt(token_seq: List[int], mask_token, corrupt_percent=0.1, poisson_lambda=2):
    # infilling, not perfect
    token_seq = token_seq.copy()
    l = len(token_seq)
    n = int(l * corrupt_percent)

    c = 0
    idx = sorted(np.random.choice(list(range(1, l - 1)), n), reverse=True)  # skip <sos>
    for i in idx:
        li = np.random.poisson(poisson_lambda)
        while li < 1:
            li = np.random.poisson(poisson_lambda)
        token_seq[i] = mask_token
        li -= 1
        p = i + 1
        while p < l and li > 0:
            del token_seq[p]
            l -= 1
            li -= 1
            c += 1
        if c >= n:
            break

    return token_seq


def get_random_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # clear isotope
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)

    return rsmiles


class SemiSmilesDataset(Dataset):

    def __init__(self, smiles_list, tokenizer: Tokenizer,
                 use_random_input_smiles=False, use_random_target_smiles=False, rsmiles=None, corrupt=True):
        """
        :param smiles_list: list of valid smiles
        :param tokenizer:
        :param use_random_input_smiles:
        :param use_random_target_smiles:
        :param rsmiles:
        :param corrupt: boolean, whether to use infilling scheme to corrupt input smiles
        """
        super().__init__()
        
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.SPECIAL_TOKENS.index('<mask>')

        self.vocab_size = len(tokenizer)
        self.len = len(smiles_list)
        
        self.use_random_input_smiles = use_random_input_smiles
        self.use_random_target_smiles = use_random_target_smiles
        self.rsmiles = rsmiles
        self.corrupt = corrupt
        
        if rsmiles is None and (use_random_input_smiles or use_random_target_smiles):
            print('WARNING: The result of rdkit.Chem.MolToSmiles(..., doRandom=True) is NOT reproducible '
                  'because this function does not provide a way to control its random seed.')

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        smiles = self.smiles_list[item]
        mol = Chem.MolFromSmiles(smiles)
        
        # clear isotope
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        
        csmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, doRandom=False)
        if self.rsmiles is not None:
            rsmiles = self.rsmiles[item]
        else:
            rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)
        
        input_smiles = rsmiles if self.use_random_input_smiles else csmiles
        target_smiles = rsmiles if self.use_random_target_smiles else csmiles
        
        input_seq = self.tokenizer.parse(input_smiles)
        target_seq, atom_idx = self.tokenizer.parse(target_smiles, return_atom_idx=True)
        
        if self.corrupt:
            corrupted_input = _corrupt(input_seq, self.mask_token)
        else:
            corrupted_input = input_seq
        
        corrupted_input = torch.LongTensor(corrupted_input)
        
        target_seq = torch.LongTensor(target_seq)

        pp_graph, mapping = smiles2ppgraph(target_smiles)
        pp_graph.ndata['h'] = \
            torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
        
        mapping = torch.FloatTensor(mapping)
        mapping[:,pp_graph.num_nodes():] = -100  # torch cross entropy loss ignores -100 by default
        
        mapping_ = torch.ones(target_seq.shape[0], MAX_NUM_PP_GRAPHS)*-100
        mapping_[atom_idx,:] = mapping
        
        return corrupted_input, pp_graph, mapping_, target_seq

    @staticmethod
    def collate_fn(batch):
        pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')

        corrupted_inputs, pp_graphs, mappings, target_seqs, *other_descriptors = list(zip(*batch))

        corrupted_inputs = \
            pad_sequence(corrupted_inputs, batch_first=True, padding_value=pad_token)
        input_mask = (corrupted_inputs==pad_token).bool()
        
        pp_graphs = dgl.batch(pp_graphs)
        
        mappings = pad_sequence(mappings, batch_first=True, padding_value=-100)  # torch cross entropy loss ignores -100 by default, but we do not use cross_entropy_loss acctually
        
        target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_token)

        return corrupted_inputs, input_mask, pp_graphs, mappings, target_seqs


if __name__ == '__main__':
    run_test_tokenizer()
