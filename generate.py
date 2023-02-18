import argparse
import pickle
from pathlib import Path

import dgl
import rdkit
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from model.pgmg import PGMG
from utils.file_utils import load_phar_file
from utils.utils import seed_torch

RDLogger.DisableLog('rdApp.*')

def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model_params = {
        "max_len": 128,
        "pp_v_dim": 7 + 1,
        "pp_e_dim": 1,
        "pp_encoder_n_layer": 4,
        "hidden_dim": 384,
        "n_layers": 8,
        "ff_dim": 1024,
        "n_head": 8,
    }

    model = PGMG(model_params, tokenizer)
    states = torch.load(model_path, map_location='cpu')
    print(model.load_state_dict(states['model'], strict=False))

    return model, tokenizer

def format_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True)

    return smiles


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=Path, help='the input file path. If it is a directory, then every file '
                                                      'ends with `.edgep` or `.posp` will be processed')
    parser.add_argument('output_dir', type=Path, help='the output directory')
    parser.add_argument('model_path', type=Path, help='the weights file (xxx.pth)')
    parser.add_argument('tokenizer_path', type=Path, help='the saved tokenizer (tokenizer.pkl)')

    parser.add_argument('--n_mol', type=int, default=10000, help='number of generated molecules for each '
                                                                 'pharmacophore file')
    parser.add_argument('--device', type=str, default='cpu', help='`cpu` or `cuda`')
    parser.add_argument('--filter', action='store_true', help='whether to save only the unique valid molecules')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()

    if args.seed != -1:
        seed_torch(args.seed)

    if args.input_path.is_dir():
        files = list(args.input_path.glob('*.posp')) + list(args.input_path.glob('*.edgep'))
    else:
        assert args.input_path.suffix in ('.edgep', '.posp')
        files = [args.input_path]

    args.output_dir.mkdir(parents=False, exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.tokenizer_path)

    model.eval()
    model.to(args.device)

    for file in files:
        output_path = args.output_dir / f'{file.stem}_result.txt'

        g = load_phar_file(file)

        g_batch = [g] * args.batch_size
        g_batch = dgl.batch(g_batch).to(args.device)
        n_epoch = (args.n_mol + args.batch_size - 1) // args.batch_size

        res = []
        for i in tqdm(range(n_epoch)):
            res.extend(tokenizer.get_text(model.generate(g_batch)))
        res = res[:args.n_mol]

        if args.filter:
            res = [format_smiles(i) for i in res]
            res = [i for i in res if i]
            res = list(set(res))

        output_path.write_text('\n'.join(res))

    print('done')

