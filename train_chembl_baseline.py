import os
import time
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool

import warnings

warnings.filterwarnings('ignore')

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.multiprocessing
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

torch.multiprocessing.set_sharing_strategy('file_system')

import dgl

import rdkit.Chem.AllChem as Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from model.pgmg import PGMG
from utils.smiles2ppgraph import MAX_NUM_PP_GRAPHS
from utils.utils import AverageMeter, timeSince, seed_torch
from utils.dataset import Tokenizer, SemiSmilesDataset

# calculated using the frequencies, trying to make the model pay more attention on rear atom types
PP_TYPE_WEIGHT = [1.4891304347826086, 1.0, 8.058823529411764, 1.0378787878787878, 1.8026315789473686, 2.174603174603175,
                  17.125]

# ====================================================
# Settings
# ====================================================


MODEL_DEFAULT_SETTINGS = {
    "max_len": 128,  # max length of generated SMILES
    "pp_v_dim": 7 + 1,  # dimension of pharmacophore embedding vectors
    "pp_e_dim": 1,  # dimension of pharmacophore graph edge (i.e. distance) embedding vectors
    "pp_encoder_n_layer": 4,  # number of pharmacophore gnn layers
    "hidden_dim": 384,  # hidden dimension
    "n_layers": 8,  # number of layers for transformer encoder and decoder
    "ff_dim": 1024,  # ff dim for transformer blocks
    "n_head": 8,  # number of attention heads for transformer blocks
    "remove_pp_dis": False,  # boolean, True to ignore any spatial information in pharmacophore graphs.
    "non_vae": False,  # boolean, True to disable the VAE framework

    'in': 'rs',  # whether to use random input SMILES
    'out': 'rs',  # whether to use random target SMILES
}

MODEL_SETTINGS = {
    # default
    'rs_mapping': {'non_vae': False, 'remove_pp_dis': False, 'in': 'rs', 'out': 'rs'},
    # others
    'cs_mapping': {'in': 'cs', 'out': 'cs'},
    'non_vae': {'non_vae': True},
    'remove_pp_dis': {'remove_pp_dis': True},
}


class CFG:
    fp16 = False  # whether to train with mixed precision (may have some bugs)

    # GENERAL SETTING
    print_freq = 200  # log frequency
    num_workers = 20

    # TRAINING
    init_lr = 3e-4
    weight_decay = 1e-6
    min_lr = 1e-6  # for CosineAnnealingLR scheduler
    T_max = 4  # for CosineAnnealingLR scheduler

    max_grad_norm = 5

    epochs = 32
    batch_size = 128
    gradient_accumulation_steps = 1
    valid_batch_size = 512
    valid_size = None  # can be used to set a fixed size validation dataset
    # we generated some molecules during training to track metrics like Validity
    gen_size = 2048  # number of pharmacophore graphs used to generate molecules during training
    gen_repeat = 2  # number of generated molecules for each input
    # the total number of generated molecules each time is `gen_size`*`gen_repeat`

    seed = 42  # random seed
    n_fold = 20  # k-fold validation
    valid_fold = 0  # which fold is used to as the validation dataset

    n_device = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))  # not used

    save_freq = 4  # save model every `save_freq` epochs
    skip_gen_test = 12  # skip saving and track Validity for `skip_gen_test` epochs

    # settings for reloading model and continue training
    init_epoch = 0  # 16
    reload_path = None  # './output/chembl_test/rs_mapping/fold0_epoch16.pth'
    reload_ignore = []


if CFG.init_epoch > 0:
    CFG.init_epoch -= 1


# ====================================================
# define training/testing steps
# ====================================================

def train_fn(train_loader, model, optimizer, epoch, scheduler, beta=1, scaler=None):
    assert not CFG.fp16 or (scaler is not None)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lm_losses = AverageMeter()
    kl_losses = AverageMeter()
    map_losses = AverageMeter()

    # switch to train mode
    model.train()

    start = end = time.time()

    accumulated_loss = 0
    grad_norm = -1

    N = len(train_loader)
    for step, batch_data in tqdm(enumerate(train_loader), disable=disable_tqdm, total=N):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, input_mask, pp_graphs, mappings, targets, *others = [i.to('cuda:0') for i in batch_data]

        batch_size = inputs.shape[0]

        if CFG.fp16:
            with amp.autocast():
                prediction_scores, mapping_scores, lm_loss, kl_loss = model(inputs, input_mask, pp_graphs, targets)
        else:
            prediction_scores, mapping_scores, lm_loss, kl_loss = model(inputs, input_mask, pp_graphs, targets)

        x = torch.zeros(batch_size, MAX_NUM_PP_GRAPHS, len(PP_TYPE_WEIGHT)).to('cuda:0')
        xx = pad_sequence(torch.split(pp_graphs.ndata['type'], tuple(pp_graphs.batch_num_nodes().cpu())),
                          batch_first=True)
        x[:, :xx.shape[1], :] = xx

        a = torch.Tensor(PP_TYPE_WEIGHT).to('cuda:0')
        sample_weight = x @ a  # (512, MAX_NUM_PP_GRAPHS)

        mapping_loss_weight = (mappings == 1) * (8 / (0.001 + (mappings == 1).sum(1))).unsqueeze(1)  # balance pos/neg samples
        mapping_loss_weight += (mappings != -100) * sample_weight.unsqueeze(1)  # balance rare pharmacophore types

        mapping_loss = F.binary_cross_entropy(mapping_scores, mappings, weight=mapping_loss_weight)

        #         loss = kl_loss*0.2+reshape_layer_l1+lm_loss
        loss = lm_loss + kl_loss * beta + mapping_loss
        accumulated_loss += loss

        # record loss
        losses.update(loss.item(), batch_size)
        lm_losses.update(lm_loss.item(), batch_size)
        kl_losses.update(kl_loss.item(), batch_size)
        map_losses.update(mapping_loss.item(), batch_size)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            accumulated_loss = accumulated_loss / CFG.gradient_accumulation_steps

            if CFG.fp16:
                scaler.scale(accumulated_loss).backward()
            else:
                accumulated_loss.backward()

            if CFG.fp16:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

            if CFG.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            accumulated_loss = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(f'Epoch: [{epoch + 1}][{step}/{len(train_loader)}] '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} '
                  f'LM Loss: {lm_losses.val:.4f}({lm_losses.avg:.4f}) '
                  f'KL Loss: {kl_losses.val:.4f}({kl_losses.avg:.4f}) '
                  f'Map Loss: {map_losses.val:.4f}({map_losses.avg:.4f}) '
                  f'Grad: {grad_norm:.4f} '
                  )
    return losses.avg


@torch.no_grad()
def valid_fn(valid_loader, model, epoch, beta=1, scaler=None):
    assert not CFG.fp16 or (scaler is not None)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lm_losses = AverageMeter()
    kl_losses = AverageMeter()
    map_losses = AverageMeter()
    map_accs = AverageMeter()

    # switch to eval mode
    model.eval()

    start = end = time.time()

    N = len(valid_loader)
    for step, batch_data in tqdm(enumerate(valid_loader), disable=disable_tqdm, total=N):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, input_mask, pp_graphs, mappings, targets, *others = [i.to('cuda:0') for i in batch_data]

        batch_size = inputs.shape[0]

        if CFG.fp16:
            with amp.autocast():
                prediction_scores, mapping_scores, lm_loss, kl_loss = model(inputs, input_mask, pp_graphs, targets)
        else:
            prediction_scores, mapping_scores, lm_loss, kl_loss = model(inputs, input_mask, pp_graphs, targets)

        x = torch.zeros(batch_size, MAX_NUM_PP_GRAPHS, len(PP_TYPE_WEIGHT)).to('cuda:0')
        xx = pad_sequence(torch.split(pp_graphs.ndata['type'], tuple(pp_graphs.batch_num_nodes().cpu())),
                          batch_first=True)
        x[:, :xx.shape[1], :] = xx

        a = torch.Tensor(PP_TYPE_WEIGHT).to('cuda:0')
        sample_weight = x @ a  # (512, MAX_NUM_PP_GRAPHS)

        mapping_loss_weight = (mappings == 1) * (8 / (0.001 + (mappings == 1).sum(1))).unsqueeze(
            1)  # balance pos/neg samples
        mapping_loss_weight += (mappings != -100) * sample_weight.unsqueeze(1)  # balance rare pharmacophore types

        mapping_loss = F.binary_cross_entropy(mapping_scores, mappings, weight=mapping_loss_weight)

        map_acc = ((mapping_scores[mappings == 0] < 0.5).sum() + (mapping_scores[mappings == 1] >= 0.5).sum()) / (
                mappings != -100).sum()

        loss = lm_loss + kl_loss * beta + mapping_loss

        # record loss
        losses.update(loss.item(), batch_size)
        lm_losses.update(lm_loss.item(), batch_size)
        kl_losses.update(kl_loss.item(), batch_size)
        map_losses.update(mapping_loss.item(), batch_size)

        map_accs.update(map_acc.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(f'VALID Epoch: [{epoch + 1}][{step}/{len(valid_loader)}] '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} '
                  f'LM Loss: {lm_losses.val:.4f}({lm_losses.avg:.4f}) '
                  f'KL Loss: {kl_losses.val:.4f}({kl_losses.avg:.4f}) '
                  f'Map Loss: {map_losses.val:.4f}({map_losses.avg:.4f}) '
                  f'Map Acc: {map_accs.val:.4f}({map_accs.avg:.4f}) '
                  )
    return losses.avg


def format_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    csmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, doRandom=False)

    return csmiles


@torch.no_grad()
def test_generate(valid_loader, model, epoch, random_sampling=False):
    from utils.match_eval import get_match_score

    # switch to eval mode
    model.eval()

    start = end = time.time()

    res = []
    pp_graph_list = []

    for step, batch_data in tqdm(enumerate(valid_loader), disable=disable_tqdm, total=len(valid_loader)):
        inputs, input_mask, pp_graphs, mappings, targets, *others = [i.to('cuda:0') for i in batch_data]
        predictions = model.generate(pp_graphs, random_sampling)
        res.extend(tokenizer.get_text(predictions))
        pp_graph_list.extend(dgl.unbatch(pp_graphs.to('cpu')))

    match_score = get_match_score(pp_graph_list, res, n_workers=CFG.num_workers, timeout=10)

    with Pool(CFG.num_workers) as pool:
        v_smiles = pool.map(format_smiles, res)

    valid_smiles = [i for i in v_smiles if i is not None]
    s_valid_smiles = set(valid_smiles)
    uniqueness = len(s_valid_smiles) / len(valid_smiles)
    novelty = len(s_valid_smiles - all_smiles) / len(s_valid_smiles)

    timeout_count = 0
    exceptions = 0
    for i in match_score:
        timeout_count += i == -2
        exceptions += i == -3

    valid_match_score = [i for i in match_score if i >= 0]

    end = time.time()

    print(f'GEN Epoch: [{epoch + 1}] '
          f'Time: {end - start} '
          f'Match Score: {np.mean(valid_match_score):.4f} '
          f'Validity: {(len(valid_smiles) / len(res)):.4f} '
          f'Uniqueness: {uniqueness:.4f} '
          f'Novelty: {novelty:.4f} '
          f'TimeoutCount: {timeout_count} '
          f'Exceptions: {exceptions} '
          )

    return np.mean(valid_match_score)


if __name__ == '__main__':

    # ====================================================
    # load configs
    # ====================================================

    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--model_type', choices=['rs_mapping', 'cs_mapping', 'non_vae', 'remove_pp_dis'],
                        default='rs_mapping')
    parser.add_argument('--show_progressbar', action='store_true')

    args = parser.parse_args()

    disable_tqdm = not args.show_progressbar

    output_dir = Path(args.output_dir) / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'OUTPUT_DIR: {output_dir}')

    seed_torch(seed=CFG.seed)

    print('CFG:')
    for k, v in vars(CFG).items():
        if k.startswith('_'):
            continue
        print(f'{k}:{v}')
    print('---------')

    # ====================================================
    # load dataset
    # ====================================================

    with open('data/chembl24_canon_train.pickle', 'rb') as f:
        train_smiles = pickle.load(f)
    with open('data/chembl24_canon_valid.pickle', 'rb') as f:
        valid_smiles = pickle.load(f)
    with open('data/chembl24_canon_test.pickle', 'rb') as f:
        test_smiles = pickle.load(f)

    all_smiles = set(train_smiles + valid_smiles + test_smiles)

    gen_smiles = list(np.random.RandomState(CFG.seed).
                      choice(valid_smiles, CFG.gen_size, replace=False)) * CFG.gen_repeat

    tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))

    with (output_dir / 'tokenizer_r_iso.pkl').open('wb') as f:
        pickle.dump(tokenizer, f)

    use_random_input_smiles = MODEL_SETTINGS[args.model_type].setdefault('in', 'rs') == 'rs'
    use_random_target_smiles = MODEL_SETTINGS[args.model_type].setdefault('out', 'rs') == 'rs'

    train_dataset = SemiSmilesDataset(train_smiles, tokenizer,
                                      use_random_input_smiles, use_random_target_smiles)
    valid_dataset = SemiSmilesDataset(valid_smiles, tokenizer,
                                      use_random_input_smiles, use_random_target_smiles)
    gen_dataset = SemiSmilesDataset(gen_smiles, tokenizer,
                                    use_random_input_smiles, use_random_target_smiles)

    print(f"========== the validation fold is {CFG.valid_fold} ==========")

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=train_dataset.collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=valid_dataset.collate_fn)

    gen_loader = DataLoader(gen_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=gen_dataset.collate_fn)

    # ====================================================
    # model & optimizer & scheduler
    # ====================================================

    model_params = dict(MODEL_DEFAULT_SETTINGS)

    for k, v in MODEL_SETTINGS[args.model_type].items():
        model_params[k] = v

    print('------model parameters------')
    for k, v in model_params.items():
        print(f'{k}:{v}')
    print('------------------------------')

    model = PGMG(model_params, tokenizer)
    if CFG.reload_path:
        print(f'reloading model weights from {CFG.reload_path}...')
        states = torch.load(CFG.reload_path, map_location=torch.device('cpu'))
        states['model'].update({k: model.state_dict()[k] for k in model.state_dict().keys()
                                if k.startswith(tuple(CFG.reload_ignore))})
        print(model.load_state_dict(states['model'], strict=False))

    model.to('cuda:0')

    optimizer = AdamW(model.parameters(), lr=CFG.init_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

    if CFG.reload_path:
        print(f'reloading optimizer & scheduler states from {CFG.reload_path}...')

        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])

    if CFG.fp16:
        print('fp16')
        scaler = amp.GradScaler()
    else:
        print('fp32')
        scaler = None


    # ====================================================
    # beta for KL annealing
    # ====================================================

    def gen_beta(start, end, T1, T2, T3):
        for i in range(T1):
            yield start
        log_s = np.log(start)
        log_e = np.log(end)
        T = T2 - T1
        AT = T3 - T1
        for i in range(T):
            cur_beta = np.exp(log_s + (log_e - log_s) / AT * i)
            yield cur_beta

        T = T3 - T2
        delta_beta = (end - cur_beta) / T
        for i in range(T):
            cur_beta += delta_beta
            yield cur_beta

        while True:
            yield end


    beta_f = gen_beta(3e-4, 1e-2, 6, 18, 24)
    for i in range(CFG.init_epoch):
        next(beta_f)

    # ====================================================
    # start training
    # ====================================================

    best_loss = np.inf

    for epoch in range(CFG.init_epoch, CFG.epochs):

        start_time = time.time()

        beta = next(beta_f)

        # train
        avg_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, beta, scaler=scaler)

        # eval
        val_loss = valid_fn(valid_loader, model, epoch, beta, scaler=scaler)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - start_time

        print(f'Epoch {epoch + 1} - beta {beta} avg_train_loss: {avg_loss:.4f} avg_valid_loss: {val_loss:.4f}  '
              f'time: {elapsed:.0f}s')

        if (epoch + 1) >= CFG.skip_gen_test and (epoch + 1) % CFG.save_freq == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        },
                       output_dir / f'fold{CFG.valid_fold}_epoch{epoch + 1}.pth')

            # mean_match_score = test_generate(gen_loader, model, epoch)
