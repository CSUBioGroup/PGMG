from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops.layers.torch import Rearrange
from fairseq.modules import MultiheadAttention

from model.ggcn_layers import GGCNEncoderBlock
from model.transformer_blocks import PositionalEncoding, TransformerEncoder, TransformerDecoder

from utils.smiles2ppgraph import MAX_NUM_PP_GRAPHS


class PGMG(nn.Module):
    PARAM_SET = {'max_len',  # max length of generated molecules
                 'pp_v_dim',  # dimension of pharmacophore embedding vectors
                 'pp_e_dim',   # dimension of pharmacophore graph edge (i.e. distance) embedding vectors
                 'pp_encoder_n_layer',  # number of pharmacophore gnn layers
                 'hidden_dim',  # hidden dimension
                 'n_layers',  # number of layers for transformer encoder and decoder
                 'ff_dim',  # ff dim for transformer blocks
                 'n_head',  # number of attention heads for transformer blocks
                 'non_vae',  # boolean, True to disable the VAE framework
                 'remove_pp_dis'  # boolean, True to ignore any spatial information in pharmacophore graphs.
                 }

    def __init__(self, params, tokenizer):
        super().__init__()

        wrong_params = set(params.keys()) - PGMG.PARAM_SET
        print(f"WARNING: parameter(s) not used: {','.join(wrong_params)}")

        self.non_vae = params.setdefault('non_vae', False)
        self.remove_pp_dis = params.setdefault('remove_pp_dis', False)

        vocab_size = len(tokenizer)

        hidden_dim = params['hidden_dim']

        self.pp_e_init = nn.Linear(params['pp_e_dim'], hidden_dim)
        self.pp_v_init = nn.Linear(params['pp_v_dim'], hidden_dim)
        self.pp_encoder = GGCNEncoderBlock(hidden_dim, hidden_dim, n_layers=params['pp_encoder_n_layer'], dropout=0,
                                           readout_pooling='max', batch_norm=True, residual=True)

        n_head = params['n_head']
        ff_dim = params['ff_dim']
        n_layers = params['n_layers']

        self.encoder = TransformerEncoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)
        self.attention = MultiheadAttention(hidden_dim, n_head, dropout=0.1)

        self.dencoder = TransformerEncoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)  # can be removed
        self.decoder = TransformerDecoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)

        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=params['max_len'])

        self.word_embed = nn.Embedding(vocab_size, hidden_dim)

        self.word_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )

        torch.nn.init.zeros_(self.word_pred[3].bias)

        self.vocab_size = vocab_size
        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.max_len = params['max_len']

        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.expand = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                Rearrange('batch_size h -> 1 batch_size h')
            )

        self.pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))
        self.zz_seg_encoding = nn.Parameter(torch.randn(hidden_dim))

        self.mapping_transform_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mapping_transform_p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.hidden_dim = hidden_dim

    def sample(self, batch_size, device):
        z = torch.randn(batch_size, self.hidden_dim).to(device)
        return z

    def calculate_z(self, inputs, input_mask, vvs, pp_mask):
        x = self.word_embed(inputs)
        xt = x.permute(1, 0, 2).contiguous()  # (seq,batch,feat)
        xt = self.pos_encoding(xt)

        ppxt = torch.cat((vvs, xt), dim=0)
        encoder_full_mask = torch.cat((pp_mask, input_mask), dim=1)  # batch seq_plus

        ppxt = self.encoder(ppxt, encoder_full_mask)  # (s b f), input masks need not transpose
        xxt = ppxt[MAX_NUM_PP_GRAPHS:, :, :]

        foo = xxt.new_ones(1, *xxt.shape[1:])
        z, _ = self.attention(foo, xxt, xxt, key_padding_mask=input_mask)
        z = z.squeeze(0)

        z, kl_loss = self.resample(z)

        return z, kl_loss

    @amp.custom_fwd(cast_inputs=torch.float32)
    def resample(self, z):
        batch_size = z.size(0)
        if self.non_vae:
            return torch.randn(batch_size, self.hidden_dim).to(z.device), z.new_zeros(1)

        z_mean = self.mean(z)
        z_log_var = -torch.abs(self.var(z))

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / batch_size

        epsilon = torch.randn_like(z_mean).to(z.device)
        z_ = z_mean + torch.exp(z_log_var / 2) * epsilon

        return z_, kl_loss

    @torch.jit.ignore
    def process_p(self, pp_graphs):
        v = self.pp_v_init(pp_graphs.ndata['h'])
        _e = pp_graphs.edata['h']
        if self.remove_pp_dis:
            _e = torch.zeros_like(_e)
        e = self.pp_e_init(_e)
        v, e = self.pp_encoder.forward_feature(pp_graphs, v, e)
        vv = pad_sequence(torch.split(v, pp_graphs.batch_num_nodes().tolist()), batch_first=False, padding_value=-999)
        vv2 = vv.new_ones((MAX_NUM_PP_GRAPHS, pp_graphs.batch_size, vv.shape[2])) * -999
        vv2[:vv.shape[0], :, :] = vv
        vv = vv2
        pp_mask = (vv[:, :, 0].T == -999).bool()  # batch, seq
        vvs = vv + self.pp_seg_encoding  # seq,batch,feat
        return vv, vvs, pp_mask

    def expand_then_fusing(self, z, pp_mask, vvs):
        zz = self.expand(z)  # seq batch feat
        zz = self.pos_encoding(zz)  ###
        zzs = zz + self.zz_seg_encoding

        # cat pp and latent
        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        full_mask = torch.cat((pp_mask, full_mask), dim=1)  # batch seq_plus

        zzz = torch.cat((vvs, zzs), dim=0)  # seq_plus batch feat
        #if not self.remove_dencoder:
        zzz = self.dencoder(zzz, full_mask)

        return zzz, full_mask

    @torch.jit.unused
    def forward(self, inputs, input_mask, pp_graphs, targets):
        vv, vvs, pp_mask = self.process_p(pp_graphs)

        z, kl_loss = self.calculate_z(inputs, input_mask, vvs, pp_mask)

        zzz, encoder_mask = self.expand_then_fusing(z, pp_mask, vvs)

        # target
        _, target_length = targets.shape
        target_mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool),
                                 diagonal=1).to(targets.device)
        target_embed = self.word_embed(targets)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2).contiguous())

        # predict
        output = self.decoder(target_embed, zzz,
                              x_mask=target_mask, mem_padding_mask=encoder_mask).permute(1, 0, 2).contiguous()
        prediction_scores = self.word_pred(output)  # batch_size, sequence_length, class

        # mapping 
        mxx = self.mapping_transform_v(output)  # b,s,f
        mvv = self.mapping_transform_p(vv)  # s,b,f
        mapping_scores = torch.sigmoid(torch.bmm(mxx, mvv.permute(1, 2, 0).contiguous()))

        # loss
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        lm_loss = F.cross_entropy(shifted_prediction_scores.view(-1, self.vocab_size), targets.view(-1),
                                  ignore_index=self.pad_value)

        return prediction_scores, mapping_scores, lm_loss, kl_loss

    def _generate(self, zzz, encoder_mask, random_sample, return_score=False):
        batch_size = zzz.shape[1]
        device = zzz.device

        token = torch.full((batch_size, self.max_len), self.pad_value, dtype=torch.long, device=device)
        token[:, 0] = self.sos_value

        text_pos = self.pos_encoding.pe

        text_embed = self.word_embed(token[:, 0])
        text_embed = text_embed + text_pos[0]
        text_embed = text_embed.unsqueeze(0)

        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
        )

        if return_score:
            scores = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for t in range(1, self.max_len):
            one = self.decoder.forward_one(text_embed, zzz, incremental_state, mem_padding_mask=encoder_mask)
            one = one.squeeze(0)
            # print(incremental_state.keys())

            l = self.word_pred(one)  # b, f
            if return_score:
                scores.append(l)
            if random_sample:
                k = torch.multinomial(torch.softmax(l, 1), 1).squeeze(1)
            else:
                k = torch.argmax(l, -1)  # predict max
            token[:, t] = k

            finished |= k == self.eos_value
            if finished.all():
                break

            text_embed = self.word_embed(k)
            text_embed = text_embed + text_pos[t]  #
            text_embed = text_embed.unsqueeze(0)

        predict = token[:, 1:]

        if return_score:
            return predict, torch.stack(scores, dim=1)
        return predict

    def ag_forward(self, inputs, input_mask, pp_graphs, random_sample=False):
        vv, vvs, pp_mask = self.process_p(pp_graphs)

        z, kl_loss = self.calculate_z(inputs, input_mask, vvs, pp_mask)

        zzz, encoder_mask = self.expand_then_fusing(z, pp_mask, vvs)

        predict, scores = self._generate(zzz, random_sample=random_sample, return_score=True)

        return predict, scores, kl_loss

    @torch.jit.export
    @torch.no_grad()
    def generate(self, pp_graphs, random_sample=False, return_z=False):
        vv, vvs, pp_mask = self.process_p(pp_graphs)

        z = self.sample(pp_graphs.batch_size, pp_graphs.device)

        zzz, encoder_mask = self.expand_then_fusing(z, pp_mask, vvs)

        predict = self._generate(zzz, encoder_mask, random_sample=random_sample, return_score=False)
        
        if return_z:
            return predict, z.detach().cpu().numpy()
        return predict
