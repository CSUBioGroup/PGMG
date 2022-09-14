
This repository contains PyTorch implementation of PGMG. A flexible and powerful tool for bioactive molecule generation.

> Through the guidance of pharmacophore, PGMG provides a flexible strategy to generate bioactive molecules 
> with structural diversity in various scenarios using a trained variational autoencoder.
> 
> *PGMG: A Pharmacophore-Guided Deep Learning Approach for Bioactive Molecule Generation*


### Training

Run `chembl_baseline.py` as `CUDA_VISIBLE_DEVICES=<gpu_num> python chembl_baseline.py <output_dir>`
- the `gpu_num` indicates which gpu you want to run the code
- the `output_dir` is the directory you want to store the trained model

Other configurations need to be changed inside `chembl_baseline.py`, including model settings and the data directory.


### Using PGMG to generate molecules

1. train PGMG
2. prepare pharmacophore models and convert them to required DGLGraphs.
3. load a trained model and run `result = model.generate(dgl_graphs)` to get results.

We provide two Jupyter notebooks in `./notebooks` to demonstrate how to prepare pharmacophore models in `dgl.DGLGraph` 
format as required from molecules or a `.phar` file. And we also demonstrate how to load a trained model and generate
molecules from prepared pharmacophore graphs in them.

### Evaluations

- use `Is_meet_phco_molecule(smiles,dgl_graph)` in `utils.match_eval` to calculate the match score between 
molecules and pharmacophores. 


### Requirements

- python>=3.8
- pytorch
- rdkit
- dgl
- fairseq
- numpy
- pandas
- tqdm
- einops

----
You may not use the material for any commercial purposes. If you have any question, please feel free to create an issue or email renyi.zhou@outlook.com.
