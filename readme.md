
# PGMG

----
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>

https://arxiv.org/abs/2207.00821

This repository contains the PyTorch implementation of *PGMG: A Pharmacophore-Guided Deep Learning Approach for Bioactive Molecule Generation*. 

Through the guidance of pharmacophore, PGMG provides a flexible strategy to generate bioactive molecules with structural diversity in various scenarios using a trained variational autoencoder.


## Overview

PGMG aims to provide a flexible strategy to generate bioactive molecules with structural diversity in various scenarios, especially when the activity data is scarce.

PGMG only requires a pharmacophore hypothesis as input. The hypothesis can be constructed using only a few ligands or the structure of the receptor or the ligand-receptor complex.
The pharmacophore hypothesis will be transformed into a weighted complete graph based on the shortest-path distance and feed into the model. The model will then rapidly generate a large number of molecules that satisfy the conditions.

![pharmacophore_example.png](pics%2Fpharmacophore_example.png)

## Requirements
- python==3.8
- pytorch==1.13.0
- rdkit==2022.09.1
- dgl-cuda10.2==0.9.1
- fairseq==0.10.2  
- numpy==1.23.5
- pandas==1.5.2
- tqdm==4.64.1
- einops==0.6.0

> If you encounter `RuntimeError: mat1 and mat2 shapes cannot be multiplied`, please check the version of `fairseq` first. It should be `10.2`.

### Creating a new environment in conda
We recommend using `conda` to manage the environment. 

```bash
conda env create -f environment.yml
```


## Training

The training process with default parameters requires a GPU card with at least 10GB of memory.

Run `train_chembl_baseline.py` using the following command:
```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python train_chembl_baseline.py <output_dir> --show_progressbar
```
- the `gpu_num` indicates which gpu you want to run the code
- the `output_dir` is the directory you want to store the trained model

Other configurations need to be changed inside `train_chembl_baseline.py`, including model settings and the data directory.


## Using a trained PGMG model to generate molecules


**1. prepare the pharmacophore hypotheses**

First of all, you need some pharmacophore hypotheses. A pharmacophore is defined as a set of chemical features and their spatial information that is necessary for a drug to bind to a target and there are many ways to acquire one. 

If you have a biochemistry background, we strongly encourage you to build it yourself by stacking active ligands or analyzing the receptor structure. There are also many tools available. 
And you can always adjust the input hypothesis according to the results.

Apart from building it yourself, you can also acquire them by searching the literature or just randomly sampling 3-6 pharmacophore elements from a reference ligand to build some hypotheses and filtering the generated molecules afterwards.

**2. format the hypotheses**

The pharmacophore hypotheses should be provided in one of the two formats:

- the `.posp` format where the type of the pharmacophore points and the 3d positions are provided, see `data/phar_demo2.posp` for example.
- the `.edgep` format where the type of the pharmacophore points and the distances between each point are provided, see `data/phar_demo1.edgep` for example.

Pharmacophore types supported by default:
- AROM: aromatic ring
- POSC: cation
- HACC: hydrogen bond acceptor
- HDON: hydrogen bond donor
- HYBL/LHYBL: hydrophobic group (Hydrophobe, LumpedHydrophobe)

**3. generate**

Use the `generate.py` to generate molecules.

usage:
```text
python generate.py [-h] [--n_mol N_MOL] [--device DEVICE] [--filter] [--batch_size BATCH_SIZE] [--seed SEED] input_path output_dir model_path tokenizer_path

positional arguments:
  input_path            the input file path. If it is a directory, then every file ends with `.edgep` or `.posp` will be processed
  output_dir            the output directory
  model_path            the weights file (xxx.pth)
  tokenizer_path        the saved tokenizer (tokenizer.pkl)

optional arguments:
  -h, --help            show this help message and exit
  --n_mol N_MOL         number of generated molecules for each pharmacophore file
  --device DEVICE       `cpu` or `cuda`, default:'cpu'
  --filter              whether to save only the unique valid molecules
  --batch_size BATCH_SIZE
  --seed SEED
```

To run generation on the demo input:
```bash
python generate.py data/phar_demo1.edgep demo_result/ weights/chembl_fold0_epoch32.pth weights/tokenizer.pkl --filter --device cpu
```

## Evaluations

- use `Is_meet_phco_molecule(smiles,dgl_graph)` in `utils.match_eval` to calculate the match score between 
molecules and pharmacophores. 


----

## License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

For commercial use, please contact [limin@csu.edu.cn](mailto:limin@csu.edu.cn).
