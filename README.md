# cons_reg_gnn_mol
Source code for the paper: Consistency-Regularized Graph Neural Networks for Molecular Property Prediction

## Data 
- MoleculeNet - https://moleculenet.org/

## Components
- **data/get_molnet.py** - data preprocessing functions
- **gnn/** - gnn backbone architectures
- **dataset.py** - data structure / data augmentation logics
- **main.py** - script for overall running code
- **trn_cons_reg.py** - trainer with training and inference methods
- **util.py** - util functions for model training / inference
- **run.sh** - run code example

## Dependencies
- **Pytorch=1.12.0**
- **DGL=1.1.1**
- **RDKit=2022.03.2**
- **Scikit-Learn=1.3.0**

## Citation
TBU
