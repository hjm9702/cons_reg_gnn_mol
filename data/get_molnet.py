import numpy as np
import pandas as pd
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import ChemicalFeatures
import argparse, pickle
from sklearn.model_selection import train_test_split


def _DA(mol):

        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

def _chirality(atom):

    if atom.HasProp('Chirality'):
        c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
    else:
        c_list = [0, 0]

    return c_list
    

def _stereochemistry(bond):

    if bond.HasProp('Stereochemistry'):
        s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
    else:
        s_list = [0, 0]

    return s_list    

def add_mol(mol_dict, mol):

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    D_list, A_list = _DA(mol)
    
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetAtomicNum()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
       
    
    node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    
    
    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
        bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]
                
        edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
        
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)

    
    
    return mol_dict



def preprocess(d_name, smiles_list, ys, task_type, d_path):
    
    length = len(smiles_list)
    
    mol_dict = {'mols': [],
                'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'smiles': [],
                'y': [],
                'task_type' : task_type
                }


    for i in range(length):
        
        mol = Chem.MolFromSmiles(smiles_list[i])
        
        try:
            Chem.SanitizeMol(mol)
            si = Chem.FindPotentialStereo(mol)
            for element in si:
                if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                    mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                    mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
            assert '.' not in Chem.MolToSmiles(mol)
        except:
            continue

        
        
        

        mol_dict['mols'].append(mol)
        mol_dict['smiles'].append(smiles_list[i])
        mol_dict['y'].append(ys[i])
        
        mol_dict = add_mol(mol_dict, mol)

        
        if (i+1)%1000==0: 
            print(f'{i+1}/{length} processed')

    

    print(mol_dict.keys())
    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    mol_dict['smiles'] = np.array(mol_dict['smiles'])
    mol_dict['y'] = np.vstack(mol_dict['y'])




    for key in mol_dict.keys(): 
        if key in ['mols', 'task_type', 'mol_brics']:
            continue
        print(key, mol_dict[key].shape, mol_dict[key].dtype)
        
    with open(os.path.join(d_path, '%s_graph.npz'%d_name), 'wb') as f:
        pickle.dump([mol_dict], f, protocol=5)





def get_molnet_graph(d_path, d_name):


    dataset = pd.read_csv(os.path.join(d_path, '%s.csv'%d_name))

    if d_name == 'esol':
        task_type = 'reg'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset['measured log solubility in mols per litre'].to_numpy().reshape(-1,1)

    elif d_name == 'freesolv':
        task_type = 'reg'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset['expt'].to_numpy().reshape(-1,1)
    
    elif d_name == 'lipophilicity':
        task_type = 'reg'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset['exp'].to_numpy().reshape(-1,1)
    
    elif d_name == 'qm9':
        task_type = 'reg'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset[['mu','alpha','homo','lumo','gap','r2','zpve', 'cv']].to_numpy()
    
    elif d_name == 'bace':
        task_type = 'clf'
        smiles_list = dataset['mol'].to_numpy()
        ys = dataset['Class'].to_numpy().reshape(-1,1)

    elif d_name == 'bbbp':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset['p_np'].to_numpy().reshape(-1,1)

    elif d_name == 'clintox':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset[['FDA_APPROVED', 'CT_TOX']].to_numpy()

    elif d_name == 'sider':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset.iloc[:, 1:].to_numpy()

    elif d_name == 'tox21':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset.iloc[:, :12].to_numpy()

    elif d_name == 'toxcast':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset.iloc[:, 1:].to_numpy()
    
    elif d_name == 'hiv':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset['HIV_active'].to_numpy().reshape(-1,1)


        _, sub_idx = train_test_split(np.arange(len(smiles_list)), test_size=20000, random_state = 27407)
        smiles_list = smiles_list[sub_idx]
        ys = ys[sub_idx]

    elif d_name == 'muv':
        task_type = 'clf'
        smiles_list = dataset['smiles'].to_numpy()
        ys = dataset.iloc[:, :17].to_numpy()


        _, sub_idx = train_test_split(np.arange(len(smiles_list)), test_size=20000, random_state = 27407)
        smiles_list = smiles_list[sub_idx]
        ys = ys[sub_idx]

    
    global atom_list, charge_list, degree_list, valence_list, hybridization_list, hydrogen_list, ringsize_list, bond_list, chem_feature_factory

    
    atom_list = list(range(0, 118))
    charge_list = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 5, 6, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    

    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    rdBase.DisableLog('rdApp.error') 
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))



    preprocess(d_name, smiles_list, ys, task_type, d_path)







if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dpath', type=str, default='./molnet_file/')
    args = arg_parser.parse_args()
    
    d_path = args.dpath

    for d_name in ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'hiv', 'muv']:
        get_molnet_graph(d_path, d_name)

    
    
