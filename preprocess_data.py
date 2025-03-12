# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit.Chem import AllChem
# %%

def generate_pocket(data_dir, distance=5):
    complex_id = os.listdir(data_dir)
    for cid in complex_id:
        
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}_out.pdb")
        protein_path= os.path.join(complex_dir, "5xr8.pdb")

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid} around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(data_df))

    for i, row in data_df.iterrows():
        cid, pKa = str(row['id']), float(row['label'])
        complex_dir = os.path.join(data_dir, cid)
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(data_dir, cid, f'{cid}_out.pdb')

        save_path = os.path.join(complex_dir, f"{cid}.rdkit")
        if os.path.exists(save_path):
            continue

        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True, sanitize=False)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=False)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'pdb'
    data_root = './data/SCsDB'
    data_dir = os.path.join(data_root, 'out')
    data_df = pd.read_csv(os.path.join(data_root, 'scs_cb1r.csv'))

    ## generate pocket within 5Å around ligand 
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)



# %%
