import sys
import numpy as np
from scipy.io import loadmat
from Bio.PDB.PDBParser import PDBParser
import warnings
warnings.filterwarnings("ignore")

def get_coord_CA(structure):
    coord = []
    atom_name = []
    residues = structure.get_residues()
    for residue in residues:
        for atoms in residue:
            atom_name.append(atoms.get_name())
            coord.append(atoms.get_coord())
    atom_name = np.array(atom_name)
    coord = np.array(coord)
    coord_CA = coord[atom_name=='CA']
    return coord_CA

def get_distance(coord_1, coord_2):
    return np.sqrt(np.sum((coord_1-coord_2)*(coord_1-coord_2)))

def get_Rg(coord_CA):
    coord_mean = np.mean(coord_CA,axis=0)
    N = np.shape(coord_CA)[0]
    coord_CA_deviate = coord_CA - coord_mean
    return np.sqrt(np.sum(coord_CA_deviate*coord_CA_deviate)/N)
    
if __name__ == "__main__":

    arg1 = sys.argv[1]
    
    # exclude residues in the core
    df_rSASA_start = loadmat('./proteins/'+ arg1 +'_start_structure_sasa_data.mat')
    rSASA_start = df_rSASA_start['each_res_data'][:,1]
    
    surface_res_id = []
    core_res_id = []
    for i in range(0,len(rSASA_start)):
        if rSASA_start[i]>0.1:
            surface_res_id.append(i+1)
        else:
            core_res_id.append(i+1)
    
    # load starting and target structure
    p = PDBParser(PERMISSIVE=1)
    starting_structure = p.get_structure('starting_structure','./proteins/'+ arg1 + '_start_structure.pdb')
    coord_CA_start = get_coord_CA(starting_structure[0])
    
    # get respair and distance seperation
    Rg_start = get_Rg(coord_CA_start)
    L_peptide = np.shape(coord_CA_start)[0]
    res_pair_1 = []
    res_pair_2 = []
    separation = []
    
    for i in range(0,L_peptide):
        for j in range(i+1,L_peptide):
            if i+1 in surface_res_id and j+1 in surface_res_id:
                distance_start = get_distance(coord_CA_start[i,:], coord_CA_start[j,:])
                if distance_start > Rg_start:
                    res_pair_1.append(i+1)
                    res_pair_2.append(j+1)
                    separation.append(distance_start)

    index_sorted = np.argsort(seperation)
    seperation_sorted = []
    res_pair_1_sorted = []
    res_pair_2_sorted = []
    for i in range(0,len(index_sorted)):
        seperation_sorted.append(seperation[index_sorted[-1-i]])
        res_pair_1_sorted.append(res_pair_1[index_sorted[-1-i]])
        res_pair_2_sorted.append(res_pair_2[index_sorted[-1-i]])

    
    
    N_restraints = 100
    index_list = np.random.choice(len(res_pair_1),size=N_restraints,replace=False)
    res_pair_random = np.zeros([N_restraints, 2],dtype='int')
    res_pair_random[:,0] = res_pair_1[index_list]
    res_pair_random[:,1] = res_pair_2[index_list]
    
    np.savetxt('./' + arg1 + '_restraints_random.txt',res_pair_random,fmt='%4d')