import sys
import numpy as np
from scipy.io import loadmat
from Bio.PDB.PDBParser import PDBParser
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    df_rSASA_start = loadmat('../proteins/'+ arg1 +'_start_structure_sasa_data.mat')
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
    starting_structure = p.get_structure('starting_structure','../proteins/'+ arg1 + '_start_structure.pdb')
    coord_CA_start = get_coord_CA(starting_structure[0])
    
    # get respair and distance seperation
    Rg_start = get_Rg(coord_CA_start)
    L_peptide = np.shape(coord_CA_start)[0]
    res_pairs = []
    
    for i in range(0,L_peptide):
        for j in range(i+1,L_peptide):
            if i+1 in surface_res_id and j+1 in surface_res_id:
                distance_start = get_distance(coord_CA_start[i,:], coord_CA_start[j,:])
                if distance_start > Rg_start:
                    res_pairs.append([i+1, j+1])

    coord_CA_traj_start = np.load('../proteins/T4L_coord_CA_traj_cMD.npy')
    coord_CA_traj_target = np.load('../proteins/T4L_coord_CA_traj_cMD_target.npy')

    # get all pairwise distance
    N_sample = np.shape(coord_CA_traj_start)[0]
    N_feature = len(res_pairs)
    X1 = np.zeros((N_sample, N_feature))
    X2 = np.zeros((N_sample, N_feature))
    for i in range(0,N_sample):
        for j in range(0,N_feature):
            vector = coord_CA_traj_start[i][res_pairs[j][0]-1,:] - coord_CA_traj_start[i][res_pairs[j][1]-1,:]
            distance = np.sqrt(np.sum(vector*vector))
            X1[i, j] = distance
    
            vector = coord_CA_traj_target[i][res_pairs[j][0]-1,:] - coord_CA_traj_target[i][res_pairs[j][1]-1,:]
            distance = np.sqrt(np.sum(vector*vector))
            X2[i, j] = distance

    X = np.concatenate((X1, X2), axis=0)
    Y1 = np.zeros(np.shape(X1)[0])
    Y2 = np.ones(np.shape(X2)[0])
    Y = np.concatenate((Y1, Y2), axis=0)
    
    # train the model
    lda = LinearDiscriminantAnalysis(solver='svd',n_components=1)
    lda.fit(X,Y)
    
    # get eigenvector W
    W = lda.coef_
    W = W[0,:]
    W = W/np.sqrt(np.sum(W*W))

    N_restraints = 5

    index_sorted = np.argsort(np.abs(W))
    res_pair_1_sorted = []
    res_pair_2_sorted = []
    for i in range(0,N_restraints):
        res_pair_1_sorted.append(res_pairs[index_sorted[-1-i]][0])
        res_pair_2_sorted.append(res_pairs[index_sorted[-1-i]][1])
    res_pair_1_sorted = np.array(res_pair_1_sorted)
    res_pair_2_sorted = np.array(res_pair_2_sorted)
    
    res_pair_select = np.zeros([N_restraints, 2],dtype='int')
    res_pair_select[:,0] = res_pair_1_sorted
    res_pair_select[:,1] = res_pair_2_sorted
    
    np.savetxt('./' + arg1 + '_restraints_linear_discriminant_analysis.txt',res_pair_select,fmt='%4d')