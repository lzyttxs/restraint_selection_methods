#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from scipy.io import loadmat
import scipy.special as sc
from Bio.PDB.PDBParser import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
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

def get_rmsd(coord_ref, coord_traj, sup):
    sup.set(coord_ref, coord_traj)
    sup.run()
    rmsd = sup.get_rms()
    return rmsd

def get_RMSD_average(distance_traj, RMSD_matrix, test_pairs_index):
    size = np.shape(distance_traj)[0]
    P_value_matrix = np.ones([size, size])
    for i in range(0,size):
        for j in range(i+1,size):
            chi_square = 0
            N = len(test_pairs_index)
            for k in range(0,len(test_pairs_index)):
                R_conf = distance_traj[i,test_pairs_index[k]]
                R_ref = distance_traj[j,test_pairs_index[k]]
                delta_R = 0.03*(R_ref+R_conf)
                chi_square = chi_square + ((R_conf - R_ref)/delta_R)**2
            P_value = 1 - sc.gammainc(N/2, chi_square/2)
            P_value_matrix[i,j] = P_value
            P_value_matrix[j,i] = P_value
    RMSD = np.mean(np.sum(RMSD_matrix*P_value_matrix,axis=1)/(np.sum(P_value_matrix,axis=1)-1))
    return RMSD

def get_RMSD_average_from_chi_square(chi_square_matrix, RMSD_matrix, N):
    P_value_matrix = 1 - sc.gammainc(N/2, chi_square_matrix/2)
    RMSD = np.mean(np.sum(RMSD_matrix*P_value_matrix,axis=1)/(np.sum(P_value_matrix,axis=1)-1))
    return RMSD

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
    
    # get all pairwise distance
    coord_CA_traj = np.load('../proteins/T4L_coord_CA_traj_NMA.npy')
    N_sample = np.shape(coord_CA_traj)[0]
    N_feature = len(res_pairs)
    distance_traj = np.zeros((N_sample, N_feature))
    for i in range(0,N_sample):
        for j in range(0,N_feature):
            vector = coord_CA_traj[i][res_pairs[j][0]-1,:] - coord_CA_traj[i][res_pairs[j][1]-1,:]
            distance = np.sqrt(np.sum(vector*vector))
            distance_traj[i, j] = distance
        
    # get RMSD matrix
    RMSD_matrix = np.zeros([N_sample, N_sample])
    sup = SVDSuperimposer()
    for i in range(0,N_sample):
        for j in range(i+1,N_sample):
            rmsd = get_rmsd(coord_CA_traj[i], coord_CA_traj[j], sup)
            RMSD_matrix[i,j] = rmsd
            RMSD_matrix[j,i] = rmsd

    # get all pairwise chi square
    chi_square_all = np.zeros([len(res_pairs), N_sample, N_sample])
    for k in range(0,len(res_pairs)):
        for i in range(0,N_sample):
            for j in range(i+1,N_sample):
                R_conf = distance_traj[i,k]
                R_ref = distance_traj[j,k]
                delta_R = 0.03*(R_ref+R_conf)
                chi_square = ((R_conf - R_ref)/delta_R)**2
                chi_square_all[k,i,j] = chi_square
                chi_square_all[k,j,i] = chi_square

    # select pairs
    res_pair_select = []
    all_index = list(range(0,len(res_pairs)))
    chi_square_matrix = np.zeros([N_sample, N_sample])
    
    N_restraints = 5
    for N in range(1,N_restraints+1):
        RMSD_min = float('inf')
        for j in all_index:
            chi_square_matrix_current = chi_square_matrix + chi_square_all[j,:,:]
            RMSD = get_RMSD_average_from_chi_square(chi_square_matrix_current, RMSD_matrix, N)
            if RMSD < RMSD_min:
                bestPair = res_pairs[j]
                bestPair_index = j
                RMSD_min = RMSD
        res_pair_select.append(bestPair)
        all_index.remove(bestPair_index)
        chi_square_matrix = chi_square_matrix + chi_square_all[bestPair_index,:,:]

    np.savetxt('./' + arg1 + '_restraints_normal_mode_analysis.txt',res_pair_select,fmt='%4d')