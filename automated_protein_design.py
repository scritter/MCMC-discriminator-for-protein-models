# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:06:24 2019

@author: seth
"""

from numba import jit
import random
import numpy as np

@jit(nopython=True)
def monte_carlo_steps(input_sequence=None,\
                      h_i=None,\
                      J_ij=None,\
                      steps=1000,\
                      bti=1.0,\
                      bti_step_increase=0.0,\
                      tolerated_diversity=None,\
                      max_mutations=None,\
                      return_best=False):
    
    # sequnece is int encoded in (l,) numby array
    # h_i is Lx20 site-wise energy contributions
    # J_ij is LxLx20x20 (i,j,ai,aj) pair-wise energy contributions
    # steps is the number of monte-carlo steps to perform
    # bti is the inverse of the Boltzmann-coeff * Temperature term
    # bti_step_increase is the fractional increase of bti to apply per step
    # tolerated_diversity is Lx20 boolean designating what diveristy tolerated at each position
    
    sequence = np.copy(input_sequence)
    L = sequence.shape[0]
    steps = int(steps)
    
    # generate tolerance array
    # cols: (position, amino-acid)
    
    if tolerated_diversity is None:
        tolerated_diversity = np.ones((L, 20))==1.0
    
    if max_mutations is None:
        max_mutations = L
    
    tolerated_mutations_index = np.where(tolerated_diversity)
    tolerated_mutations = np.concatenate((tolerated_mutations_index[0].reshape(-1,1),\
                                          tolerated_mutations_index[1].reshape(-1,1)), axis=1)
    tolerated_mutations_length = tolerated_mutations.shape[0]
        
    if return_best:
        best_seq = np.copy(input_sequence)
        best_seq_energy = 0.0
    
    current_energy = 0.0
    for step in range(steps):
        mutation = random.randint(0,tolerated_mutations_length-1)
        i = tolerated_mutations[mutation, 0]
        A_i = tolerated_mutations[mutation, 1]
        
        delta_hi = h_i[i, A_i] - h_i[i, sequence[i]]
        delta_Jij = 0.0

        for j in range(L):
            if i != j:
                delta_Jij += (J_ij[i, j, A_i, sequence[j]] - J_ij[i, j, sequence[i], sequence[j]])
        
        if (np.exp(bti*(delta_hi+delta_Jij))>random.random()):
            sequence[i] = A_i
            if np.sum(sequence!=input_sequence)>max_mutations:
                sequence[i] = input_sequence[i]
            else:
                current_energy += delta_hi+delta_Jij
            
            if return_best&(current_energy > best_seq_energy):
                best_seq_energy = current_energy
                best_seq = np.copy(sequence)
            
        bti+=(bti*bti_step_increase)
        
    if return_best:
        return best_seq
    else:
        return sequence

@jit(nopython=True)
def best_single_mutations(input_sequence=None,\
                      h_i=None,\
                      J_ij=None,\
                      steps=1000,\
                      tolerated_diversity=None):
    
    # sequnece is int encoded in (l,) numby array
    # h_i is Lx20 site-wise energy contributions
    # J_ij is LxLx20x20 (i,j,ai,aj) pair-wise energy contributions
    # steps is the number of monte-carlo steps to perform
    # bti is the inverse of the Boltzmann-coeff * Temperature term
    # bti_step_increase is the fractional increase of bti to apply per step
    # tolerated_diversity is Lx20 boolean designating what diveristy tolerated at each position
    
    sequence = np.copy(input_sequence)
    L = sequence.shape[0]
    steps = int(steps)
    
    # generate tolerance array
    # cols: (position, amino-acid)
    
    if tolerated_diversity is None:
        tolerated_diversity = np.ones((L, 20))==1.0
    
    tolerated_mutations_index = np.where(tolerated_diversity)
    tolerated_mutations = np.concatenate((tolerated_mutations_index[0].reshape(-1,1),\
                                          tolerated_mutations_index[1].reshape(-1,1)), axis=1)
    tolerated_mutations_length = tolerated_mutations.shape[0]
        
    for step in range(steps):
        single_mutation_de = np.zeros((tolerated_mutations_length))
        
        for mutation in range(tolerated_mutations_length):
            
            i = tolerated_mutations[mutation, 0]
            A_i = tolerated_mutations[mutation, 1]
            
            delta_hi = h_i[i, A_i] - h_i[i, sequence[i]]
            delta_Jij = 0.0
    
            for j in range(L):
                if i != j:
                    delta_Jij += (J_ij[i, j, A_i, sequence[j]] - J_ij[i, j, sequence[i], sequence[j]])
            
            single_mutation_de[mutation] = delta_hi+delta_Jij
            
        best_mutation = np.where(np.max(single_mutation_de)==single_mutation_de)[0][0]
        sequence[tolerated_mutations[best_mutation, 0]] = tolerated_mutations[best_mutation, 1]
            
    return sequence