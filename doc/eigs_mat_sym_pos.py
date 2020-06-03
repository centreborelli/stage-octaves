'''
Recherche du spectre complet d'une matrice symétrique définie positive
'''

import numpy as np
import numpy.linalg
import torch

n = 5 # taille de la matrice

# vecteur pour les initialisations
v0 = np.array([[1.]]*n)  # il est important que se soit un vecteur colonne !
v0 = torch.from_numpy(v0)



def PowerIteration(A,v=v0, maxIters=40, tolerance=1e-6): 
   v = v / v.norm()
   iteration = 0
   lambdaOld = 0
   while iteration <= maxIters:
     z = A @ v
     lamb = z.norm()
     v = z / lamb
     iteration = iteration + 1
     if abs((lamb - lambdaOld)/lamb) < tolerance:
       return lamb, v
     lambdaOld = lamb
   if abs((lamb - lambdaOld)/lamb) >= tolerance:
     print('No convergence')

def eigs(A) :
  sp = []
  vV = []
  for i in range(n) :
    lamb , v = PowerIteration(A)
    sp.append(lamb)
    vV.append(v)
    A = A - lamb * (v @ v.t())  # on retire à A le projeté orthogonal sur v
  return sp , vV
    



A = np.diag(np.arange(1,n+1,1.))
A = torch.from_numpy(A)

eigs(A)
