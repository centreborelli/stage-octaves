## Le code ci-dessous fournit une matrice dont la plus grand valeur propre est un paramètre theta.

import numpy as np
import numpy.linalg
import torch

theta = np.pi
n = 4 # Dimension

# La fonction PowerIteration prend en argument une matrice, une prédiction de vecteur propre, un nombre maximum d'itération ainsi qu'une tolérance de précision.
# Elle renvoit la valeur absolue de la plus grande valeur propre de la matrice et le vecteur propre associé.

def PowerIteration(v, A, maxIters, tolerance):
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

# Prédiction

v0 = np.random.rand(n,1)
v0 = torch.from_numpy(v0)

# Fonction de coût

def loss_fun(vM) :
  lamb, v = PowerIteration(v0 , vM.reshape(n,n), 40, 10e-6)
  return (theta-lamb)**2

# Recherche du pas

def backtracking_line_search(y,p):
    alpha = 1/4
    beta = 1/2
    t=1
    while loss_fun(y-t*p) > loss_fun(y) - alpha * t * p.norm() :
        t = beta * t
    return t

# Initialisation de la descente de gradient

iter_num = 500
vX = np.eye(n).flatten()
vX = torch.from_numpy(vX)
vX.requires_grad = True

# Descente

for i in range(iter_num):
    loss = loss_fun(vX) # forward pass
    loss.backward() # backward pass
    grad_vX = vX.grad
    learning_rate = backtracking_line_search(vX,grad_vX)
    with torch.no_grad():
        vX -= grad_vX * learning_rate
        vX.grad.zero_()

#Résultat

M = vX.reshape(n,n)

print('argmin =', M)

lamb , v = PowerIteration(v0,M,40,1e-6)

print('rayon spectral M =', lamb)