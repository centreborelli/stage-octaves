import numpy as np
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
from math import sqrt
import torch

w = 11 # Dimension de la grille

# Construction de la matrice d'incidence

def grid_incidence(h, w):
    x = np.eye(w - 1, w, 1) - np.eye(w  -1, w)
    y = np.eye(h - 1, h, 1) - np.eye(h - 1, h)
    B = np.vstack([np.kron(np.eye(h),x) , np.kron(y,np.eye(w)) ])
    return B

B = grid_incidence(w,w)
B = torch.from_numpy(B).double()

# Fonction de construction du rectangle

def image_of_a_rect(l,L): # l : largeur, L : longueur
    im = torch.zeros((w,w))
    y = w//2
    for i in range(w) :
        for j in range(w) :
            if abs(y-i) < l * y and abs(y-j) < L * y:
                im[i,j] = 0.7
            else :
                im[i,j] = 0.
    return im

# Fonction de coût
def loss_fun(couple) :
    l = couple[0]
    L = couple[1]
    M = image_of_a_rect(l,L).double)()
    dM = M.flatten()
    BM = B @ torch.diag(dM)
    L = - BM.t() @ BM
    U,S,V = torch.svd(-L) # récupération des éléments propres
    S_f = S[S>1e-5]
    return (4 - S_f[-2]/S_f[-1])**2 + (9 - S_f[-3]/S_f[-1])**2 + (16 - S_f[-4]/S_f[-1])**2

# Fonction de recherche du pas
def backtracking_line_search(y,p):
    alpha = 1/4
    beta = 1/2
    t=1.
    while loss_fun(y-t*p) > loss_fun(y) - alpha * t * p.norm() :
        t = beta * t
    return t

# Initialisation de la descente de gradient
l = 0.3
L = 0.7
ep = 0.7
couple = torch.tensor([l,L],requires_grad=True)

iter_num = 51

# Descente
for i in range(iter_num):
    loss = loss_fun(couple) # forward pass
    loss.backward() # backward pass
    grad_couple = couple.grad
    learning_rate_couple = backtracking_line_search(couple,grad_couple)
    with torch.no_grad():
        couple -= grad_couple * learning_rate_couple
        couple.grad.zero_()

# Résultat final

l = couple[0]
L = couple[1]
M = image_of_a_rect(l,L).double()
np.save(f"image.npy", M.detach().numpy())