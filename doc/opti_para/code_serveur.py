import numpy as np
import numpy.linalg
import torch
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
import random

## Données initiales

#Taille du domaine entier
w = 32

# Nombre d'itération
iter_num = 21

# Fournit les coordonnées du centre du domaine
a = w/2

#Demi-longueur du rectangle, qui restera constante
Longueur = 3*a/4

numpy.save(f"donnees_init.npy",np.array([w,iter_num,Longueur]))

## Création de la matrice d'incidence
x = eye(w-1,w,1) - eye(w-1,w)
y = eye(w-1,w,1) - eye(w-1,w)
B = vstack([kron(eye(w),x),kron(y,eye(w))])
B = torch.from_numpy(B.toarray())

## Etape de création du rectangle

# Construction du tableau initial
init_table = a*torch.ones((w,w))
y = w//2
for i in range(w):
    for j in range(w):
        if abs(j-y)<Longueur:
            init_table[i,j]=abs(i-y)

# Fonction d'activation
sigma = lambda x : 1/(1+torch.exp(-x))

# Construction du rectangle dans le domaine: l est la demi-largeur du rectangle, c'est l qu'on optimise.
rect = lambda l : sigma(l-init_table)


## Fonction de coût
def loss_fun(l) :
        # Construction du laplacien
        BM = B @ torch.diag(rect(l).flatten().double())
        L = - BM.t() @ BM

        # récupération des éléments propres
        U,S,V = torch.svd(-L)
        S_f = S[S>1e-5]

        return (4 - S_f[-2]/S_f[-1])**2 + (9 - S_f[-3]/S_f[-1])**2

## Fonction de recherche du pas
def backtracking_line_search(y,p):
    alpha = 1/4
    beta = 1/2
    t=1
    while loss_fun(y-t*p) > loss_fun(y) - alpha * t * p.norm() :
        t = beta * t
    return t

## Initialisation de la descente de gradient

# On prend l aléatoire entre 0 et la demi-largeur du domaine entier
l=np.array(a*random.random())
numpy.save(f"param_-1.npy",l)
l = torch.from_numpy(l)
l.requires_grad = True

# Initialisation des listes d'enregistrement des données
I = []
loss_tab = []
Partiel1 = []
Partiel2 = []
fondamental = []

## Descente
for i in range(iter_num):
    # forward pass
    loss = loss_fun(l)

    # backward pass
    loss.backward()
    grad_l = l.grad

    # Maj du pas
    learning_rate = backtracking_line_search(l,grad_l)
    with torch.no_grad():
        l -= grad_l * learning_rate
        l.grad.zero_()

    # Calcul de données pour l'affichage
    BM = B @ torch.diag(rect(l).flatten().double())
    L = - BM.t() @ BM
    U,S,V = torch.svd(-L)
    S_f = S[S>1e-5].detach().numpy().tolist()
    Partiel1.append(S_f[-2]/S_f[-1])
    Partiel2.append(S_f[-3]/S_f[-1])
    loss_tab.append(loss.detach().numpy().tolist())
    fondamental.append(S_f[-1])
    I.append(i)

    # Enregistrement régulier des données
    if i%10 == 0:
        numpy.save(f"iteration_num_{i}.npy", np.array(I))
        numpy.save(f"loss_tab_num_{i}.npy", np.array(loss_tab))
        numpy.save(f"partiel1_num_{i}.npy", np.array(Partiel1))
        numpy.save(f"partiel2_num_{i}.npy", np.array(Partiel2))
        numpy.save(f"param_{i}.npy", l.detach().numpy())
        numpy.save(f"fondamental_num_{i}.npy", np.array(fondamental))
