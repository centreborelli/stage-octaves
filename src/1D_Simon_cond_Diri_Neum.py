"""
Conditions de Dirichlet, Neumann et Mi-Dirichlet, Mi-Neumann en 1D
"""

import numpy as np
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

num_shown = 9  # nombres d'éléments propres à afficher

## Conditions de Dirichlet

w = 600  # nombres de noeuds dans le graphe
B = sp.eye(w-1,w,1) - sp.eye(w-1,w)  # matrice d'incidence pour le graphe ligne
N = w//5  # on divise la taille pour fixer 1/5 de la corde nulle aux bords

A = sp.diags([0]*N + [1]*3*N + [0]*N)  # matrice d'inertie de taille (w,w)

BA = B @ A  
L = - BM.T @ BM     # matrice du laplacien avec les conditions de Dirichlet

Va,Ve = eigsh(-L,k=500, which="SM")  # récupération des éléments propres

Vaf = Va[Va > 1e-10]    # tri des valeurs propres, on ne garde que les nons nulles
Vef = Ve.T[Va > 1e-10]  # tri des vecteurs propres associés

fig, axes = plt.subplots(nrows=num_shown, ncols=1)
idx = 0
for ax in axes.flat:
    v = np.zeros((w//10,w))
    v[:,] = Vef[idx]
    im = ax.imshow(v, cmap="bwr", vmin=-0.1, vmax=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    m = int(1000*sqrt(Vaf[idx]/Vaf[0]))/1000
    ax.set_ylabel(f"μ_{idx} = {m}", rotation=0, ha="right")
    idx = idx + 1
fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig("Dirichlet_1D.png")

## Conditions de Neumann

w = 601
B = sp.eye(w-1,w,1) - sp.eye(w-1,w)
N = (w-1)//5

W = sp.diags([0]*N + [1]*3*N + [0]*N)  # matrice des pondérations d'arêtes de taille (w-1,w-1)

L = - B.T @ W @ B   # matrice du laplacien

Va,Ve = eigsh(-L,k=500, which="SM")
Vaf = Va[Va > 1e-10]
Vef = Ve.T[Va > 1e-10]

fig, axes = plt.subplots(nrows=num_shown, ncols=1)
idx = 0
for ax in axes.flat:
    v = np.zeros((w//10,w))
    v[:,] = Vef[idx]
    im = ax.imshow(v, cmap="bwr", vmin=-0.1, vmax=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    m = int(1000*sqrt(Vaf[idx]/Vaf[0]))/1000
    ax.set_ylabel(f"μ_{idx} = {m}", rotation=0, ha="right")
    idx = idx + 1
fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig("Neumann_1D.png")

## Conditions de Dirichlet à gauche, Neumann à droite

w = 600
B = sp.eye(w-1,w,1) - sp.eye(w-1,w)

N = w//5
A = sp.diags([0]*N + [1]*3*N + [0]*N)  # matrice des inerties
BM = B @ M

W = sp.diags([1]*(4*N-1) + [0]*N)  # matrice des rigidités : les noeuds à gauche de la barre sont indépendants

L = -BM.T @ W @ BM

Va,Ve = eigsh(-L,k=500, which="SM")
Vaf = Va[Va > 1e-10]
Vef = Ve.T[Va > 1e-10]

fig, axes = plt.subplots(nrows=num_shown, ncols=1)
idx = 0
for ax in axes.flat:
    v = np.zeros((w//10,w))
    v[:,] = Vef[idx]
    im = ax.imshow(v, cmap="bwr", vmin=-0.1, vmax=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    m = int(1000*sqrt(Vaf[idx]/Vaf[0]))/1000
    ax.set_ylabel(f"μ_{idx} = {m}", rotation=0, ha="right")
    idx = idx + 1
fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig("Mi_Diri_Mi_Neum_1D.png")











