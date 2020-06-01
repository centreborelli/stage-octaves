"""
Conditions de Dirichlet, Neumann et Mi-Dirichlet, Mi-Neumann en 1D
"""

import numpy as np
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt


num_shown = 9

# Dirichlet

w = 600
B = sp.eye(w-1,w,1) - sp.eye(w-1,w)
N = w//5


M = sp.diags([0]*N + [1]*3*N + [0]*N)

BM = B @ M

L = - BM.T @ BM

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
plt.savefig("Dirichlet_1D.png")

# Neuman

w = 601
B = sp.eye(w-1,w,1) - sp.eye(w-1,w)
N = (w-1)//5

W = sp.diags([0]*N + [1]*3*N + [0]*N)

L = - B.T @ W @ B

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

# Dirichlet à gauche, Neumann à droite

w = 600

B = sp.eye(w-1,w,1) - sp.eye(w-1,w)

N = w//5
M = sp.diags([0]*N + [1]*3*N + [0]*N)

BM = B @ M

W = sp.diags([1]*(4*N-1) + [0]*N)

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











