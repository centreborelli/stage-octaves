import numpy as np
import numpy.linalg
import torch
import numpy as np
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from math import sqrt

## Récupération des données

donnees_init = np.load(f"donnees_init.npy").tolist()
im_init = np.load(f"forme_num_-1.npy")**2
n = int(donnees_init[0])
m = int(donnees_init[1])
iter_num = donnees_init[2]
i = int(iter_num-1)
I = np.load(f"iteration_num_{i}.npy")
im = np.load(f"forme_num_{i}.npy")**2
loss_tab = np.load(f"loss_tab_num_{i}.npy")
Partiel1 = np.sqrt(np.load(f"Partiel1_num_{i}.npy"))
Partiel2 = np.sqrt(np.load(f"Partiel2_num_{i}.npy"))
Partiel3 = np.sqrt(np.load(f"Partiel3_num_{i}.npy"))
fondamental = np.sqrt(np.load(f"fondamental_num_{i}.npy"))


## Enregistrement des résultats généraux

plt.clf()

fig = plt.figure(figsize=(16,9))

ax1 = fig.add_subplot(3,2,2)

plt.set_cmap("jet")
forme_finale = ax1.imshow(im.reshape(n,m), cmap = 'jet')
fig.colorbar(forme_finale,ax=ax1)

plt.title(label='Forme finale')

ax2 = fig.add_subplot(3,2,3)
ax2.plot(I,loss_tab,label='Coût fonction itération')
ax2.plot(I,[0]*len(I),label='Limite attendue')

plt.legend()

ax3 = fig.add_subplot(3,2,4)
ax3.plot(I,Partiel1,label='Partiel 1 fonction itération')
ax3.plot(I,Partiel2,'r',label='Partiel 2 fonction itération')
ax3 .plot(I,Partiel3,'g',label='Partiel 3 fonction itération')

plt.legend()

ax4 = fig.add_subplot(3,2,1)
forme_initiale = ax4.imshow(im_init.reshape(n,m), cmap = 'jet')
fig.colorbar(forme_initiale,ax=ax4)

plt.title(label='Forme initiale')

ax5 = fig.add_subplot(3,2,5)
ax5.plot(I,fondamental,label='Fondamental fonction itération')

plt.legend()

plt.savefig("resultat_experience.png")

## Enregistrement des éléments propres finaux

x = eye(n-1,n,1) - eye(n-1,n)
y = eye(m-1,m,1) - eye(m-1,m)
B = vstack([kron(eye(m),x),kron(y,eye(n))])
BM = B @ diags(im)
L = -BM.T @ BM

D,U = eigsh(-L, k=30, which="SM")

Df = D[D > 1e-5]
Uf = U.T[D > 1e-5]

plt.set_cmap("bwr")

vv = [ Uf[i] for i in range(0,len(Uf)) ]

a = int(np.sqrt(len(Uf)))

fig, axes = plt.subplots(nrows = a , ncols = a)

idx = 0

for ax in axes.flat:
    v = vv[idx]
    im = ax.imshow(v.reshape(n,m), cmap='bwr', vmin=-0.1, vmax=0.1)
    val = int(1000*sqrt(Df[idx]/Df[0]))/1000
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"μ_{idx} = {val}", fontsize = 8)
    idx +=1

plt.gcf().subplots_adjust(wspace = 0, hspace = 0.7)
plt.savefig("fonctions_propres_finales.png")

'''## Enregistrement des éléments propres initiaux

BM = B @ diags(im_init)
L = -BM.T @ BM

D,U = eigsh(-L, k=30, which="SM")

Df = D[D > 1e-5]
Uf = U.T[D > 1e-5]

plt.set_cmap("bwr")

num_shown = 3

vv = [ Uf[i] for i in range(0,num_shown) ]

fig, axes = plt.subplots(nrows=1, ncols=3)

idx = 0

for ax in axes.flat:
    v = vv[idx]
    im = ax.imshow(v.reshape(n,m), cmap='bwr', vmin=-0.1, vmax=0.1)
    val = int(1000*sqrt(Df[idx]/Df[0]))/1000
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"μ_{idx} = {val}", fontsize = 8)
    idx +=1

plt.gcf().subplots_adjust(wspace = 0, hspace = 0.7)
plt.savefig("fonctions_propres_intitiales.png")'''