import numpy as np
import numpy.linalg
import torch
import numpy as np
import scipy.sparse
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from math import sqrt

## Récupération des données

donnees_init = np.load(f"donnees_init.npy").tolist()
l_init = torch.from_numpy(np.load(f"param_-1.npy"))
w = int(donnees_init[0])
iter_num = donnees_init[1]
Longueur= donnees_init[2]
i = int(iter_num-1)
I = np.load(f"iteration_num_{i}.npy")
l = torch.from_numpy(np.load(f"param_{i}.npy"))
loss_tab = np.load(f"loss_tab_num_{i}.npy")
Partiel1 = np.sqrt(np.load(f"partiel1_num_{i}.npy"))
Partiel2 = np.sqrt(np.load(f"partiel2_num_{i}.npy"))
fondamental = np.sqrt(np.load(f"fondamental_num_{i}.npy"))

## Calcul des images finales et initiales
a = w/2
init_table = a*torch.ones((w,w))
y = w//2
for i in range(w):
    for j in range(w):
        if abs(j-y)<Longueur:
            init_table[i,j]=abs(i-y)

sigma = lambda x : 1/(1+torch.exp(-x))

rect = lambda r : sigma(r-init_table)

im = rect(l)
im_init = rect(l_init)

## Affichage des résultats généraux

plt.clf()

fig = plt.figure(figsize=(16,9))

ax1 = fig.add_subplot(3,2,2)

plt.set_cmap("jet")
forme_finale = ax1.imshow(im, cmap = 'jet')
fig.colorbar(forme_finale,ax=ax1)
val=l.numpy()
ax1.set_title(f"Forme finale l = {val}")

ax2 = fig.add_subplot(3,2,3)
ax2.plot(I,loss_tab,label='Coût fonction itération')
ax2.plot(I,[0]*len(I),label='Limite attendue')

plt.legend()

ax3 = fig.add_subplot(3,2,4)
ax3.plot(I,Partiel1,marker='o',label='Partiel 1 fonction itération')
ax3.plot(I,Partiel2,'r',label='Partiel 2 fonction itération')
ax3.plot(I,[2]*len(I))
ax3.plot(I,[3]*len(I))

plt.legend()

ax4 = fig.add_subplot(3,2,1)
forme_initiale = ax4.imshow(im_init, cmap = 'jet')
fig.colorbar(forme_initiale,ax=ax4)
val=l_init.numpy()
ax4.set_title(f"Forme initiale l = {val}")

ax5 = fig.add_subplot(3,2,5)
ax5.plot(I,fondamental,label='Fondamental fonction itération')

plt.legend()

plt.savefig("resultat_experience.png")