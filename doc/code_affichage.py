import numpy as np
import numpy.linalg
import torch
import numpy as np
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from math import sqrt

plt.clf()

n = 20
m = 20
i=100

im = np.load(f"forme_num_{i}.npy")**2

fig = plt.figure(figsize=(16,9))

ax1 = fig.add_subplot(3,2,1)

plt.set_cmap("bwr")
ima = ax1.imshow(im.reshape(n,m), cmap = 'bwr',vmin = -1,vmax = 1)
fig.colorbar(ima,ax=ax1)

I = np.load(f"iteration_num_{i}.npy")
loss_tab = np.load(f"loss_tab_num_{i}.npy")
Partiel1 = np.load(f"Partiel1_num_{i}.npy")
Partiel2 = np.load(f"Partiel2_num_{i}.npy")
Partiel3 = np.load(f"Partiel3_num_{i}.npy")
fondamental = np.load(f"fondamental_num_{i}.npy")

ax2 = fig.add_subplot(3,2,2)
ax2.plot(I,loss_tab,label='Coût fonction itération')
ax2.plot(I,[0]*len(I),label='Limite attendue')

plt.legend()

ax3 = fig.add_subplot(3,2,3)
ax3.plot(I,Partiel1,label='Partiel 1 fonction itération')
ax3.plot(I,[4]*len(I),label='Limite attendue')

plt.legend()

ax4 = fig.add_subplot(3,2,4)
ax4.plot(I,Partiel2,label='Partiel 2 fonction itération')
ax4.plot(I,[9]*len(I),label='Limite attendue')

plt.legend()

ax5 = fig.add_subplot(3,2,5)
ax5.plot(I,Partiel3,label='Partiel 3 fonction itération')
ax5.plot(I,[16]*len(I),label='Limite attendue')

plt.legend()

ax6 = fig.add_subplot(3,2,6)
ax6.plot(I,fondamental,label='Fondamental fonction itération')

plt.legend()

plt.savefig("resultat_experience.png")