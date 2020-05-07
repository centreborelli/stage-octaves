# juste une version de "code_1D.py" qui sauvegarde les résultats dans des
# fichiers, sans rien afficher (elle peut tourner dans un serveur sans écran)

from scipy.sparse import eye, diags
from scipy.linalg import eigh


##Données:
N = 502
alpha = 0
beta = 1
gamma = 0

##Construction matrices
B = eye(N-1,N,1) - eye(N-1,N)
N3 = (N-1)//3
W = diags([alpha]*N3 + [beta]*N3 + [gamma]*N3)
L = -B.T @ W @ B

##Calculs éléments propres
#D,U = eigsh(-L, k=500, which='SM')
D,U = eigh(-L.todense())
U_filt = U.T[D > 1e-30]
D_filt = D[D > 1e-30]

##Tracés
num_shown = 8

vv = [ U_filt[i] for i in range(0,num_shown) ]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=num_shown, ncols=1)

idx = 0
for ax in axes.flat:
	from numpy import zeros
	from math import sqrt
	v = zeros((N//10,N))
	v[:,] = vv[idx]
	im = ax.imshow(v, cmap="bwr", vmin=-0.1, vmax=0.1)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_ylabel(f"φ_{idx}", rotation=0, ha="right")
	#print(f"λ_{str(idx)} = {N*D_filt[idx]}")
	print(f"μ_{str(idx)} = {sqrt(D_filt[idx]/D_filt[1])}")
	idx = idx + 1

fig.colorbar(im, ax=axes.ravel().tolist())
plt.savefig("first_partials_1d.png")
