from numpy import linspace, meshgrid, vectorize
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from math import sqrt

def disk(r,x,y) :           # fonction indicatrice d'un disque de rayon r, évaluée en (x,y)
    if x**2 + y**2 < r**2 :
        return 1
    else :
        return 0
    
disk = vectorize(disk)      # fonction vectorialisée


def image_of_a_disk(r,n) :  # création d'une image de disque
    x,y = meshgrid(linspace(-1,1,n), linspace(-1,1,n))
    return disk(r,x,y)

def grid_incidence(h, w):
	x = eye(w - 1, w, 1) - eye(w  -1, w)             # path of length W
	y = eye(h - 1, h, 1) - eye(h - 1, h)             # path of length H
	B = vstack([ kron(eye(h),x) , kron(y,eye(w)) ])  # kronecker sum
	return B

w = 64 # image de taille wxw


M = image_of_a_disk(0.7,w)

B = grid_incidence(w, w)    
BM = B @ diags(M.flatten()) # conditions de Dirichlet : on impose 0 à l'extérieur du disque
L = -BM.T @ BM          # laplavien du domaine M

D,U = eigsh(-L, k=100, which="SM") 

Df = D[D > 1e-10]       # selection des valeurs propres non nulles
Uf = U.T[D > 1e-10]     # selection des vectuers propres associes

# affichage 
    
plt.set_cmap("bwr")  # blue-white-red palette
    
num_shown = 25

vv = [ Uf[i] for i in range(0,num_shown) ]

fig, axes = plt.subplots(nrows=5, ncols=5)  # a modifier si l'on modifie num_shown

idx = 0

for ax in axes.flat:
    v = vv[idx]
    im = ax.imshow(v.reshape(w,w), cmap='bwr', vmin=-0.1, vmax=0.1)
    m = int(1000*sqrt(Df[idx]/Df[0]))/1000
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"μ_{idx} = {m}", fontsize = 8)
    idx +=1

plt.gcf().subplots_adjust(wspace = 0, hspace = 0.7)
plt.savefig("peau_ronde_dirichlet.png")