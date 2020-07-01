#'''
#Recherche d'un ellipse telle que la deuxième partielle vaille 2
#'''
#
from numpy import linspace, meshgrid, vectorize
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from math import sqrt

# fonction indicatrice d'une ellipse d'exentricité e, de demi-grand axe a
def ellipse(e,x,y) :       
    a = 0.8
    b = a *  sqrt(1-e*e)
    if (x/a)**2 + (y/b)**2 < 1 :
        return 1
    else :
        return 0

# fonction ellipse vectorialisée
ellipse = vectorize(ellipse)     


# création d'une image de disque
def image_of_a_ellipse(e,n) :  
    x,y = meshgrid(linspace(-1,1,n), linspace(-1,1,n))
    return ellipse(e,x,y)

def grid_incidence(h, w):
	x = eye(w - 1, w, 1) - eye(w  -1, w)             
	y = eye(h - 1, h, 1) - eye(h - 1, h)             
	B = vstack([ kron(eye(h),x) , kron(y,eye(w)) ])  
	return B

w = 64 

B = grid_incidence(w, w)

E = linspace(0,1,10)
V1 = []
P2 = []
V2 = []

for e in E[:-1] :
    M = image_of_a_ellipse(e,w)
    BM = B @ diags(M.flatten())
    L = - BM.T @ BM
    D,U = eigsh(-L, k=100, which="SM") 
    Df = D[D > 1e-10]
    Uf = U.T[D > 1e-10]
    p = sqrt(Df[1]/Df[0])
    v1 = Uf[0]
    v2 = Uf[1]
    P2.append(p)
    V1.append(v1)
    V2.append(v2)

plt.set_cmap("bwr")  # blue-white-red palette
    
idx = 0

fig , axes = plt.subplots(nrows=9, ncols=2)

for ax in axes.flat:
    if idx % 2 == 0 :
        im = ax.imshow(V1[idx//2].reshape(w,w), cmap='bwr', vmin=-0.1, vmax=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
    else :
        im = ax.imshow(V2[(idx-1)//2].reshape(w,w), cmap = 'bwr', vmin = -0.1, vmax = 0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"μ_{idx} = {P2[(idx-1)//2]}", fontsize = 8)
    idx +=1
    
    

plt.gcf().subplots_adjust(wspace = 0, hspace = 1.5)
plt.savefig("ellipse_dirichlet.png")

plt.figure(2)
plt.clf()

plt.plot(E[:-1],P2, 'r+')
plt.

plt.savefig("partielle_2_exentricite.png")
    


    
    


