import numpy as np
import scipy
#import pylab as plt
from scipy.linalg import norm, eig
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint
from scipy.linalg import solve
#from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import matplotlib

plt.clf()
plt.ion()

##Données:
N=500
alpha=0
beta=1
gamma=0

##Construction matrices
B=np.eye(N-1,N,1)-np.eye(N-1,N)
W=np.zeros((N-1,N-1))
for j in range(0,(N-1)//3):
    W[j,j]=alpha
for j in range((N-1)//3,2*(N-1)//3):
    W[j,j]=beta
for j in range(2*(N-1)//3,N-1):
    W[j,j]=gamma
L=-np.dot(np.dot(np.transpose(B),W),B)

##Calculs éléments propres
(D,U)=eigs(-L,k=500,which='SM')
D_ind_full = D.argsort()
D_sorted = np.sort(D)
D_ind_filt = D_ind_full[D_sorted>1e-30]
D_filt = D_sorted[D_sorted>1e-30]


##Tracés
plt.subplot(911)
v1=U[:,D_ind_filt[0]]
v=np.zeros((N//10,N))
v[:,]=255*(v1-min(v1))/(max(v1)-min(v1))
plt.imshow(v)

plt.subplot(912)
v1=U[:,D_ind_filt[1]]
v=np.zeros((N//10,N))
v[:,]=255*(v1-min(v1))/(max(v1)-min(v1))
plt.imshow(v)

plt.subplot(913)
v1=U[:,D_ind_filt[2]]
v=np.zeros((N//10,N))
v[:,]=255*(v1-min(v1))/(max(v1)-min(v1))
plt.imshow(v)

plt.subplot(914)
v1=U[:,D_ind_filt[3]]
v=np.zeros((N//10,N))
v[:,]=255*(v1-min(v1))/(max(v1)-min(v1))
plt.imshow(v)

plt.subplot(915)
v1=U[:,D_ind_filt[4]]
v=np.zeros((N//10,N))
v[:,]=255*(v1-min(v1))/(max(v1)-min(v1))
plt.imshow(v)

plt.show()
