import numpy as np
import numpy.linalg
import torch
from scipy.sparse import eye, kron, vstack, diags
from scipy.sparse.linalg import eigsh

# Données initiales
n = 10
m = 20
iter_num = 101
eps_reg = 8.25

numpy.save(f"donnees_init.npy",np.array([n,m,iter_num,eps_reg]))

# Création de la matrice d'incidence
x = eye(n-1,n,1) - eye(n-1,n)
y = eye(m-1,m,1) - eye(m-1,m)
B = vstack([kron(eye(m),x),kron(y,eye(n))])
B = torch.from_numpy(B.toarray())


# Fonction de coût
def loss_fun(dM) :
  BM = B @ torch.diag(dM**2)
  L = - BM.t() @ BM
  U,S,V = torch.svd(-L) # récupération des éléments propres
  S_f = S[S>1e-5]
  return (4 - S_f[-2]/S_f[-1])**2 + (9 - S_f[-3]/S_f[-1])**2 + (16 - S_f[-4]/S_f[-1])**2 + eps_reg * (B @ dM).t() @ (B @ dM)

# Fonction de recherche du pas
def backtracking_line_search(y,p):
    alpha = 1/4
    beta = 1/2
    t=1
    while loss_fun(y-t*p) > loss_fun(y) - alpha * t * p.norm() :
        t = beta * t
    return t

# Initialisation de la descente de gradient
eps = 0.05
dM = np.ones(n*m) - eps * np.random.rand(n*m)
numpy.save(f"forme_num_-1.npy",dM)
dM = torch.from_numpy(dM)
dM.requires_grad = True
I = []
loss_tab = []
Partiel1 = []
Partiel2 = []
Partiel3 = []
fondamental = []

# Descente
for i in range(iter_num):
    loss = loss_fun(dM) # forward pass
    loss.backward() # backward pass
    grad_dM = dM.grad
    learning_rate = backtracking_line_search(dM,grad_dM)
    with torch.no_grad():
        dM -= grad_dM * learning_rate
        dM.grad.zero_()

    #Données pour l'affichage de la convergence
    BM = B @ torch.diag(dM**2)
    L = - BM.t() @ BM
    U,S,V = torch.svd(-L)
    S_f = S[S>1e-5].detach().numpy().tolist()
    Partiel1.append(S_f[-2]/S_f[-1])
    Partiel2.append(S_f[-3]/S_f[-1])
    Partiel3.append(S_f[-4]/S_f[-1])
    loss_tab.append(loss.detach().numpy().tolist())
    fondamental.append(S_f[-1])
    I.append(i)
    if i%10 == 0:
        numpy.save(f"iteration_num_{i}.npy", np.array(I))
        numpy.save(f"loss_tab_num_{i}.npy", np.array(loss_tab))
        numpy.save(f"partiel1_num_{i}.npy", np.array(Partiel1))
        numpy.save(f"partiel2_num_{i}.npy", np.array(Partiel2))
        numpy.save(f"partiel3_num_{i}.npy", np.array(Partiel3))
        numpy.save(f"forme_num_{i}.npy", dM.detach().numpy())
        numpy.save(f"fondamental_num_{i}.npy", np.array(fondamental))