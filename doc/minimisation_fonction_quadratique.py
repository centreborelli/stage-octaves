import numpy as np
import torch

## Données

S = np.array( [ [3 , -1] , [-1 , 2] , [2 , 3] ] , dtype='float32')
b = np.array( [ [5] , [6] ] , dtype='float32')

# Conversion dans le "langage" pyTorch

S = torch.from_numpy(S)
b = torch.from_numpy(b)


## Solution par descente de gradient

# Initalisation de x

x = torch.randn(b.size(), requires_grad=True)

# Calcul de la fonction cout

def loss_fun(x): return (1/2) * ( S @ x ).t() @ ( S @ x ) + b.t() @ x

# Recherche du learning rate

def backtracking_line_search(y,p):
    alpha = 1/4
    beta = 1/2
    t=1
    while loss_fun(x-t*p) > loss_fun(x) - alpha * t * p.norm() :
        t = beta * t
    return t

# Itération

iter_num = 500

for i in range(iter_num):
    loss = loss_fun(x)
    loss.backward()
    grad_x = x.grad
    learning_rate = backtracking_line_search(x,grad_x)
    with torch.no_grad():
        x -= grad_x * learning_rate
        x.grad.zero_()

# Résultat

print('argmin_exp = ',x)
print('min_exp = ',loss_fun(x))

## Solution exacte x=-Inv(Transpose(S)S)b

H = (1/195) * np.array( [ [14 , -1] , [-1 , 14] ] , dtype='float32') # Inv(Transpose(S)S)
H = torch.from_numpy(H)
x_ex = - H @ b
print('argmin_th = ',x_ex)
print('min_th = ',loss_fun(x_ex))
