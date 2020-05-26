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

x = torch.randn(2, 1, requires_grad=True)

# Comment choisir précisément le taux d'apprentissage ?

learning_rate = 1e-3

# Itération

iter_num = 500

for i in range(iter_num):
    loss = (1/2) * ( S @ x ).t() @ ( S @ x ) + b.t() @ x
    loss.backward()
    with torch.no_grad():
        x -= x.grad * learning_rate
        x.grad.zero_()

# Résultat

print(x)

## Solution exacte x=-Inv(Transpose(S)S)b

H = (1/195) * np.array( [ [14 , -1] , [-1 , 14] ] , dtype='float32') # Inv(Transpose(S)S)
H = torch.from_numpy(H)
x_ex = - H @ b
print(x_ex)

