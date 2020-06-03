import numpy as np
import numpy.linalg
import torch


''' Les arguments sont:
- v : une prédiction de vecteur propre, à voir comment on choisit.
- maxIters : le maximum des itérations pour le calcul.
- tolerance : la tolérance pour le calcul'''

def PowerIteration(v, A, maxIters, tolerance):
   v = v / v.norm()
   iteration = 0
   lambdaOld = 0
   while iteration <= maxIters:
    z = A @ v
    v = z / z.norm()
    lamb = np.transpose(v) @ z
    iteration = iteration + 1
    if abs((lamb - lambdaOld)/lamb) < tolerance:
        return lamb, v
    lambdaOld = lamb
   if abs((lamb - lambdaOld)/lamb) >= tolerance:
    print('No convergence')


A = np.array([[-261, 209, -49],
    [-530, 422, -98],
    [-800, 631, -144]], dtype='float32')

A = torch.from_numpy(A)

v = np.array([1,2,3], dtype='float32')

v = torch.from_numpy(v)

lamb, v = PowerIteration(v, A, 40, 10e-6)

print(A@v)
print(lamb*v)