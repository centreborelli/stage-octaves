
# characteristic function of a bar centered at the origin
def marimba_bar(
		a, b,  # width, height of the rectangle
		d,     # vertical offset of the disk center
		r,     # radius of the disk
		x, y   # point where the function is evaluated
	):
	from numpy import fabs, hypot
	return (fabs(x)<a) * (fabs(y)<b) * (hypot(x,y-d)>r)


# characteristic function of a tilted and rotated bar
def rotated_marimba_bar(
		a, b, d, r, x, y,  # same parameters as above
		α,                 # tilt coefficient (1=isotropic)
		θ                  # rotation angle
	):
	from numpy import sin, cos
	X = cos(θ) * x - sin(θ) * y
	Y = sin(θ) * x + cos(θ) * y
	return marimba_bar(a,b, d,r, X, α * Y)


# characteristic function of a tilted and rotated bar (discrete image)
def image_of_a_rotated_marimba_bar(
		a, b, d, r, α, θ,  # shape parameters as above
		n                  # number of samples along each dimension
	):
	from numpy import linspace, meshgrid
	x,y = meshgrid(linspace(-1,1,n), linspace(-1,1,n))
	return rotated_marimba_bar(a,b, d,r, x, y, α, θ)


# build the incidence matrix of a grid graph
def grid_incidence(h, w):
	from scipy.sparse import eye, kron, vstack
	x = eye(w - 1, w, 1) - eye(w  -1, w)             # path of length W
	y = eye(h - 1, h, 1) - eye(h - 1, h)             # path of length H
	B = vstack([ kron(eye(h),x) , kron(y,eye(w)) ])  # kronecker sum
	return B



#########################
## try the codes above ##
#########################

# import libraries
import numpy
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# side of the whole image domain (a square)
w = 64

# build the shape of the region of interest
M = image_of_a_rotated_marimba_bar(.9,.4, .5,.4, 1, 3.1416/3,  w)

# save a black and white image representing the region of interest
numpy.save("bar.npy", 255.0*M)

# build the matrices associated to the domain "M"
B = grid_incidence(w, w)    # incidence matrix
C = abs(B)/2                # centering matrix
W = diags(C @ M.flatten())  # weight matrix
L = -B.T @ W @ B            # laplacian of the domain M

# compute the eigensystem of the laplacian operator
D,U = eigsh(-L, k=100, which="SM")   # XXX: the k parameter here is delicate
#print(D)

# select the non-zero eigenvalues and eigenvectors
Df = D[D > 1e-10]
Uf = U.T[D > 1e-10]

# save the first 20 eigenvectors
for i in range(20):
	numpy.save(f"v_{i:02d}.npy", Uf[i].reshape(w,w))

# print the partials
μ = [ numpy.sqrt(Df[i] / Df[0]) for i in range(len(Df)) ]
for i in range(20):
	print(f"μ_{i} = {μ[i]}")


# display the first eigenfunctions using a signed palette
import matplotlib.pyplot
matplotlib.pyplot.set_cmap("bwr")  # blue-white-red palette
for i in range(20):
	φ = Uf[i].reshape(w,w)
	matplotlib.pyplot.imshow(φ, vmin=-0.05, vmax=0.05)
	matplotlib.pyplot.title(f"μ_{i} = {μ[i]}")
	matplotlib.pyplot.colorbar()
	matplotlib.pyplot.savefig(f"v_{i:02d}.png")
	matplotlib.pyplot.clf()
