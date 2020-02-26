from random import randint
from random import randrange
from matplotlib import pyplot as plt

# line function
def f(x, w):
	y = w[0] + (w[1] * x)
	return y

# cost function
def cost(x, y, w):
	yp = f(x, w)
	return (y - yp) ** 2

# gradient function
def gradient(x, y, w):
	g = [0, 0]
	g[0] = -2 * (y - f(x, w))
	g[1] = -2 * (y - f(x, w)) * x
	return g

# learning function
def learn(X, Y, w, e, n):
	p = len(X)
	for i in range(n):
		k = randrange(0,p)
		g = gradient(X[k], Y[k], w)
		for j in range(len(w)):
			w[j] = w[j] - e * g[j]
	return w

# input/ouput vectors normalization for training
def normalize_vector(V):
	Vn = []
	Vmin = min(V)
	Vmax = max(V)
	for i in range(len(X)):
		Vn.append((V[i] - Vmin) / (Vmax - Vmin))
	return Vn

# denormalization of the weights for recognition
def denormalize_weights(X, Y, w):
	Ymin = min(Y)
	Ymax = max(Y)
	Xmin = min(X)
	Xmax = max(X)
	w[1] *= (Ymax - Ymin) / (Xmax - Xmin)
	w[0] *= (Ymax - Ymin) + Ymin - w[1] * Xmin
	return w

# data reading
X = []
Y = []
with open('data.csv') as file:
	for line in file.readlines()[1:]:
		data = line.split(sep=',')
		X.append(int(data[0]))
		Y.append(int(data[1].strip('\n')))
# data normalization
Xn = normalize_vector(X)
Yn = normalize_vector(Y)
# learning process
e = 0.001
n = 100000
w = [0, 0]
w = learn(Xn, Yn, w, e, n)
# denormalization of weights
w = denormalize_weights(X, Y, w)
# calculation of the line
Yp = []
for x in X:
	Yp.append(f(x, w))
# drawing
plt.plot(X,Y,'o')
plt.plot(X,Yp)
plt.xlabel("kilometers")
plt.ylabel("price")
plt.title('ft_linear_regression')
plt.show()
# error ratio
error = 0
size = len(X)
for i in range(size):
	error += abs(Y[i] - Yp[i])
print(error/size)