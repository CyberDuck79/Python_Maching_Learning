from random import randrange


# 2D vector multiplication function
# x : input vector
# y : output vector


def f(x, w):
    #y = (x[0] * w[0], x[1] * w[1])
    # or power function ?
    # y = (x[0] ** w[0], x[1] ** w[1])
    y = w[0] + (w[1] * x)
    return y

# cost computing function
# x : input vector
# y : desired output vector
# w : parameter vector
# f() : output computing function


def cost(x, y, w):
    yp = f(x, w)
    c = (y - yp) ** 2
    #c = 0
    #for i in range(len(y)):
    #    c += (y[i] - yp[i]) ** 2
    return c

# cost computing function for a set
# X : array of input vector of the set
# Y : array desired output vector of the set
# w : parameter vector


def set_cost(X, Y, w):
    p = len(X)
    s = 0
    for i in range(p):
        s += cost(X[i], Y[i], w)
    return s / p

# perturbation gradient computing function
# X : array of input vector of the set
# Y : array desired output vector of the set
# w : parameter vector
# dw : perturbation


def gradient(x, y, w):
    g = [0, 0]
    g[0] = -2 * (y - f(x, w))
    g[1] = -2 * (y - f(x, w)) * x
    return g


# learning function
# X : array of input vector of the set
# Y : array desired output vector of the set
# w : parameter vector
# e : gradient step
# dw : perturbation
# n : max number of gradient descend steps
# last : stop the function if cost stop to change


def learn(X, Y, w, e, n):
    p = len(X)
    for i in range(n):
        k = randrange(0,p)
        g = gradient(X[k], Y[k], w)
        for j in range(len(w)):
            w[j] = w[j] - e * g[j]
    return w
