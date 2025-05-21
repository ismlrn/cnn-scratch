#%%
import numpy as np

# random input
X = np.array([[1, 2, 0],
              [3, 4, 1],
              [1, 2, 0]])

# random weights
W = np.array([[1, -1],
              [0, 1]])

# all ones target for MSE loss
target = np.ones((2, 2))  # MSE target

#%%
# convolution layer
def conv2d(X, W):
    out = np.zeros((X.shape[0] - W.shape[0] + 1,
                    X.shape[1] - W.shape[1] + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = X[i:i+W.shape[0], j:j+W.shape[1]]
            out[i, j] = np.sum(region * W)
    return out

def relu(X):
    return np.maximum(0, X)

#%%
# loss function
def mse_loss(pred, target):
    return np.mean((pred - target)**2)
#%%
Z = conv2d(X, W)           # (2, 2)
A = relu(Z)                
loss = mse_loss(A, target)

print("Forward Output (A):\n", A)
print("Loss:", loss)


# %%
# backward pass
dA = 2 * (A - target) / target.size
dZ = dA * (Z > 0).astype(float)

def conv2d_backward(X, dZ, W_shape):
    dW = np.zeros(W_shape)
    for i in range(dZ.shape[0]):
        for j in range(dZ.shape[1]):
            region = X[i:i+W_shape[0], j:j+W_shape[1]]
            dW += dZ[i, j] * region
    return dW

dW = conv2d_backward(X, dZ, W.shape)
print("Gradient dW:\n", dW)

# %%
