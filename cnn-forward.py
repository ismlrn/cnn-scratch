#%%
import numpy as np

# for simplicity, i'll use a 1x1x8x8 input image only
# Batch size = 1 Channels = 1 Height = 8 Width = 8

#%%
np.random.seed(42)
input_image = np.random.rand(1, 1, 8, 8)

print("Input shape:", input_image.shape)

# %%
# adding convolution layer with no padding and stride = 1
def conv2d(X, kernel, stride=1):
    B, C, H, W = X.shape
    F, _, KH, KW = kernel.shape
    out_h = (H - KH) // stride + 1
    out_w = (W - KW) // stride + 1

    out = np.zeros((B, F, out_h, out_w))

    for b in range(B):
        for f in range(F):
            for i in range(out_h):
                for j in range(out_w):
                    region = X[b, 0, i:i+KH, j:j+KW]
                    out[b, f, i, j] = np.sum(region * kernel[f, 0])
    return out



# %%
# creating a 3*3 filter with random values
# kernel = np.array([[[[1, 0, -1],
#                      [1, 0, -1],
#                      [1, 0, -1]]]])  # shape (1, 1, 3, 3)

kernel = np.array([
    [[[1, 0, -1],
      [1, 0, -1],
      [1, 0, -1]]],  # filter 1 (edge detection)

    [[[0, 1, 0],
      [0, 1, 0],
      [0, 1, 0]]]   # filter 2 (vertical focus)
])  # shape (2, 1, 3, 3)


conv_out = conv2d(input_image, kernel)
# the output shape is (1, 1, 6, 6)
print("Conv output shape:", conv_out.shape)

# %%
# applying relu activation
def relu(X):
    return np.maximum(0, X)

relu_out = relu(conv_out)

# %%
# applying maxpooling with a size of 2 and stride = 2
def maxpool2d(X, size=2, stride=2):
    B, C, H, W = X.shape
    out_h = H // size
    out_w = W // size
    out = np.zeros((B, C, out_h, out_w))

    for b in range(B):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    region = X[b, c, i*stride:i*stride+size, j*stride:j*stride+size]
                    out[b, c, i, j] = np.max(region)
    return out

pool_out = maxpool2d(relu_out)
# the output shape is (1, 1, 3, 3)
print("Pooled shape:", pool_out.shape)

# %%
# flattening the output
def flatten(X):
    return X.reshape(X.shape[0], -1)

flat = flatten(pool_out)
# the output shape is (1, 9)
print("Flattened:", flat.shape)

# %%
# adding a dense layer with random weights and bias
def dense(X, weights, bias):
    return X @ weights + bias

W = np.random.randn(flat.shape[1], 2)  # output size = 2
b = np.random.randn(2)

fc_out = dense(flat, W, b)

# %%
# adding softmax activation for classification
def softmax(X):
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

probs = softmax(fc_out)
# the output shape is (1, 2) obviously
print("Probabilities:", probs)

# %%
