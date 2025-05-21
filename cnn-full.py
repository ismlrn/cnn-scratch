#%%
import numpy as np

#%%
# implementation of full CNN forward and backward pass using numpy with class
# -----------------------
# Convolution Layer (forward and backward) 
# -----------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        limit = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weights = np.random.uniform(-limit, limit,
                                         (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros((out_channels, 1))
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        batch_size, C, H, W = x.shape
        F, _, kH, kW = self.weights.shape

        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.x_padded = x_padded

        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1

        out = np.zeros((batch_size, F, out_H, out_W))

        for b in range(batch_size):
            for f in range(F):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = x_padded[b, :, h_start:h_start + kH, w_start:w_start + kW]
                        out[b, f, i, j] = np.sum(region * self.weights[f]) + self.bias[f]
        return out

    def backward(self, dout):
        batch_size, C, H, W = self.x.shape
        F, _, kH, kW = self.weights.shape
        _, _, out_H, out_W = dout.shape

        dx = np.zeros_like(self.x_padded)
        self.dw.fill(0)
        self.db.fill(0)

        for b in range(batch_size):
            for f in range(F):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = self.x_padded[b, :, h_start:h_start + kH, w_start:w_start + kW]

                        self.dw[f] += dout[b, f, i, j] * region
                        self.db[f] += dout[b, f, i, j]
                        dx[b, :, h_start:h_start + kH, w_start:w_start + kW] += dout[b, f, i, j] * self.weights[f]

        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx

# %%
# -----------------------
# ReLU Activation
# -----------------------
class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

# -----------------------
# Max Pooling Layer
# -----------------------
class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        self.x = x
        batch_size, C, H, W = x.shape
        out_H = (H - self.size) // self.stride + 1
        out_W = (W - self.size) // self.stride + 1
        out = np.zeros((batch_size, C, out_H, out_W))
        self.max_indices = np.zeros_like(x)

        for b in range(batch_size):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = x[b, c, h_start:h_start + self.size, w_start:w_start + self.size]
                        max_val = np.max(region)
                        out[b, c, i, j] = max_val
                        mask = (region == max_val)
                        self.max_indices[b, c, h_start:h_start + self.size, w_start:w_start + self.size] += mask
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        batch_size, C, out_H, out_W = dout.shape

        for b in range(batch_size):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        dx[b, c, h_start:h_start + self.size, w_start:w_start + self.size] += (
                            self.max_indices[b, c, h_start:h_start + self.size, w_start:w_start + self.size] *
                            dout[b, c, i, j]
                        )
        return dx
# %%

# %%
np.random.seed(42)
x = np.random.randn(1, 1, 28, 28)  # One grayscale image
y = np.array([3])  # assuming 3 classes

# define the layers of the CNN
conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
relu = ReLU()
pool = MaxPool2D(size=2, stride=2)
# %%
