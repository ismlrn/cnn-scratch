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
