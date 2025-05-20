#%%
import numpy as np

# for simplicity, i'll use a 1x1x8x8 input image only
# Batch size = 1 Channels = 1 Height = 8 Width = 8

#%%
np.random.seed(42)
input_image = np.random.rand(1, 1, 8, 8)

print("Input shape:", input_image.shape)
