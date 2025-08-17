# import pickle
# import os

# predictions = pickle.load(
#     open("exps/thumos/actionformer_i3d/outputs/video_test_0000004.pkl", "rb")
# )
# print(predictions)


import numpy as np

# Load the .npy file
feature = np.load("data/charades/features/i3d_charades_finetuned_stride8/F44A2.npy")

# Now `data` is a NumPy array
print(feature.shape)

# Load the .npy file
feature_ego = np.load("data/charades/features_ego/A2PCKEGO.npy")

# Now `data` is a NumPy array
print(feature_ego.shape)
