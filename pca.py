import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas

def pca(x, n_components=None, whiten=False):
    pca = PCA(n_components=n_components, whiten=whiten)
    return pca.fit_transform(x).components_.astype(np.float32)
    
def plot_pca(x, n_components=None, whiten=False):
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(x)
    print(pca.explained_variance_ratio_)
    plt.scatter(pca.components_[0], pca.components_[1])
    plt.show()

vectors_file = "steering_vectors/unsloth_Llama-3_2-3B-Instruct_torch_vectors/layer_0_negative.pt"
vectors = torch.load(vectors_file, map_location=torch.device('cpu') ).squeeze()

vectors_df = pandas.DataFrame(vectors)
print(vectors_df.head())
vectors_df.to_csv("vectors.csv")



plot_pca(vectors)

