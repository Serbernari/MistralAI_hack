import numpy as np
from sklearn.decomposition import PCA


def compute_embeddings(elements, model, tokenizer, layer='mean', layer_num=32, pca=False):
    embeddings = []
    for elem in elements:
        inputs = tokenizer(elem, return_tensors="pt").to('cuda')
        out = model(**inputs, output_hidden_states=True)
        if layer == 'mean':
            emb = out['hidden_states'][layer_num].mean(dim=1)  # [:, -1]
        elif layer == 'last':
            emb = out['hidden_states'][layer_num][:, -1]
        else:
            raise ValueError(f'unknown method {layer}, choose from "mean", "last"')
        embeddings.append(emb.view(-1).cpu().detach().float().numpy())
    if pca:
        pca_instance = PCA(n_components=pca, whiten=True)
        pca_instance.fit(embeddings)
        embeddings = pca_instance.transform(embeddings)

    return np.array(embeddings)
