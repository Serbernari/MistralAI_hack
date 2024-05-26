import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MistralModel, AutoTokenizer

from utils import compute_embeddings

app = Flask(__name__)

tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "annamos/Mistral-7B-Instruct-v0.3-pruned-CL"
embedding = 'last'
pca = False
n = -1

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = MistralModel.from_pretrained(model_name, device_map="auto")

dataset = pd.read_csv('datasets/auchan_dataset_filtered.csv')
cat_embs_dict = dict()
for cat, groupdf in dataset.groupby('category'):
    cat_emb = compute_embeddings(groupdf.elem.tolist(), model, tokenizer, layer=embedding, layer_num=n, pca=pca)
    cat_centroid = cat_emb.mean(axis=0)
    cat_embs_dict[cat] = cat_centroid


@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.json
    grocerylist = data['grocerylist']
    print(grocerylist)
    embs = compute_embeddings(grocerylist, model, tokenizer, layer=embedding, layer_num=n, pca=pca)
    sims = cosine_similarity(embs, list(cat_embs_dict.values()))
    preds = [list(cat_embs_dict.keys())[x] for x in sims.argmax(axis=1)]
    return jsonify({'clusters': preds})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
