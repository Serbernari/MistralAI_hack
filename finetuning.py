import argparse
import re

import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler, MistralModel, AutoConfig

from clustering_metrics import evaluate_clusters
from datasets import Dataset
from loss import SenseContrastLoss
from utils import compute_embeddings

with open('datasets/grocery_annotated_filtered.txt') as grocery_annotated:
    grocery_annotated = grocery_annotated.readlines()
    grocerylabels = [x.strip().split('\t')[1] for x in grocery_annotated][1:]
    grocerylist = [x.strip().split('\t')[0] for x in grocery_annotated][1:]

measures_products = "|".join(['tbsp', 'cl', 'cup', 'g'])

grocerylist = [re.sub('[\W^\d]+', ' ', x) for x in grocerylist if x != '']
grocerylist = [re.sub(f'\\b{measures_products}\\b', ' ', x) for x in grocerylist]
grocerylist = [x.strip() for x in grocerylist]

dataset = pd.read_csv('datasets/auchan_dataset_filtered.csv')


def get_last_token(output, tokenized):
    attention_mask = tokenized.attention_mask
    last_hidden_state = output['last_hidden_state']
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def eval_clustering(fn_model, tokenizer):
    print('Clustering starts...')
    embedding = 'last'
    pca = False
    n = -1
    embs = compute_embeddings(grocerylist, fn_model, tokenizer, layer=embedding, layer_num=n, pca=pca)
    cat_embs_dict = dict()
    for cat, groupdf in dataset.groupby('category'):
        cat_emb = compute_embeddings(groupdf.elem.tolist(), fn_model, tokenizer, layer=embedding, layer_num=n, pca=pca)
        cat_centroid = cat_emb.mean(axis=0)
        cat_embs_dict[cat] = cat_centroid

    tmp = pd.DataFrame(columns='text label pred sim'.split())
    for text, label, line in zip(grocerylist, grocerylabels, cosine_similarity(embs, list(cat_embs_dict.values()))):
        tmp.loc[tmp.shape[0]] = text, label, list(cat_embs_dict.keys())[line.argmax()], line.max()
    wandb.log(evaluate_clusters(tmp.label, tmp.pred))
    my_table = wandb.Table(dataframe=tmp)
    wandb.log({"clustering": my_table})
    print('Clustering finished...')


def train(model_id, model_num_layers, num_epochs, lr, lr_scheduler_type, weight_decay, temperature):
    batch_size = 8
    eval_step = 100

    model_config = AutoConfig.from_pretrained(model_id)
    model_config.num_hidden_layers = model_num_layers
    model = MistralModel.from_pretrained(model_id, device_map="auto", config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_pandas(dataset[dataset.split == 'train'])
    test_dataset = Dataset.from_pandas(dataset[dataset.split == 'test'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = 0.03 * num_training_steps

    run = wandb.init(project="contrastive_mistral",
                     config={"model": model_id, 'model_layers': model_num_layers, "task": 'contrastive',
                             "epochs": num_epochs, "lr": lr, "batch_size": batch_size, "warmup_steps": num_warmup_steps,
                             'temperature': temperature, 'eval_step': eval_step, 'dataset': 'reduced',
                             'lr_scheduler_type': lr_scheduler_type
                             })

    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    loss_fn = SenseContrastLoss(temperature=temperature)
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler)

    progress_bar = tqdm(range(num_training_steps), leave=True, position=0)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            tokenized = tokenizer(batch['elem'], padding=True, return_tensors="pt").to('cuda')
            labels = batch['category_encoded']
            output = model(**tokenized)
            out = get_last_token(output, tokenized)
            loss = loss_fn(out, labels)
            wandb.log({'loss': loss})
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f'loss: {round(loss.item(), 3)}', refresh=True)
            if step % eval_step == 0:
                model.eval()
                eval_clustering(model, tokenizer)
                model.train()
    accelerator.end_training()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune Mistral')
    parser.add_argument('--model_id', type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument("--decay", type=float, help='weight decay',
                        default=0.01)
    parser.add_argument("--lr", type=float, help='learning rate',
                        default=5e-6)
    parser.add_argument('--temperature', type=float, help='temperature to test',
                        default=0.5)
    parser.add_argument('--num_epochs', type=int, help='how many epochs are needed',
                        default=10)
    parser.add_argument('--lr_scheduler_type', type=str, help='scheduler',
                        default='linear')
    parser.add_argument('--model_num_layers', type=int, help='how many layers to leave',
                        default=16)

    args = parser.parse_args()
    print(args)

    train(args.model_id, args.model_num_layers, args.num_epochs, args.lr, args.lr_scheduler_type, args.decay,
          args.temperature)
