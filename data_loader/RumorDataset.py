import os
import json
import shutil
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import preprocess

class RumorDataset(Dataset):
    def __init__(self, root, split, classes, language='en', max_length=64, transform=None, pre_transform=None, aug = False):

        self.split = split
        self.filename = "{}.json".format(split)
        self.aug = aug
        self.classes = classes
        self.language = language
        self.root = root
        self.max_length = max_length
        # self.max_nodes = 128 ## Twitter15 / Twitter16
        
        if language == 'en':
            self.max_nodes = 50 ## PHEME
        else:
            self.max_nodes = 128 if split == "train" else 100 ## Other than PHEME

        self.textTokenizer = self._get_tokenizer()

        super(RumorDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        data_len = (len(self.data))
        return [f'data_{self.split}_{i}.pt' for i in range(data_len)]

    def download(self):
        download_path = self.raw_dir
        os.makedirs(download_path,exist_ok=True)
        file_path = os.path.join(self.root, self.filename)
        shutil.copy(file_path,download_path)

    def process(self):
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        for index, tweet in (enumerate(tqdm(self.data))):
            tweet["nodes"] = tweet["nodes"][:self.max_nodes]
            tweet["edges"] = tweet["edges"][:self.max_nodes-1]
            tweet_id = tweet['id']
            
            node_feats = self._get_node_features(tweet["nodes"])

            if self.aug:
                edge_index = self._get_adjacency_info(tweet["edges"])
            else:
                edge_index= self._get_adjacency_info1(tweet["edges"])

            label = self._get_labels(tweet['label'])

            data = Data(x=node_feats,
                        edge_index = edge_index,                        
                        y=label,
                        id=tweet_id,
                        )

            torch.save(data,
                       os.path.join(self.processed_dir,
                                    f'data_{self.split}_{index}.pt'))

    def _get_tokenizer(self):
        if self.language == 'en':
            return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
        elif self.language == 'cn':
            return AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        # if self.language == 'en':
        #     return AutoTokenizer.from_pretrained("bert-base-uncased")
        # elif self.language == 'cn':
        #     return AutoTokenizer.from_pretrained("bert-base-chinese")

    def _get_node_features(self, nodes):
        texts = [preprocess(node['text']) for node in nodes]
        encoded_input = self.textTokenizer.batch_encode_plus(
            texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')

        all_node_feats = torch.stack([
            encoded_input["input_ids"], encoded_input["attention_mask"]], dim=-1)
        return all_node_feats

    def _get_edge_features(self, edge_len):
        return torch.ones(edge_len, 1)

    def _get_adjacency_info1(self, edges):
        edge_indices = []
        
        for edge in edges:
            i = int(edge['from'])
            j = int(edge['to'])
            edge_indices += [[j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices


    def _get_adjacency_info(self, edges):
        edge_indices = []
        
        for edge in edges:
            i = int(edge['from'])
            j = int(edge['to'])
            edge_indices += [[j, i]]

            while j != 0:
                for edge2 in edges:
                    edge_from = int(edge2['from'])
                    edge_to = int(edge2['to'])
                    if edge_from == j:
                        j = edge_to
                        edge_indices += [[j, i]]
                        continue

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices

    def _get_labels(self, label):
        label = self.classes.index(label)
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int32)

    def len(self):
        return len(self.data)

    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir,
                          f'data_{self.split}_{idx}.pt'))
    
        return data