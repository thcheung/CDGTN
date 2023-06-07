from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch
from transformers import AutoModel
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor
from utils import length_to_mask

class CDGTN(nn.Module):
    def __init__(self, hidden_dim=768, label_dim=3, dropout_rate=0.1, language='en', mh_size = 2):
        super(CDGTN, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language
        self.label_dim = label_dim
        self.mh_size = mh_size

        self.textEncoder = self.get_text_model()
        self.freeze_textEncoder()

        self.gnn1 = TransformerConv(hidden_dim, hidden_dim // mh_size, heads=mh_size)
        self.gnn2 = TransformerConv(hidden_dim, hidden_dim // mh_size, heads=mh_size)

        self.att_i = nn.Linear(hidden_dim*2, hidden_dim)  
        self.att_s = nn.Linear(hidden_dim, 1)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, label_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.act = nn.ReLU()
     
    def get_text_model(self):
        if self.language == 'en':
            return AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        elif self.language == 'cn':
            return AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")

    def freeze_textEncoder(self):
        for name, param in list(self.textEncoder.named_parameters()):
            name = name.lower()
            if self.language == 'en':
                if 'pooler' in name or 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'encoder.layer.9' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            elif self.language == 'cn':
                if 'pooler' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def pad(self, tensor, batch_index):
        num_seq = torch.unique(batch_index)
        tensors = [tensor[batch_index == seq_id] for seq_id in num_seq]
        lengths = [len(tensor) for tensor in tensors]
        lengths = torch.tensor(lengths).to(num_seq.device)
        masks = length_to_mask(lengths)
        return pad_sequence(tensors, batch_first=True), masks.bool()
    
    def get_x(self, x):
        x_id = x[:, :, 0].int()
        x_mask = x[:, :, 1]
        x = self.textEncoder(input_ids=x_id, attention_mask=x_mask)
        x = x.pooler_output
        return x

    def attention_pooling(self,x ,x_mask):
        
        x_i = x[:,0]
        x_k = x
        x_v = x
        x_mask = x_mask
        x_q = x_i.unsqueeze(1).expand(-1,x_k.size(1),-1)

        x_c = torch.cat([x_q,x_k],dim=-1)
        x_c = torch.relu(self.att_i(x_c))
        x_score = self.att_s(x_c)

        x_v[x_mask] = float("0.0")
        x_score[x_mask] = float("-inf")

        if self.training:
            x_score = torch.exp(x_score)
            x_out = x_v * x_score
            x_out = torch.cumsum(x_out,1).squeeze(1)
            x_score = torch.cumsum(x_score,1) + 1e-10
            x_out = x_out / x_score
            
        else:          
            x_score = torch.softmax(x_score,1)
            x_out = x_v * x_score
            x_out = torch.sum(x_out , 1)

        return x_out , x_score.squeeze(-1)

    def mean_pooling(self, x, x_mask):
        x_score = 1.0 / torch.sum((~x_mask).int().float(),1).unsqueeze(-1).expand(-1, x.size(1))
        x_score[x_mask] = float(0.0)
        x = x * x_score.unsqueeze(-1)
        x = torch.sum(x,1)
        return x

    def get_graph_mask(self,edge_index, batch_index):
        graph_mask = to_dense_adj(edge_index=edge_index,batch=batch_index)
        g_mask = graph_mask.repeat(self.mh_size,1,1)
        g_mask_bool = (~(g_mask.int().bool()))
        mask = torch.eye(g_mask.size(1)).repeat(g_mask.size(0), 1, 1).bool()
        g_mask_bool[mask] = False
        return g_mask_bool

    def forward(self, data):

        x, edge_index, batch_index = data.x, data.edge_index, data.batch
  
        x = self.get_x(x)

        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(x.size(0), x.size(0))).t()
        
        x = self.gnn1(x,adj)
        x = self.act(x)
        
        x = self.gnn2(x,adj)
        x = self.act(x)

        x , x_mask = self.pad(x, batch_index)
        
        x , _ = self.attention_pooling(x, x_mask)
        
        # x = self.mean_pooling(x,x_mask)

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        x = torch.log_softmax(x,-1)
        
        if self.training:
            x = torch.mean(x,1)

        return x