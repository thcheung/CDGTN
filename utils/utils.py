import re
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
import re
import os
import shutil

def preprocess(sentence):
    sentence = re.sub("[hH]ttp\S*", "http", sentence)    # remove url
    sentence = sentence.lower()                      # convert into lowercase
    return sentence.strip()

def print_metrics(y_true, y_pred):
    print(f"Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro')}")
    print(f"Accuracy): {accuracy_score(y_true, y_pred)}")

def labels_to_weights(labels):
    num = max(labels) + 1
    counts = [labels.count(i)+1 for i in range(0, num)]
    total = sum(counts)
    counts = [total/(count) for count in counts]
    return torch.tensor(counts, dtype=torch.float)

def length_to_mask(length, max_len=None, dtype=None):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    mask = ~mask
    return mask

def pad_tensor(tensor, batch_index):
    num_seq = torch.unique(batch_index)
    tensors = [tensor[batch_index == seq_id] for seq_id in num_seq]
    lengths = [len(tensor) for tensor in tensors]
    lengths = torch.tensor(lengths).to(num_seq.device)
    masks = length_to_mask(lengths)
    return pad_sequence(tensors, batch_first=True), masks.bool()

def clean_cache():
    remove_dirs = []

    for root, dirs, files in os.walk("preprocessed"):
        if root.endswith("\processed") or root.endswith("raw"):
            remove_dirs.append(root)

    for remove_dir in remove_dirs:
        shutil.rmtree(remove_dir)
