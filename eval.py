import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from sklearn.metrics import classification_report
from data_loader import RumorDataset
from model import get_model
from utils import print_metrics, clean_cache
from experiment import get_experiment
from transformers import logging
import warnings

os.environ["CUDA_VISIBLE_DEVICES"]="0"

warnings.filterwarnings("ignore")

logging.set_verbosity_error()

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Early Graph Rumor Detection and Verification (baseline)')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
                    
parser.add_argument('--hidden_dim', type=int, default=768, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=64, metavar='N',
                    help='maximum length of the conversation (default: 32)')

parser.add_argument('--experiment', type=str, metavar='N',
                    help='experiment name')

parser.add_argument('--model', type=str, default="CDGTN", metavar='N',
                    help='model name')

parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='experiment name')

parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='experiment name')

parser.add_argument('--aug', type=bool, default=True, metavar='N',
                    help='experiment name')

args = parser.parse_args()


def eval():
    
    RANDOM_SEED = args.seed

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # clean_cache()
    
    experiment = get_experiment(args.experiment)

    root_dir = os.path.join(experiment["root_dir"], str(args.fold))

    language = experiment["language"]

    classes = experiment["classes"]

    test_dataset = RumorDataset(
        root=root_dir,
        classes=classes,
        split='test',
        language=language,
        max_length=args.max_len,
        aug=args.aug
        )    
        
    test_loader = DataListLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    print('testing samples : {} '.format(len(test_dataset)))

    model = get_model(args.model,args.hidden_dim, len(classes), dropout=0.0, language=language)

    model = DataParallel(model).to(device)

    comment = f'{args.model}_{args.experiment}_{args.fold}_{args.seed}'

    checkpoint_dir = os.path.join("checkpoints/",comment)

    with torch.no_grad():
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        model.module.load_state_dict(torch.load(checkpoint_path))

    model.eval()

    test_count = 0

    predicts = []
    test_labels = []

    for _, batch in enumerate(tqdm(test_loader)):

        labels = torch.cat([data.y for data in batch]).to(device).long()

        with torch.no_grad():
            outputs = model(batch)
            outputs = outputs[0] if type(outputs) is tuple else outputs

        _, predict = torch.max(outputs, 1)

        test_count += labels.size(0)

        test_labels.append(labels.cpu().detach().numpy())
        predicts.append(predict.cpu().detach().numpy())

    test_labels = np.concatenate(test_labels).ravel()
    predicts = np.concatenate(predicts).ravel()

    all_labels = [int(test_label) for test_label in test_labels]

    all_labels = list(set(all_labels))
    all_labels = sorted(all_labels)
    print_metrics(test_labels, predicts)
    print(classification_report(test_labels, predicts,digits=3))


if __name__ == "__main__":
    eval()