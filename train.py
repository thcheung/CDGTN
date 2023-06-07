import argparse
import time
import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from data_loader import RumorDataset
from model import get_model
from utils import print_metrics, clean_cache
from experiment import get_experiment
from transformers import logging
import warnings
from transformers import get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

warnings.filterwarnings("ignore")

logging.set_verbosity_error()

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Early Graph Rumor Detection and Verification (baseline)')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')

parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')

parser.add_argument('--lr', type=float, default=5e-5, metavar='N',
                    help='learning rate (default: 5e-5)')

parser.add_argument('--hidden_dim', type=int, default=768, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=64, metavar='N',
                    help='maximum length of the conversation (default: 64)')

parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                    help='dropout rate (default: 0.1)')

parser.add_argument('--experiment', type=str, default="CR_Twitter", metavar='N',
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


def train():

    RANDOM_SEED = args.seed

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    clean_cache()

    experiment = get_experiment(args.experiment)

    root_dir = os.path.join(experiment["root_dir"], str(args.fold))

    language = experiment["language"]

    classes = experiment["classes"]

    train_dataset = RumorDataset(
        root=root_dir,
        classes=classes,
        split='train',
        language=language,
        max_length=args.max_len,
        aug=args.aug
        )
    
    train_loader = DataListLoader(
        train_dataset, batch_size=args.batch_size,shuffle=True,drop_last=False)

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

    print('num of training / testing samples : {} / {} '.format(len(train_dataset), len(test_dataset)))

    model = get_model(args.model,args.hidden_dim, len(classes),
                         args.dropout, language=language)

    model = DataParallel(model).to(device)

    labels = [int(item.y) for item in train_dataset]

    criterion = nn.NLLLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer,0 , int(args.epochs))

    comment = f'{args.model}_{args.experiment}_{args.fold}_{args.seed}'

    writer = SummaryWriter(log_dir="runs/{}_{}".format(str(int(time.time())),"train_" + comment))

    checkpoint_dir = os.path.join("checkpoints/",comment)
    os.makedirs(checkpoint_dir,exist_ok=True)

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):

        model.train()

        train_loss = 0.0
        train_count = 0
        train_predicts = []
        train_labels = []

        for _, batch in enumerate(tqdm(train_loader)):
            
            optimizer.zero_grad()

            outputs = model(batch)

            labels = torch.cat([data.y for data in batch]).to(device).long()
            
            outputs = outputs
            loss = criterion(outputs, labels) 

            train_loss += loss.item() * labels.size(0)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            train_count += labels.size(0)

            train_labels.append(labels.cpu().detach().numpy())
            train_predicts.append(preds.cpu().detach().numpy())

        scheduler.step()
        train_labels = np.concatenate(train_labels).ravel()
        train_predicts = np.concatenate(train_predicts).ravel()

        train_loss = train_loss / train_count
        train_acc = accuracy_score(train_labels, train_predicts)
        train_f1_mac = f1_score(train_labels, train_predicts, average='macro')

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1_Score/train", train_f1_mac, epoch)

        model.eval()

        test_loss = 0.0
        test_count = 0
        test_predicts = []
        test_labels = []

        for _, batch in enumerate(tqdm(test_loader)):

            labels = torch.cat([data.y for data in batch]).to(device).long()

            with torch.no_grad():

                outputs = model(batch)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, 1)

            test_count += labels.size(0)

            test_labels.append(labels.cpu().detach().numpy())
            test_predicts.append(preds.cpu().detach().numpy())

        test_labels = np.concatenate(test_labels).ravel()
        test_predicts = np.concatenate(test_predicts).ravel()

        test_loss = test_loss / test_count
        test_acc = accuracy_score(test_labels, test_predicts)
        test_f1_mac = f1_score(test_labels, test_predicts, average='macro')

        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("F1_Score/test", test_f1_mac, epoch)

        print("Epoch: {} / {}".format(epoch, args.epochs))
        print_metrics(test_labels, test_predicts)

        if test_f1_mac > best_f1:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.module.state_dict(),checkpoint_path)
            best_f1 = test_f1_mac

    checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.module.state_dict(),checkpoint_path)

if __name__ == "__main__":
    train()