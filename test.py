import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from torch.utils.tensorboard import SummaryWriter
from data_loader import RumorDataset
from model import get_model
from experiment import get_experiment
import time
import torch_geometric.transforms as T
import warnings
from utils import clean_cache

os.environ["CUDA_VISIBLE_DEVICES"]="0"

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Tree Rumor Detection and Verification')

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 32)')

parser.add_argument('--hidden_dim', type=int, default=768, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=64, metavar='N',
                    help='maximum length of the conversation (default: 50)')

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


def test():

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

    print('num of training / testing samples : {} / {} '.format(len(test_dataset), len(test_dataset)))

    model = get_model(args.model,args.hidden_dim, len(classes),0.0 , language=language)
        
    model = DataParallel(model).to(device)
    model.eval()
    
    comment = f'{args.model}_{args.experiment}_{args.fold}_{args.seed}'

    writer = SummaryWriter(log_dir="runs/{}_{}".format(str(int(time.time())),"time_" + comment))

    MAX_NODES = [25, 50, 75, 100]

    for MAX_NODE in MAX_NODES:
        total_times = 0.0
        total_count = 0.0
        for _, batch in enumerate(tqdm(test_loader)):
            
            labels = torch.cat([data.y for data in batch]).to(device).long()

            for idx, data in enumerate(batch):

                num_nodes = int(data.num_nodes)

                if num_nodes == 1:
                    continue

                if num_nodes < MAX_NODE:
                    continue

                for step in range(1,num_nodes+1):

                    if step > MAX_NODE:
                        continue

                    num_true = int(step)
                    num_false = int(num_nodes - num_true)

                    tensor_false = torch.zeros(num_false, dtype=torch.bool)
                    tensor_true = torch.ones(num_true, dtype=torch.bool)

                    subset = torch.cat([tensor_true,tensor_false])
                    batch[idx] = data.subgraph(subset)

                    if step > 1:
                        start = time.time()
                        outputs = model(batch)
                        outputs = outputs[0] if type(outputs) is tuple else outputs
                        _, preds = torch.max(outputs, 1)
                        end = time.time()                        
                        total_time = end - start
                        total_times += total_time
                        total_count += 1

        times = total_times/ total_count
        writer.add_scalar("Time(s)", times, MAX_NODE)        

if __name__ == "__main__":
    test()
