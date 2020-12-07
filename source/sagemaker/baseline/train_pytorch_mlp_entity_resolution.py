import argparse
import os
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiamesePairwiseClassification(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size=16):
        super(SiamesePairwiseClassification, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        if n_layers > 1:
            for i in range(n_layers - 1):
                layers.extend((nn.Linear(hidden_size, hidden_size), nn.ReLU()))
        self.layers = nn.Sequential(*layers)
        self.score = nn.CosineSimilarity()
        self.w_relation = nn.Parameter(torch.Tensor(1, hidden_size))

    def forward(self, x):
        return self.layers(x)

    def calc_score(self, embed_i, embed_j):
        score = torch.sum(self.w_relation * embed_i * embed_j, dim=1)
        # score = self.score(embed[i], embed[j])
        return score

    def get_loss(self, embed, i, j, labels):
        score = self.calc_score(embed[i] , embed[j])
        loss = F.binary_cross_entropy_with_logits(score, labels) + torch.mean(embed.pow(2))
        return loss


def read_data(training_dir, user_features, url_features, transient_edges, train_edges):
    user_features_df = pd.read_csv(os.path.join(training_dir, user_features), header=None).set_index(0)
    logging.info("Read user features".format(os.path.join(training_dir, user_features)))

    url_features_df = pd.read_csv(os.path.join(training_dir, url_features), header=None).set_index(0)
    logging.info("Read url features from {}".format(os.path.join(training_dir, url_features)))

    transient_interactions = pd.read_csv(os.path.join(training_dir, transient_edges), header=None)
    logging.info("Read transient_interactions {}".format(os.path.join(training_dir, transient_edges)))

    transient_interactions = transient_interactions.groupby([0])[1].apply(','.join).reset_index().drop_duplicates().set_index(0)
    logging.info("Grouped transient_interactions")

    (n_user, d_user), d_url,  = user_features_df.shape, url_features_df.shape[1]
    features = np.zeros((n_user, d_user + d_url))
    for i, (uid, row) in enumerate(user_features_df.iterrows()):
        features[i, :d_user] = row
        features[i, d_user:] = url_features_df.loc[transient_interactions.loc[uid].values[0].split(',')].mean(axis=0)

    train_pairs = pd.read_csv(os.path.join(training_dir, train_edges), header=None)
    logging.info("Read ground truth training pairs {}".format(os.path.join(training_dir, train_edges)))
    uid_to_idx = {uid: i for i, uid in enumerate(user_features_df.index.values)}
    map_uid_to_idx = lambda x: uid_to_idx[x]
    true_i = train_pairs[0].apply(map_uid_to_idx)
    true_j = train_pairs[1].apply(map_uid_to_idx)
    return features.astype(np.float32), true_i, true_j, uid_to_idx

def convert_to_adj_list(i, j):
    adj_list = {}
    for (a, b) in zip(i, j):
        if a in adj_list:
            adj_list[a].append(b)
        else:
            adj_list[a] = [b]
        if b in adj_list:
            adj_list[b].append(a)
        else:
            adj_list[b] = [a]
    return adj_list

def train(model, dataloader, features, n_epochs, optimizer, neg_rate, cuda):
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.
        duration = []
        metric = -1
        for n, (i, j) in enumerate(dataloader):
            labels = torch.zeros((neg_rate + 1) * len(i))
            labels[:len(i)] = 1
            i = torch.cat((i, torch.tensor(np.random.choice(features.shape[0], neg_rate*len(i)))))
            j = torch.cat((j, torch.tensor(np.random.choice(features.shape[0], neg_rate*len(j)))))

            if cuda:
                i, j, labels = i.cuda(), j.cuda(), labels.cuda()

            embed = model(features)
            loss = model.get_loss(embed, i, j, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            duration.append(time.time() - tic)
        print(loss_val)
        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | MRR {:.4f}".format(
            epoch, np.mean(duration), loss_val / (n + 1), metric))

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train-edges', type=str, default='user_train_edges.csv')
    parser.add_argument('--test-edges', type=str, default='user_test_edges.csv')
    parser.add_argument('--transient-edges', type=str, default='transient_edges.csv')
    parser.add_argument('--user-features', type=str, default='transient_nodes.csv')
    parser.add_argument('--url-features', type=str, default='website_nodes.csv')
    parser.add_argument('--n-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--n-layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--negative-sampling-rate', type=int, default=10, help='rate of negatively sampled edges')
    parser.add_argument('--n-epochs', type=int, default=20)

    return parser.parse_args()

if __name__ == '__main__':
    logging = get_logger(__name__)
    logging.info('numpy version:{} Pytorch version:{}'.format(np.__version__, torch.__version__))

    args = parse_args()
    features, true_i, true_j, uid_to_idx = read_data(args.training_dir,
                                                     args.user_features,
                                                     args.url_features,
                                                     args.transient_edges,
                                                     args.train_edges)

    train_idxs = np.random.choice(len(true_i), int(0.7 * len(true_i)), replace=False)
    test_idxs = np.setdiff1d(np.arange(len(true_i)), train_idxs)
    train_i, train_j, test_i, test_j = true_i[train_idxs], true_j[train_idxs], true_i[test_idxs], true_j[test_idxs]

    adj_list = convert_to_adj_list(true_i, true_j)
    features = torch.tensor(features)

    model = SiamesePairwiseClassification(args.n_layers, features.shape[1], hidden_size=args.n_hidden)

    cuda = args.num_gpus > 0 and torch.cuda.is_available()
    device = 'cpu'
    if cuda:
        torch.cuda.set_device(0)
        model.cuda()
        features = features.cuda()
        device = 'cuda:%d' % torch.cuda.current_device()


    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(train_i.values),
                                                                                 torch.tensor(train_j.values)),
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   drop_last=False,
                                                   num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(model, train_dataloader, features, args.n_epochs, optimizer, args.negative_sampling_rate, cuda)




