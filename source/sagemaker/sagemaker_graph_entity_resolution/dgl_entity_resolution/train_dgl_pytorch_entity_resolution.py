import torch
import dgl

import numpy as np

import time
import logging

from estimator_fns import *
from graph import *
from data import *
from model import *
from utils import *

def evaluate(model, g, train_triplets, test_triplets, user_features, web_features, batch_size, n_neighbors,
             hits=[1, 3, 10], device=None, filtered=False, mean_ap=False):

    logging.info("Performing model inference to get embeddings")
    embed = model.inference(g, user_features, web_features, batch_size, n_neighbors, device)
    logging.info("Got embeddings, computing metrics")

    w = model.w_relation.detach().clone().cpu()

    if mean_ap:
        metric = calc_mAP(embed, w, train_triplets, test_triplets)
    else:
        if filtered:
            metric = calc_filtered_mrr(embed, w, train_triplets, test_triplets, hits=hits)
        else:
            metric = calc_raw_mrr(embed, w, test_triplets, hits=hits, eval_bz=10000)
    return metric


def train(g, model, train_dataloader, train_triplets, test_triplets, user_features, web_features, optimizer, batch_size,
          n_neighbors, n_epochs, negative_rate, grad_norm, cuda, device=None, run_eval=True):
    for epoch in range(n_epochs):
        tic = time.time()
        duration = []
        loss_val = 0.
        mrr = -1.

        model.train()
        for n, (input_nodes, pos_pair_graph, neg_pair_graph, blocks) in enumerate(train_dataloader):
            user_nodes, website_nodes = input_nodes['user'], input_nodes['website']
            u, w = user_features[input_nodes['user']], web_features[input_nodes['website']]

            true_srcs, true_dsts = pos_pair_graph.all_edges(etype='same_entity')
            false_srcs, false_dsts = neg_pair_graph.all_edges(etype='same_entity')
            sources, sinks = torch.cat((true_srcs, false_srcs)), torch.cat((true_dsts, false_dsts))
            labels = torch.zeros((negative_rate + 1) * len(true_srcs))
            labels[:len(true_srcs)] = 1

            if cuda:
                user_nodes, website_nodes, u, w = user_nodes.cuda(), website_nodes.cuda(), u.cuda(), w.cuda()
                blocks = [blk.to(device) for blk in blocks]
                sources, sinks, labels = sources.cuda(), sinks.cuda(), labels.cuda()
            embeddings = model(blocks,user_nodes, website_nodes, u, w)

            loss = model.get_loss(embeddings, sources, sinks, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()

            loss_val += loss.item()

        duration.append(time.time() - tic)
        do_eval = run_eval and ((epoch %  5 == 0) or (epoch == n_epochs-1))
        if do_eval:
            mrr = evaluate(model, g, train_triplets, test_triplets, user_features, web_features, batch_size, n_neighbors,
                       device=device)

        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | MRR {:.4f}".format(
            epoch, np.mean(duration), loss_val / (n + 1), mrr))
    return model

def run():
    logging = get_logger(__name__)
    logging.info('numpy version:{} Pytorch version:{} DGL version:{}'.format(np.__version__,
                                                                             torch.__version__,
                                                                             dgl.__version__))

    args = parse_args()

    g, (user_features, website_features), id_to_node, reverse = construct_graph(args.training_dir, args.train_edges,
                                                                                args.transient_nodes,
                                                                                args.transient_edges,
                                                                                args.website_nodes,
                                                                                args.website_group_edges)

    logging.info("""----Data statistics------
                            #Nodes: {}
                            #Edges: {}
                            #Same entity train edges: {}
                            #User Features Shape: {}
                            #Web Features Shape: {}""".format(sum([g.number_of_nodes(n_type) for n_type in g.ntypes]),
                                                              sum([g.number_of_edges(e_type) for e_type in g.etypes]),
                                                              g.number_of_edges('same_entity'),
                                                              user_features.shape,
                                                              website_features.shape))

    user_features, website_features = torch.tensor(user_features), torch.tensor(website_features)

    model = EntityResolution(g, args.embedding_size, args.n_hidden, user_features.shape[1], website_features.shape[1],
                             num_hidden_layers=args.n_layers, reg_param=args.regularization_param)

    cuda = args.num_gpus > 0 and torch.cuda.is_available()
    device = 'cpu'
    if cuda:
        torch.cuda.set_device(0)
        model.cuda()
        device = 'cuda:%d' % torch.cuda.current_device()

    # split into train and test
    us, vs, eids = g.all_edges(etype='same_entity', form='all')
    train_eids = np.random.choice(len(eids), int(0.7 * len(eids)), replace=False)
    test_eids = np.setdiff1d(np.arange(len(eids)), train_eids)
    train_triplets = torch.tensor(
        np.vstack((us[train_eids], np.zeros(len(train_eids), dtype=int), vs[train_eids])).transpose())
    test_triplets = torch.tensor(
        np.vstack((us[test_eids], np.zeros(len(test_eids), dtype=int), vs[test_eids])).transpose())
    logging.info("Split into train and test edges with {} train and {} test".format(len(train_eids), len(test_eids)))

    sampler = dgl.dataloading.MultiLayerNeighborSampler([args.n_neighbors] * args.n_layers) if args.mini_batch \
        else dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.negative_sampling_rate)
    collator = dgl.dataloading.EdgeCollator(g, {'same_entity': train_eids}, sampler, exclude='reverse_types',
                                            reverse_etypes=reverse, negative_sampler=neg_sampler)
    train_dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate, shuffle=True,
                                                   batch_size=args.batch_size, drop_last=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = train(g, model, train_dataloader, train_triplets, test_triplets, user_features, website_features, optimizer,
                  args.batch_size, args.n_neighbors, args.n_epochs, args.negative_sampling_rate, args.grad_norm, cuda,
                  device=device)


if __name__ == '__main__':
    model = run()
