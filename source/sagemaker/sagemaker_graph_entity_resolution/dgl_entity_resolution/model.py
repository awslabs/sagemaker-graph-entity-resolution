"""RGCN layer implementation"""
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_sizes, out_sizes, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name, in_size, out_size in zip(etypes, in_sizes, out_sizes)
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                # Specify per-relation message passing functions: (message_func, reduce_func).
                # Note that the results are saved to the same destination feature 'h', which
                # hints the type wise reducer for aggregation.
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.dstnodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.dstnodes[ntype].data}


class HeteroRGCN(nn.Module):
    def __init__(self, g, in_size, hidden_size, n_layers):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), in_size['default']))
                      for ntype in g.ntypes if ntype != 'user' and ntype != 'website'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)

        # Prepare R-GCN layer input output size for each relation
        in_sizes = []
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype in in_size:
                in_sizes.append(in_size[srctype])
            else:
                in_sizes.append(in_size['default'])

        hidden_sizes = [hidden_size] * len(g.etypes)

        # create layers
        layers = [HeteroRGCNLayer(in_sizes, hidden_sizes, g.etypes)]
        if n_layers > 1:
            # additional hidden layers
            for i in range(n_layers - 1):
                layers.append(HeteroRGCNLayer(hidden_sizes, hidden_sizes, g.etypes))
        self.layers = nn.Sequential(*layers)

    def forward(self, g, user_features, website_features):
        # get embeddings for all node types. for user node type, use passed in user features
        h_dict = {}
        h_dict['user'] = nn.Parameter(user_features)
        h_dict['website'] = nn.Parameter(website_features)

        for ntype in self.embed:
            if g[0].number_of_nodes(ntype) > 0:
                h_dict[ntype] = self.embed[ntype][g[0].nodes(ntype).long(), :]

        # pass through all layers
        for i, layer in enumerate(self.layers):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g[i], h_dict)

        # get user logits
        return h_dict['user']


class EmbeddingLayer(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(EmbeddingLayer, self).__init__()
        self.embed = nn.Embedding(input_size, embedding_size)

    def forward(self, nodes):
        features = self.embed(nodes)
        return features


class EntityResolution(nn.Module):
    def __init__(self, g, in_dim, h_dim, user_feature_dim, website_feature_dim,
                 num_hidden_layers=1, reg_param=0):
        super(EntityResolution, self).__init__()
        self.user_embedding = EmbeddingLayer(g.number_of_nodes('user'), in_dim)
        self.website_embedding = EmbeddingLayer(g.number_of_nodes('website'), in_dim)
        in_size = {'default': in_dim, 'user': in_dim + user_feature_dim, 'website': in_dim + website_feature_dim}

        self.rgcn = HeteroRGCN(g, in_size, h_dim, num_hidden_layers)
        self.n_hidden = h_dim
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(1, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, sources, sinks):
        # DistMult
        h = embedding[sources]
        t = embedding[sinks]
        score = torch.sum(self.w_relation * h * t, dim=1)
        return score

    def forward(self, g, user_nodes, website_nodes, user_features, website_features):
        user_embed, website_embed = self.user_embedding(user_nodes), self.website_embedding(website_nodes)
        u = torch.cat((user_embed, user_features), 1)
        w = torch.cat((website_embed, website_features), 1)

        return self.rgcn(g, u, w)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, sources, sinks, labels):
        # sources and sinks is a list of edge data samples (positive and negative)
        score = self.calc_score(embed, sources, sinks)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

    def inference(self, g, user_features, web_features, batch_size, n_neighbors, device, num_workers=0):
        for l, layer in enumerate(self.rgcn.layers):
            sampler = dgl.dataloading.MultiLayerNeighborSampler([n_neighbors])
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {ntype: torch.arange(g.number_of_nodes(ntype)) for ntype in g.ntypes},
                sampler,
                batch_size= batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            y_user =  torch.zeros(g.number_of_nodes('user'), self.n_hidden)
            y_website = torch.zeros(g.number_of_nodes('website'), self.n_hidden)
            y_others = {ntype: torch.zeros(g.number_of_nodes(ntype), self.n_hidden)
                        for ntype in g.ntypes if ntype != 'user' and ntype != 'website'}

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)

                # get initial features
                if l == 0:
                    u_f, w_f = user_features[input_nodes['user']], web_features[input_nodes['website']]
                    u_f, w_f = u_f.to(device), w_f.to(device)
                    user_nodes, website_nodes = input_nodes['user'].to(device), input_nodes['website'].to(device)

                    # get embeddings and concat with initial features
                    user_embed, website_embed = self.user_embedding(user_nodes), self.website_embedding(website_nodes)
                    u = torch.cat((user_embed, u_f), 1)
                    w = torch.cat((website_embed, w_f), 1)

                # get intermediate representations
                else:
                    u = y_user[input_nodes['user']].to(device)
                    w = y_website[input_nodes['website']].to(device)

                h_dict = {}
                h_dict['user'] = nn.Parameter(u)
                h_dict['website'] = nn.Parameter(w)

                for ntype in self.rgcn.embed:
                    if block.number_of_nodes(ntype) > 0:
                        if l == 0:
                            h_dict[ntype] = self.rgcn.embed[ntype][block.nodes(ntype).long(), :]
                        else:
                            h_dict[ntype] = y_others[ntype][input_nodes[ntype]].to(device)

                h_dict = layer(block, h_dict)
                if l != len(self.rgcn.layers) - 1:
                    h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
                if len(output_nodes['user']):
                    y_user[output_nodes['user']] = h_dict['user'].cpu()
                if len(output_nodes['website']):
                    y_website[output_nodes['website']] = h_dict['website'].cpu()
                for ntype in self.rgcn.embed:
                    if len(output_nodes[ntype]):
                        y_others[ntype][output_nodes[ntype]] = h_dict[ntype].cpu()

        return y_user