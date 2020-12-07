import os
import dgl

from data import *


def construct_graph(training_dir, training_edges, transient_nodes, transient_edges, website_nodes, website_edges):

    def _full_path(f):
        return os.path.join(training_dir, f)

    edgelists, id_to_node = {}, {}

    # parse and add training edges
    training_edgelist, id_to_node = parse_edgelist(_full_path(training_edges), id_to_node,
                                                   source_type='user', sink_type='user')
    print("Read user -> user training edgelist from {}".format(_full_path(training_edges)))
    edgelists[('user', 'same_entity', 'user')] = training_edgelist
    edgelists[('user', 'same_entity_reversed', 'user')] = [(b, a) for a, b in training_edgelist]

    # parse and add transient edges
    transient_edgelist, id_to_node = parse_edgelist(_full_path(transient_edges), id_to_node,
                                                    source_type='user', sink_type='website')
    print("Read user -> website edgelist from {}".format(_full_path(transient_edges)))
    edgelists[('user', 'visits', 'website')] = transient_edgelist
    edgelists[('website', 'visited_by', 'user')] = [(b, a) for a, b in transient_edgelist]

    # parse and add website edges
    website_edgelist, id_to_node = parse_edgelist(_full_path(website_edges), id_to_node,
                                                  source_type='website', sink_type='domain')
    print("Read website -> domain edgelist from {}".format(_full_path(website_edges)))
    edgelists[('website', 'owned_by', 'domain')] = website_edgelist
    edgelists[('domain', 'owns', 'website')] = [(b, a) for a, b in website_edgelist]

    # get user features
    user_features, new_nodes = get_features(id_to_node['user'], _full_path(transient_nodes))
    print("Got user features from {}".format(_full_path(transient_nodes)))

    # add self relation to user nodes
    edgelists[('user', 'self_relation', 'user')] = [(u, u) for u in id_to_node['user'].values()]

    # get website features
    website_features = get_website_features(id_to_node['website'], _full_path(website_nodes))
    print("Got website features from {}".format(_full_path(website_nodes)))

    g = dgl.heterograph(edgelists)
    print("Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
            g.ntypes, g.canonical_etypes))
    print("Number of user nodes : {}".format(g.number_of_nodes('user')))

    reverse_etypes = {'same_entity': 'same_entity_reversed',
                      'same_entity_reversed': 'same_entity',
                      'visits': 'visited_by',
                      'visited_by': 'visits',
                      'owned_by': 'owns',
                      'owns': 'owned_by'
                      }

    print(g)

    return g, (user_features, website_features), id_to_node, reverse_etypes
