import numpy as np
import pandas as pd


def parse_edgelist(edges, id_to_node, source_type='user', sink_type='user'):
    """
    Parse an edgelist path file and return the edges as a list of tuple
    :param edges: path to comma separated file containing bipartite edges with header for edgetype
    :param id_to_node: dictionary containing mapping for node names(id) to dgl node indices
    :param source_type: type of the source node in the edge. defaults to 'user' if no header
    :param sink_type: type of the sink node in the edge. defaults to 'user' if no header.
    :return: (list, dict) a list containing edges of a single relationship type as tuples and updated id_to_node dict.
    """
    edge_list = []
    source_pointer, sink_pointer = 0, 0
    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")
            if i == 0:
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue
            source_node, id_to_node, source_pointer = _get_node_idx(id_to_node, source_type, source, source_pointer)
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(id_to_node, sink_type, sink, source_pointer)
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(id_to_node, sink_type, sink, sink_pointer)

            edge_list.append((source_node, sink_node))

    return edge_list, id_to_node


def _get_node_idx(id_to_node, node_type, node_id, ptr):
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1

    return node_idx, id_to_node, ptr


def get_features(id_to_node, node_features):
    """
    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_features: path to file containing node features
    :return: (np.ndarray, list) node feature matrix in order and new nodes not yet in the graph
    """
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())
    with open(node_features, "r") as fh:
        for line in fh:
            node_feats = line.strip().split(",")
            node_id = node_feats[0]
            feats = np.array(list(map(float, node_feats[1:])))
            features.append(feats)
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)

            indices.append(id_to_node[node_id])

    features = np.array(features).astype('float32')
    features = features[np.argsort(indices), :]
    return features, new_nodes


def get_website_features(id_to_node, website_features):
    """
    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param website_features: path to file containing website features
    :return: (np.ndarray) website feature matrix in order
    """
    features_df = pd.read_csv(website_features, header=None, index_col=0)
    features_df.reindex(sorted(features_df.index, key=lambda x: id_to_node[x]))
    return features_df.values.astype(np.float32)
