import argparse
import logging
import os

import pandas as pd
import numpy as np
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--logs', type=str, default='logs.csv', help='log of transient id web activity')
    parser.add_argument('--train', type=str, default='train.csv', help='pairs of transient ids that are the same user')
    parser.add_argument('--num-sample-src', type=int, default=20000, help='number of src nodes to sample')
    return parser.parse_args()

def sample_train(data_dir, output_dir, train_file, sample_src):
    train = pd.read_csv(os.path.join(data_dir, train_file), header=None)
    sampled_train_srcs = pd.DataFrame({0:np.random.choice(train[0].unique(), sample_src, replace=False)})
    sampled_train = train.merge(sampled_train_srcs, how='inner', on=[0])

    initial_node_count = len(pd.concat([train[0], train[1]]).unique())
    final_nodes = pd.concat([sampled_train[0], sampled_train[1]]).unique()
    final_node_count = len(final_nodes)
    print("Sampled {} train edges from original train set of size {}".format(sampled_train.shape[0], train.shape[0]))
    print("{} unique nodes sampled from a set of size {}".format(final_node_count, initial_node_count))

    with open(os.path.join(output_dir, train_file), 'w') as f:
        sampled_train.to_csv(f, index=False, header=False)

    return final_nodes


def reduce_logs(data_dir, output_dir, log_file, ids):
    logs = pd.read_csv(os.path.join(data_dir, log_file))
    reduced_logs = logs.merge(pd.DataFrame({'uid':ids}), how='inner', on='uid')

    with open(os.path.join(output_dir, log_file), 'w') as f:
        reduced_logs.to_csv(f, index=False, header=True)

if __name__ == '__main__':

    args = parse_args()
    node_ids = sample_train(args.data_dir, args.output_dir, args.train, args.num_sample_src)
    reduce_logs(args.data_dir, args.output_dir, args.logs, node_ids)