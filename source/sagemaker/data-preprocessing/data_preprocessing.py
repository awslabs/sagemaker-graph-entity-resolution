import argparse
import logging
import os

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
np.random.seed(0)

MIN_TIMESTAMP = 1461340800 #2016, 04, 23
MAX_TIMESTAMP = 1466611200 #2016, 06, 23


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--logs', type=str, default='logs.csv', help='log of transient id web activity')
    parser.add_argument('--train', type=str, default='train.csv', help='pairs of transient ids that are the same user')
    parser.add_argument('--test-ratio', type=float, default=0, help='fraction of train data to use for test')
    return parser.parse_args()


def prepare_ground_truth_links(data_dir, train_file, test_ratio, output_dir):
    logging.info("Reading train data edges from: {}".format(os.path.join(data_dir, train_file)))
    train = pd.read_csv(os.path.join(data_dir, train_file), header=None)
    if test_ratio:
        logging.info("Partitioning train data edges into train and test set with test ratio: {}".format(test_ratio))
        train, test = train_test_split(train, test_size=test_ratio)
        logging.info("Saving test set to {}".format(os.path.join(output_dir, 'user_test_edges.csv')))
        with open(os.path.join(output_dir, 'user_test_edges.csv'), 'w') as f:
            test.to_csv(f, index=False, header=False)
    logging.info("Saving train set to {}".format(os.path.join(output_dir, 'user_train_edges.csv')))
    with open(os.path.join(output_dir, 'user_train_edges.csv'), 'w') as f:
        train.to_csv(f, index=False, header=False)


def process_logs(data_dir, log_file, output_dir):
    logs = pd.read_csv(os.path.join(data_dir, log_file))
    logging.info("Read user website visit logs from: {}".format(os.path.join(data_dir, log_file)))

    transient_edges = os.path.join(output_dir, 'transient_edges.csv')
    save_file(logs[['uid', 'urls']].drop_duplicates(), transient_edges,
              "Saved user -> url transient edges to {}".format(transient_edges))

    transient_nodes_file = os.path.join(output_dir, 'transient_nodes.csv')
    user_features = get_user_features(logs[['uid', 'ts']])
    save_file(user_features, transient_nodes_file, "Saved transient user features to {}".format(transient_nodes_file))

    website_nodes_file = os.path.join(output_dir, 'website_nodes.csv')
    website_features = get_website_features(logs[['urls', 'titles']].drop_duplicates().fillna(""))
    save_file(website_features, website_nodes_file, "Saved website_features to {}".format(website_nodes_file))

    website_group_file = os.path.join(output_dir, 'website_group_edges.csv')
    website_groupings = get_website_groupings(logs[['urls']].drop_duplicates())
    save_file(website_groupings, website_group_file, "Saved url -> domain edges to {}".format(website_group_file))


def get_user_features(user_data):
    logging.info("Number of unique users is {}".format(len(user_data['uid'].unique())))
    logging.info("User data has shape {}, columns: {} before transformation".format(user_data.shape, user_data.columns))
    user_data['ts'] = user_data['ts'].apply(preprocess_timestamp)
    user_data = user_data.drop(user_data[(user_data['ts'] < MIN_TIMESTAMP) | (user_data['ts'] > MAX_TIMESTAMP)].index)

    user_features = np.zeros((user_data.shape[0], 7*24))
    user_features[np.arange(user_features.shape[0]), user_data['ts'].apply(get_activity_index)] = 1

    logging.info("User data has shape {} after transformation".format(user_features.shape))
    user_features_df = pd.DataFrame(user_features)
    user_features_df['uid'] = user_data['uid'].values
    final_user_feature =  user_features_df.groupby('uid').sum().reset_index()
    logging.info("Final user features shape {}".format(final_user_feature.shape))
    return final_user_feature


def preprocess_timestamp(ts):
    if ts > 9999999999:
        ts = ts / 1000
    if ts > 9999999999:
        ts = ts / 1000
    return ts


def get_activity_index(ts):
    dt = datetime.fromtimestamp(ts)
    return dt.weekday()*24 + dt.hour


def get_website_features(web_data):
    logging.info("Web data has shape {}, columns: {} before transformation".format(web_data.shape, web_data.columns))
    split_url = lambda url: " " + " ".join(url.split("/")[:3])
    transform_pipeline = Pipeline([('tf_idf', TfidfVectorizer()), ('dim_reduce', TruncatedSVD(n_components=20))])
    web_features = transform_pipeline.fit_transform(web_data['titles'].values+web_data['urls'].apply(split_url).values)
    logging.info("Web data has shape {} after transformation".format(web_features.shape))
    web_features_df = pd.DataFrame(web_features)
    web_features_df.insert(0, 'urls', web_data['urls'].values)
    return web_features_df


def get_website_groupings(urls):
    urls['domain'] = urls['urls'].apply(lambda x: x.split("/")[0])
    return urls


def save_file(df, file_name, message):
    with open(file_name, 'w') as f:
        df.to_csv(f, index=False, header=False)
    logging.info(message)


if __name__ == '__main__':
    logging = get_logger(__name__)

    args = parse_args()

    prepare_ground_truth_links(args.data_dir, args.train, args.test_ratio, args.output_dir)
    process_logs(args.data_dir, args.logs, args.output_dir)