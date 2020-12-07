import argparse
import logging
import os
import json

import pandas as pd

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
    parser.add_argument('--urls', type=str, default='urls.csv', help='map fact id to urls')
    parser.add_argument('--titles', type=str, default='titles.csv', help='map fact id to url titles')
    parser.add_argument('--facts', type=str, default='facts.json', help='map user to list of facts')
    parser.add_argument('--logs', type=str, default='logs.csv', help='file to store output normalized log files')
    parser.add_argument('--primary-key', type=str, default='fid', help='id key that corresponds to url')
    return parser.parse_args()


def load_url_data(data_dir, urls_path, titles_path, primary_key):
    logging.info("Loading website urls from file: {}".format(os.path.join(data_dir, urls_path)))
    urls_df = pd.read_csv(os.path.join(data_dir, urls_path), header=None, names=[primary_key, 'urls'])
    logging.info("Loading website titles from file: {}".format(os.path.join(data_dir, titles_path)))
    titles_df = pd.read_csv(os.path.join(data_dir, titles_path), header=None, names=[primary_key, 'titles'])
    logging.info("Merging website urls with website titles")
    return urls_df.merge(titles_df, how='left', on=primary_key).fillna("").set_index(primary_key)


def merge_websites_with_user_visits(data_dir, facts, url_data, primary_key, output_dir, logs):
    with open(os.path.join(data_dir, facts)) as f_in:
        for i, line in enumerate(f_in):
            j = json.loads(line.strip())
            user_visits = pd.json_normalize(j.get("facts"))
            fids = user_visits[primary_key].values
            user_visits = pd.concat((user_visits.set_index(primary_key), url_data.loc[fids]), axis=1)
            user_visits['uid'] = j.get('uid')
            mode, header = ('w', True) if i == 0 else ('a', False)
            with open(os.path.join(output_dir, logs), mode) as f:
                user_visits.to_csv(f, index=False, header=header)


if __name__ == '__main__':
    logging = get_logger(__name__)

    args = parse_args()

    websites = load_url_data(args.data_dir, args.urls, args.titles, args.primary_key)
    logging.info("Obtained website info; merging with user visits")
    merge_websites_with_user_visits(args.data_dir, args.facts, websites, args.primary_key, args.output_dir, args.logs)
