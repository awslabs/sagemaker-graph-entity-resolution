import os
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train-edges', type=str, default='user_train_edges.csv')
    parser.add_argument('--test-edges', type=str, default='user_test_edges.csv')
    parser.add_argument('--transient-nodes', type=str, default='transient_nodes.csv')
    parser.add_argument('--transient-edges', type=str, default='transient_edges.csv')
    parser.add_argument('--website-nodes', type=str, default='website_nodes.csv')
    parser.add_argument('--website-group-edges', type=str, default='website_group_edges.csv')
    parser.add_argument('--mini-batch', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='use mini-batch training and sample graph')
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--embedding-size', type=int, default=64, help="embedding size for node embedding")
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--n-neighbors', type=int, default=100, help='number of neighbors to sample')
    parser.add_argument('--negative-sampling-rate' ,type=int, default=10, help='rate of negatively sampled edges')
    parser.add_argument('--n-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--n-layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')
    parser.add_argument('--regularization-param', type=float, default=5e-4, help='Weight for regularization of decoder')
    parser.add_argument('--grad-norm', type=float, default=1.0, help='norm to clip gradient to')

    return parser.parse_args()


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger