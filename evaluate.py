"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import megengine as mge
import megengine.distributed as dist
from megengine.distributed.functional import all_reduce_sum
import megengine.random as rand
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (megengine.module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        data_batch, labels_batch = mge.Tensor(data_batch), mge.Tensor(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from megengine Tensor, move to cpu, convert to numpy arrays
        output_batch = output_batch.numpy()
        labels_batch = labels_batch.numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def evaluate_dist(rank, model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        rank: the process rank
        model: (megengine.module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        data_batch, labels_batch = mge.Tensor(data_batch), mge.Tensor(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from megengine Tensor, move to cpu, convert to numpy arrays
        output_batch = output_batch.numpy()
        labels_batch = labels_batch.numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: all_reduce_sum(
                            mge.Tensor(
                            np.mean([x[metric] for x in summ]))).numpy() / dist.get_world_size() for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    if rank == 0:
        logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    rand.seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)