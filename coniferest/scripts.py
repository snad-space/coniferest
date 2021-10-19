import argparse
import numpy as np
import pandas as pd

from coniferest.pineforest import PineForest, PineForestAnomalyDetector
from coniferest.experiment import AnomalyDetectionExperiment
from coniferest.datasets import Dataset


class PlasticcDataset(Dataset):
    def __init__(self, datafile, labelsfile):
        data_df = pd.read_csv(datafile)
        labels_df = pd.read_csv(labelsfile)
    
        lc_data = data_df.loc[:, 'g-20':'i+100'].to_numpy()
        lc_data_norm = np.amax(lc_data, axis=1).reshape(-1, 1)
        theta_data = data_df.loc[:, 'log_likehood':'theta_8'].to_numpy()
        theta_data_norm = np.amax(theta_data, axis=0) - np.amin(theta_data, axis=0)
        theta_data = theta_data / theta_data_norm
        data = np.hstack([lc_data / lc_data_norm, -2.5 * np.log10(lc_data_norm), theta_data])

        labels = labels_df['is_anomaly'].to_numpy()
        labels[labels == 1] = -1
        labels[labels == 0] = 1

        super(PlasticcDataset, self).__init__(data.copy(order='C'), labels)
        self.ids = data_df['object_id'].to_numpy()


def pinead_argparser():
    parser = argparse.ArgumentParser(description='Run pine forest AD simulation on PLAsTiCC data')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='CSV file with curves and features')
    parser.add_argument('-l', '--labels', type=str, required=True,
                        help='CSV file with labels for data')
    parser.add_argument('-o', '--output', type=str,
                        help='Output filename to save anomaly guess trajectory in')
    return parser


def pinead():
    args = pinead_argparser().parse_args()
    dataset = PlasticcDataset(args.data, args.labels)

    pineforest = PineForest(n_spare_trees=900)
    pineforest_det = PineForestAnomalyDetector(pineforest)
    pineforest_exp = AnomalyDetectionExperiment(pineforest_det, dataset.data, dataset.labels, capacity=1000)
    pineforest_exp.run()
    index = pineforest_exp.trajectory

    labels = dataset.labels[index]
    labels[labels == 1] = 0
    labels[labels == -1] = 1
    d = {'object_id': dataset.ids[index], 'is_anomaly': labels}
    df = pd.DataFrame(data=d)
    s = df.to_csv(args.output, index=False)
    if s is not None:
        print(s)

