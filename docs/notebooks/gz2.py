from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from coniferest.aadforest import AADForest
from coniferest.datasets import Dataset, DevNetDataset
from coniferest.isoforest import IsolationForest
from coniferest.label import Label
from coniferest.pineforest import PineForest
from coniferest.session.oracle import OracleSession, create_oracle_session

class Compare:
    models = {
        'Isolation Forest': IsolationForest,
        'AAD': AADForest,
        'Pine Forest': PineForest,
    }

    def __init__(self, dataset: Dataset, *, iterations=100, n_jobs=-1, sampletrees_per_batch=1<<20):
        self.model_kwargs = {
            'n_trees': 128,
            'sampletrees_per_batch': sampletrees_per_batch,
            'n_jobs': n_jobs,
        }
        self.session_kwargs = {
            'data': dataset.data,
            'labels': dataset.labels,
            'max_iterations': iterations,
        }
        self.results = {}
        self.steps = np.arange(1, iterations + 1)
        self.total_anomaly_fraction = np.mean(dataset.labels == Label.A)

    def get_sessions(self, random_seed):
        model_kwargs = self.model_kwargs | {'random_seed': random_seed}

        return {
            name: create_oracle_session(model=model(**model_kwargs), **self.session_kwargs)
            for name, model in self.models.items()
        }

    def run(self, random_seeds):
        assert len(random_seeds) == len(set(random_seeds)), "random seeds must be different"
        
        results = defaultdict(dict)

        futures = []
        for random_seed in tqdm(random_seeds):
            sessions = self.get_sessions(random_seed)
            for name, session in sessions.items():
                session.run()
                anomalies = np.cumsum(np.array(list(session.known_labels.values())) == Label.A)
                results[name][random_seed] = anomalies

        self.results |= results
        return self

    def plot(self, dataset_name: str, savefig=False):
        plt.figure(figsize=(8, 6))
        plt.title(f'Dataset: {dataset_name}')

        for name, anomalies_dict in self.results.items():
            anomalies = np.stack(list(anomalies_dict.values()))
            q5, median, q95 = np.quantile(anomalies, [0.05, 0.5, 0.95], axis=0)

            plt.plot(self.steps, median, alpha=0.75, label=name)
            plt.fill_between(self.steps, q5, q95, alpha=0.5)

        plt.plot(self.steps, self.steps * self.total_anomaly_fraction, ls='--', color='grey',
                 label='Theoretical random')

        plt.xlabel('Iteration')
        plt.ylabel('Number of anomalies')
        plt.grid()
        plt.legend()
        if savefig:
            plt.savefig(f'{dataset_name}.pdf')

        return self

import pickle
from pathlib import Path

import pandas as pd

class GalaxyZoo2Dataset(Dataset):
    def __init__(self, path: Path, *, anomaly_class='Class6.1', anomaly_threshold=0.9):
        astronomaly = pd.read_parquet(path / "astronomaly.parquet")
        self.data = astronomaly.drop(columns=['GalaxyID', 'anomaly']).to_numpy().copy(order='C')
        ids = astronomaly['GalaxyID'].to_numpy()

        solutions = pd.read_csv(path / "training_solutions_rev1.csv", index_col="GalaxyID")
        anomaly = solutions[anomaly_class][ids] >= anomaly_threshold
        self.labels = np.full(anomaly.shape, Label.R)
        self.labels[anomaly] = Label.A


seeds = range(12, 212)

path = Path("/home/hombit/gz2")
dataset_obj = GalaxyZoo2Dataset(path)
compare_zoo = Compare(dataset_obj, iterations=100, n_jobs=1, sampletrees_per_batch=1<<16).run(seeds)
compare_zoo.plot("Galaxy Zoo 2 (Anything odd? 90%)", savefig=True)
with open("galaxyzoo2_compare.pickle", "wb") as fh:
    pickle.dump(compare_zoo, fh)
