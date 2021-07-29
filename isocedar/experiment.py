import io
from itertools import count

import numpy as np

import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas  # noqa

from PIL import Image


class AnomalyDetector:
    def __init__(self, title):
        self.title = title
        self.known_data = None
        self.known_labels = None

    def train(self, data):
        raise NotImplementedError('abstract method called')

    def score(self, data):
        raise NotImplementedError('abstract method called')

    def observe(self, point, label):
        if self.known_data is None:
            self.known_data = np.reshape(point, (-1, len(point)))
            self.known_labels = np.reshape(label, (-1,))
        else:
            self.known_data = np.vstack((self.known_data, point))
            self.known_labels = np.hstack((self.known_labels, label))

        return False


class AnomalyDetectionExperiment:
    COLORS = {-1: 'red', 1: 'blue'}

    def __init__(self, regressor, data_features, data_labels, capacity=300):
        self.regressor = regressor
        self.capacity = capacity
        self.data_features = data_features
        self.data_labels = data_labels
        self.trajectory = None
        self.trace = None

    def run(self):
        regressor = self.regressor
        data_features = self.data_features
        data_labels = self.data_labels

        # Indices of all the anomalies we are going to detect
        anomalies_list, = np.where(self.data_labels == -1)
        anomalies = set(anomalies_list)
        n_anomalies = len(anomalies)

        # Known labels. Dict preserves order, so we have full history.
        # The values are the currect outliers with correction for missed points.
        knowns = {}
        n_misses = 0

        # Train before doing anything
        regressor.train(data_features)

        # Should we recalculate scores now?
        recalculate = True
        ordering = None
        outlier = None
        while not anomalies.issubset(knowns) and len(knowns) < self.capacity:
            # Calculate the scores
            if recalculate:
                scores = regressor.score(data_features)
                ordering = np.argsort(scores)

            # Find the most anomalous unknown object
            for outlier in ordering:
                if outlier not in knowns:
                    break

            # Keep the anomaly predictions at each point
            knowns[outlier] = ordering[:n_anomalies + n_misses]
            if data_labels[outlier] == 1:
                n_misses += 1

            # ... and observe it
            recalculate = regressor.observe(data_features[outlier], data_labels[outlier])

        self.trajectory = np.fromiter(knowns, dtype=int)
        self.trace = list(knowns.values())
        return self.trajectory

    def draw_cartoon(self):
        if self.trajectory is None:
            self.run()

        data_features = self.data_features
        data_labels = self.data_labels
        COLORS = self.COLORS

        images = []
        for i, outlier, trace in zip(count(), self.trajectory, self.trace):
            fig = Figure()
            canvas = FigureCanvas(fig)

            ax = fig.subplots()
            ax.set(title=f'{self.regressor.title}, iteration {i}',
                   xlabel='x1', ylabel='x2')

            ax.scatter(*data_features.T, color=COLORS[1], s=10)
            ax.scatter(*data_features[trace, :].T, color=COLORS[-1], s=10)

            prehistory = self.trajectory[:i]
            index = data_labels[prehistory] == -1
            if np.any(index):
                ax.scatter(*data_features[prehistory[index], :].T, marker='*', color=COLORS[-1], s=80)

            index = ~index
            if np.any(index):
                ax.scatter(*data_features[prehistory[index], :].T, marker='*', color=COLORS[1], s=80)

            ax.scatter(*data_features[self.trajectory[i], :].T, marker='*', color='k', s=80)

            normal_patch = mpatches.Patch(color=COLORS[1], label='Normal')
            anomalous_patch = mpatches.Patch(color=COLORS[-1], label='Anomalous')
            ax.legend(handles=[normal_patch, anomalous_patch], loc='lower left')

            canvas.draw()
            size = (int(canvas.renderer.width), int(canvas.renderer.height))
            s = canvas.tostring_rgb()

            image = Image.frombytes('RGB', size, canvas.tostring_rgb())
            images.append(image)
            del canvas
            del fig

        return images

    def save_cartoon(self, file):
        images = self.draw_cartoon()
        images[0].save(file, format='GIF',
                       save_all=True, append_images=images[1:],
                       optimize=False, duration=500, loop=0)

    def display_cartoon(self):
        import IPython.display

        with io.BytesIO() as buffer:
            self.save_cartoon(buffer)
            return IPython.display.Image(buffer.getvalue())

    def plot_performance(self):
        raise NotImplementedError("sorry, don't know how to plot performance yet")