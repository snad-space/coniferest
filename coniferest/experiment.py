import io
from itertools import count

import numpy as np

import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas  # noqa

from PIL import Image

from .datasets import Label


class AnomalyDetector:
    def __init__(self, title):
        """
        Base class of anomaly detectors for different forests.
        Anomaly detectors are the wrappers of forests for interaction with
        the AnomalyDetectionExperiment.

        Parameters
        ----------
        title
            Title for different performance plots.
        """
        self.title = title
        self.known_data = None
        self.known_labels = None

    def train(self, data):
        raise NotImplementedError('abstract method called')

    def score(self, data):
        raise NotImplementedError('abstract method called')

    def observe(self, point, label):
        """
        Record the new data point.

        Parameters
        ----------
        point
            Data features.

        label
            Data label.

        Returns
        -------
        bool, was the forest updated?
        """
        if self.known_data is None:
            self.known_data = np.reshape(point, (-1, len(point)))
            self.known_labels = np.reshape(label, (-1,))
        else:
            self.known_data = np.vstack((self.known_data, point))
            self.known_labels = np.hstack((self.known_labels, label))

        return False


class AnomalyDetectionExperiment:
    COLORS = {Label.ANOMALY: 'red', Label.UNKNOWN: 'grey', Label.REGULAR: 'blue'}

    def __init__(self, regressor, data_features, data_labels, capacity=300):
        """
        Perform an experiment of anomaly detection with the expert.

        Parameters
        ----------
        regressor
            The forest to detect anomalies with.

        data_features
            Training data. Array of features.

        data_labels
            Labels for the data.

        capacity
            Maximum number of the iterations to perform. 300 by default.
        """
        self.regressor = regressor
        self.capacity = capacity
        self.data_features = data_features
        self.data_labels = data_labels
        self.trajectory = None
        self.trace = None

    def run(self):
        """
        Run the experiment.

        Returns
        -------
        Trajectory. The ndarray with indices of the explored data.
        """
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
            if data_labels[outlier] == Label.REGULAR:
                n_misses += 1

            # ... and observe it
            recalculate = regressor.observe(data_features[outlier], data_labels[outlier])

        self.trajectory = np.fromiter(knowns, dtype=int)
        self.trace = list(knowns.values())
        return self.trajectory

    def draw_cartoon(self):
        """
        Draw a animation how regressor performs.

        Returns
        -------
        List of PIL images.
        """
        if self.trajectory is None:
            self.run()

        data_features = self.data_features
        data_labels = self.data_labels
        COLORS = self.COLORS

        images = []
        for i, trace in zip(count(), self.trace):
            fig = Figure()
            canvas = FigureCanvas(fig)

            ax = fig.subplots()
            ax.set(title=f'{self.regressor.title}, iteration {i}',
                   xlabel='x1', ylabel='x2')

            ax.scatter(*data_features.T, color=COLORS[Label.REGULAR], s=10)
            ax.scatter(*data_features[trace, :].T, color=COLORS[Label.ANOMALY], s=10)

            prehistory = self.trajectory[:i]
            index = data_labels[prehistory] == Label.ANOMALY
            if np.any(index):
                ax.scatter(*data_features[prehistory[index], :].T, marker='*', color=COLORS[Label.ANOMALY], s=80)

            index = ~index
            if np.any(index):
                ax.scatter(*data_features[prehistory[index], :].T, marker='*', color=COLORS[Label.REGULAR], s=80)

            ax.scatter(*data_features[self.trajectory[i], :].T, marker='*', color='k', s=80)

            normal_patch = mpatches.Patch(color=COLORS[Label.REGULAR], label='Regular')
            anomalous_patch = mpatches.Patch(color=COLORS[Label.ANOMALY], label='Anomalous')
            ax.legend(handles=[normal_patch, anomalous_patch], loc='lower left')

            canvas.draw()
            size = (int(canvas.renderer.width), int(canvas.renderer.height))
            s = canvas.tostring_rgb()
            image = Image.frombytes('RGB', size, s)

            images.append(image)
            del canvas
            del fig

        return images

    def save_cartoon(self, file):
        """
        (Draw and) save a cartoon.

        Parameters
        ----------
        file
            Filename or file object to write GIF file to.

        Returns
        -------
        None
        """
        images = self.draw_cartoon()
        images[0].save(file, format='GIF',
                       save_all=True, append_images=images[1:],
                       optimize=False, duration=500, loop=0)

    def display_cartoon(self):
        """
        IPython display of the drawn GIF.

        Returns
        -------
        None
        """
        import IPython.display

        with io.BytesIO() as buffer:
            self.save_cartoon(buffer)
            return IPython.display.Image(buffer.getvalue())
