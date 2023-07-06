import subprocess

import matplotlib.pyplot as plt
import numpy as np

from coniferest.datasets import Label


class TreeViz:
    """
    Tree vizualization with matplotlib or graphviz.
    """

    def __init__(self, tree, known_data=None, known_labels=None, dataset=None):
        self.tree = tree
        self.known_data = known_data
        self.known_labels = known_labels
        self.dataset = dataset

        if known_data is None:
            self.known_data = np.empty((0, tree.n_features))
            self.known_labels = np.empty((0,))

    @staticmethod
    def _node_label(reds, blues):
        if reds + blues == 0:
            return '""'

        n_float = np.sqrt((reds + blues) / 12)
        columns = int(np.ceil(n_float * 4))

        labels = ['<font color="red">&#x2605;</font>'] * reds + ['<font color="blue">&#x2605;</font>'] * blues
        text = []
        while len(labels) > 0:
            text.append("".join(labels[:columns]))
            labels = labels[columns:]

        return "<" + "<br/>".join(text) + ">"

    def _dot_walker(self, current, known_data, known_labels):
        "Walk trough the tree recursively and make a dot file for graphviz"

        tree = self.tree
        text = []
        if tree.children_left[current] != -1:
            text.append(f' n{current} [label=""];')
            # If there are children nodes, render them too.

            # We need to split data and labels
            index = known_data[:, tree.feature[current]] <= tree.threshold[current]

            # Left one
            left = tree.children_left[current]
            text.append(f" n{current} -- n{left};")
            text.extend(self._dot_walker(left, known_data[index, :], known_labels[index]))

            # And right one
            right = tree.children_right[current]
            text.append(f" n{current} -- n{right};")
            text.extend(self._dot_walker(right, known_data[~index, :], known_labels[~index]))
        else:
            reds = np.sum(known_labels == Label.ANOMALY)
            blacks = np.sum(known_labels == Label.REGULAR)
            label = self._node_label(reds, blacks)
            text.append(f" n{current} [label={label}];")

        return text

    def generate_dot(self):
        text = []
        text.append('graph ""')
        text.append("{")
        text.extend(self._dot_walker(0, self.known_data, self.known_labels))
        text.append("}")

        return "\n".join(text)

    def draw_graph(self):
        p = subprocess.Popen(["dot", "-Tpng"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        (image, _) = p.communicate(input=self.generate_dot().encode("utf-8"), timeout=10)
        return image

    def display_graph(self):
        "Display tree as a graph"
        image = self.draw_graph()

        import IPython.display

        return IPython.display.Image(image)

    def display_2d(self, full=False):
        "Display tree like a k2-tree"
        dataset = self.dataset
        known_data = self.known_data
        known_labels = self.known_labels
        if dataset is None:
            raise ValueError("no data to plot")

        if dataset.data.shape[1] != 2:
            raise ValueError("only 2d plots are supported")

        fig, ax = plt.subplots()

        frame = np.empty((2, 2))
        frame[0, :] = dataset.data.min(axis=0)
        frame[1, :] = dataset.data.max(axis=0)

        self._draw_subsets(ax, frame, 0)

        if full:
            ax.scatter(*dataset.data[dataset.labels == Label.R, :].T, color="blue", s=10, label="Normal")
            ax.scatter(*dataset.data[dataset.labels == Label.A, :].T, color="red", s=10, label="Anomaluous")
            ax.scatter(*known_data[known_labels == Label.R, :].T, color="blue", marker="*", s=80)
            ax.scatter(*known_data[known_labels == Label.A, :].T, color="red", marker="*", s=80)
        else:
            ax.scatter(*known_data[known_labels == Label.R, :].T, color="blue", marker="*", s=80, label="Normal")
            ax.scatter(*known_data[known_labels == Label.A, :].T, color="red", marker="*", s=80, label="Anomaluous")

        ax.set(xlabel="x1", ylabel="x2", xlim=frame[:, 0], ylim=frame[:, 1])
        ax.legend()

        return fig, ax

    def _draw_subsets(self, ax, frame, current):
        tree = self.tree
        if tree.feature[current] < 0:
            return

        threshold = tree.threshold[current]
        feature = tree.feature[current]
        if feature == 0:
            ax.plot([threshold, threshold], [frame[0, 1], frame[1, 1]], color="gray", zorder=1)
        else:
            ax.plot([frame[0, 0], frame[1, 0]], [threshold, threshold], color="gray", zorder=1)

        left_frame = frame.copy()
        left_frame[1, feature] = threshold
        self._draw_subsets(ax, left_frame, tree.children_left[current])

        right_frame = frame.copy()
        right_frame[0, feature] = threshold
        self._draw_subsets(ax, right_frame, tree.children_right[current])
