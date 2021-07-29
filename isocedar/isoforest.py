from .experiment import AnomalyDetector


class ClassicIsoforestAnomalyDetector(AnomalyDetector):
    def __init__(self, title='Classic Isolation Forest', **kwargs):
        super().__init__(title)
        self.isoforest = IsolationForest(**kwargs)

    def train(self, data):
        return self.isoforest.fit(data)

    def score(self, data):
        return self.isoforest.score_samples(data)

    def observe(self, point, label):
        super().observe(point, label)
        # do nothing, it's classic, you know
        return False