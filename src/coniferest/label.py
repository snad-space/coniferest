from enum import IntEnum

__all__ = ["Label"]


class Label(IntEnum):
    """
    Anomalous classification labels.

    Three types of labels:

      * -1 for anomalies, referenced either as `Label.ANOMALY` or as `Label.A`,
      * 0 for unknowns: `Label.UNKNOWN` or `Label.U`,
      * 1 for regular data: `Label.REGULAR` or `Label.R`.
    """

    ANOMALY = -1
    A = ANOMALY
    UNKNOWN = 0
    U = UNKNOWN
    REGULAR = 1
    R = REGULAR
