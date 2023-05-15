from enum import IntEnum


class Label(IntEnum):
    """
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
