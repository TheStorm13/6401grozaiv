from dataclasses import dataclass

import numpy as np


@dataclass
class Image:
    filename: str
    extension: str
    data: np.ndarray
