from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class InputData:
    """INPUT DATA"""
    x: np.ndarray
    input_y: np.ndarray
    normalized_y: np.ndarray


@dataclass(frozen=True)
class ApproxResults:
    """RESULTS of approximations - both FW and INV"""
    x: np.ndarray
    upper_y: np.ndarray
    bottom_y: np.ndarray
    

@dataclass(frozen=True)
class FuzzaproxResult:
    """FINAL RETURN Data Class"""
    input_data: InputData
    forward: ApproxResults
    inverse: ApproxResults
