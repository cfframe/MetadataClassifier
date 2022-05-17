# evaluate_result.py
import pandas as pd


class EvaluateResult:
    def __init__(self, predictions: pd.DataFrame,
                 accuracy: str,
                 probabilities: list):

        self.predictions = predictions
        self.accuracy = f'{accuracy: .3f}'
        self.probabilities = probabilities
