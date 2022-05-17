# process_result.py
import torch


class ProcessResult:
    def __init__(self, batch_loss: torch.tensor,
                 total_loss: float,
                 total_acc: int,
                 output: torch.tensor,
                 prediction: torch.tensor):

        self.batch_loss = batch_loss
        self.total_loss = total_loss
        self.total_acc = total_acc
        self.output = output
        self.prediction = prediction
