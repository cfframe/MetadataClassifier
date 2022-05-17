# model_result.py

class ModelResult:
    def __init__(self, epoch: int = -1,
                 train_loss: float = 0.0, train_accuracy: float = 0.0,
                 validation_loss: float = 0.0, validation_accuracy: float = 0.0,
                 test_accuracy: float = 0.0):

        self.epoch = epoch
        self.train_loss = round(train_loss, 3)
        self.train_accuracy = round(train_accuracy, 3)
        self.validation_loss = round(validation_loss, 3)
        self.validation_accuracy = round(validation_accuracy, 3)
        self.test_accuracy = round(test_accuracy, 3)

        self.list = [self.epoch,
                     self.train_loss, self.train_accuracy,
                     self.validation_loss, self.validation_accuracy,
                     self.test_accuracy]
