# model_tools.py

import numpy as np
import os
import pandas as pd
import torch

from pathlib import Path
from src.bert_dataset import Dataset
from src.config import Config
from src.evaluate_result import EvaluateResult
from src.misc_tools import MiscTools
from src.model_result import ModelResult
from src.process_result import ProcessResult

from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class ModelTools:
    """Model utilities for this repo"""

    @staticmethod
    def train_model(
            model: nn.Module,
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            test_data: pd.DataFrame,
            config: Config,
            best_val_loss_max_epochs: int = 10,
            soft_max: nn.Module = nn.Softmax(dim=1)
    ):
        train, val = Dataset(train_data, config.labels_dict, config.bert_model), \
                     Dataset(val_data, config.labels_dict, config.bert_model)

        # An arbitrary number of epochs where validation loss is considered to have stopped improving
        # if no further improvement shown after n epochs, at which point training will be stopped

        best_val_loss_epochs = 0

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=config.batch_size)

        device = config.device

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        if config.use_cuda:
            criterion = criterion.cuda()

        results = []

        lowest_loss_val = 1000.0
        saved_model_train_result = ModelResult
        saved_model_paths = []
        saved_train_probabilities_by_label = []
        saved_val_probabilities_by_label = []

        for epoch in range(config.num_epochs):

            best_val_loss_epochs += 1

            total_acc_train = 0
            total_loss_train = 0

            train_probabilities = []
            val_probabilities = []

            for train_input, train_label in tqdm(train_dataloader):
                pr = \
                    ModelTools.process_model(
                        text_input=train_input, label=train_label,
                        device=device, model=model,
                        total_loss=total_loss_train, total_acc=total_acc_train, criterion=criterion)

                total_loss_train, total_acc_train = pr.total_loss, pr.total_acc

                model.zero_grad()
                pr.batch_loss.backward()
                optimizer.step()

                train_probabilities = ModelTools.append_probabilities_and_label(
                    process_result=pr, soft_max=soft_max, labels=train_label, probabilities=train_probabilities
                )

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    pr = \
                        ModelTools.process_model(
                            text_input=val_input, label=val_label,
                            device=device, model=model,
                            total_loss=total_loss_val, total_acc=total_acc_val, criterion=criterion)

                    total_loss_val, total_acc_val = pr.total_loss, pr.total_acc

                    val_probabilities = ModelTools.append_probabilities_and_label(
                        process_result=pr, soft_max=soft_max, labels=val_label, probabilities=val_probabilities
                    )

            train_result = ModelResult(
                epoch + 1,
                total_loss_train / len(train_data), total_acc_train / len(train_data),
                total_loss_val / len(val_data), total_acc_val / len(val_data))

            results.append([train_result.epoch,
                            train_result.train_loss,
                            train_result.train_accuracy,
                            train_result.validation_loss,
                            train_result.validation_accuracy])

            print(
                f'\nEpoch: {train_result.epoch} | Train Loss: {train_result.train_loss} \
                    | Train Accuracy: {train_result.train_accuracy} \
                    | Val Loss: {train_result.validation_loss} \
                    | Val Accuracy: {train_result.validation_accuracy}')

            # Save best model so far, going by lowest validation loss. Delete previous saved model if it exists.
            if epoch >= 2 and total_loss_val / len(val_data) < lowest_loss_val:
                # Have best val loss so far, so reset counter for epochs since best validation
                best_val_loss_epochs = 0

                for path in saved_model_paths:
                    os.remove(path)
                lowest_loss_val = total_loss_val / len(val_data)
                saved_model_paths = MiscTools.save_trained_model(epoch=epoch, model=model, config=config)

                # Trial state dictionary against separate test data
                saved_model_train_result = ModelTools.get_and_save_test_result(
                    model=model, df_test=test_data, config=config, model_name=Path(saved_model_paths[1]).name,
                    model_result=train_result)

                saved_train_probabilities_by_label = train_probabilities
                saved_val_probabilities_by_label = val_probabilities

            if best_val_loss_epochs >= best_val_loss_max_epochs:
                # Exit for loop
                break

        result_columns = ['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy']
        # Round results
        results = [[round(x, 4) for x in row] for row in results]
        results.insert(0, result_columns)

        return results, saved_model_train_result, saved_train_probabilities_by_label, saved_val_probabilities_by_label

    @staticmethod
    def evaluate(
            model: nn.Module,
            eval_data: pd.DataFrame,
            config: Config,
            soft_max: nn.Module = nn.Softmax(dim=1)
    ) -> EvaluateResult:

        evaluation = Dataset(eval_data, config.labels_dict, config.bert_model)

        # Hard-coding batch size to 4. For the size of data currently in use, doesn't matter much.
        eval_dataloader = torch.utils.data.DataLoader(evaluation, 4, shuffle=False)

        predictions = []
        eval_probabilities = []

        total_acc_eval = 0

        device = config.device

        with torch.no_grad():
            for eval_input, eval_label in eval_dataloader:
                eval_label = eval_label.to(device)
                mask = eval_input['attention_mask'].to(device)
                input_id = eval_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                acc = (output.argmax(dim=1) == eval_label).sum().item()
                total_acc_eval += acc

                pr = ProcessResult(None, None, None, output, output.argmax(dim=1))

                eval_probabilities = ModelTools.append_probabilities_and_label(
                    process_result=pr, soft_max=soft_max, labels=eval_label, probabilities=eval_probabilities
                )

                predictions.extend(output.argmax(dim=1).tolist())

        predictions = pd.DataFrame(predictions, columns=['category_index'])

        predictions['category'] = eval_data['category']
        predictions['predicted_category'] = \
            predictions.apply(lambda row:
                              [label for label in config.labels_dict
                               if config.labels_dict[label] == row.category_index][0], axis=1)
        predictions['is_match'] = \
            predictions.apply(lambda row:
                              row.predicted_category == row.category, axis=1)

        predictions = predictions.reindex(columns=['predicted_category', 'is_match'])

        predictions = pd.concat([eval_data, predictions], axis=1)

        eval_accuracy = total_acc_eval / len(eval_data)
        print(f'Evaluation accuracy: {eval_accuracy: .3f}')

        results = EvaluateResult(predictions, eval_accuracy, eval_probabilities)
        return results

    @staticmethod
    def process_model(text_input: dict, label: torch.tensor,
                      device: str, model: nn.Module,
                      total_loss: int, total_acc: int,
                      criterion: nn.Module) -> ProcessResult:
        label = label.to(device)
        mask = text_input['attention_mask'].to(device)
        text_input_id = text_input['input_ids'].squeeze(1).to(device)

        output = model(text_input_id, mask)

        label = label.to(torch.int64)

        batch_loss = criterion(output, label)
        total_loss += batch_loss.item()

        acc = (output.argmax(dim=1) == label).sum().item()
        total_acc += acc

        process_result = ProcessResult(batch_loss, total_loss, total_acc, output, output.argmax(dim=1))

        return process_result

    @staticmethod
    def get_and_save_test_result(model: nn.Module, df_test: pd.DataFrame, config: Config, model_name: str,
                                 model_result: ModelResult):
        # Save result, overwriting previous one if it exists
        model_result.test_accuracy = ModelTools.evaluate(model, df_test, config).accuracy

        save_path = os.path.join(config.output_dir, 'model_test_results.txt')
        with open(save_path, 'w', encoding='utf-8', newline='') as outfile:
            header = 'ModelName, Epoch, ValidationLoss, ValidationAccuracy, TestAccuracy'
            data = f'{model_name},{model_result.epoch},' \
                   f'{model_result.validation_loss},' \
                   f'{model_result.validation_accuracy},' \
                   f'{model_result.test_accuracy}'
            outfile.write(f'{header}\n{data}\n')

        return model_result

    @staticmethod
    def append_probabilities_and_label(process_result: ProcessResult, soft_max: nn.Module, labels: torch.tensor,
                                       probabilities: list) -> list:
        for j in range(len(process_result.output)):
            probs = list(np.round(soft_max(process_result.output)[j].tolist(), 4))
            probs.append(labels[j].item())
            probs.append(process_result.prediction[j].item())
            probabilities.append(probs)

        return probabilities
