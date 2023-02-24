import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np

from product_classifier.config.training import ConfigTraining
from product_classifier.models.product_classifier import ProductClassifier
from product_classifier.training.evaluation import run_evaluation


def run_training_loop(net: ProductClassifier,
                      optimiser: Optimizer,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int,
                      model_output_dir: str
                      ) -> float:

    best_val_loss = np.inf
    tensorboard_writer = tensorboard.SummaryWriter(log_dir=model_output_dir)

    patience_count = 0
    for epoch in range(epochs):
        net.train()
        for x, y in tqdm(train_loader):
            net.zero_grad()
            logits = net(x)  # x[0] is product image, x[1] is product title
            loss = F.nll_loss(F.log_softmax(logits, dim=1), y)
            loss.backward()
            optimiser.step()

        net.eval()
        train_accuracy, train_loss = run_evaluation(net, train_loader)
        val_accuracy, val_loss = run_evaluation(net, val_loader)

        # Log losses and metrics
        tensorboard_writer.add_scalar('Train Loss', train_loss, epoch)
        tensorboard_writer.add_scalar('Train accuracy', train_accuracy, epoch)
        tensorboard_writer.add_scalar('Val Loss', val_loss, epoch)
        tensorboard_writer.add_scalar('Val accuracy', val_accuracy, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(model_output_dir, "model_state_dict.pkl"))
            patience_count = 0
        else:
            patience_count += 1

        if patience_count > ConfigTraining.patience:
            print(f'Stopping training early here as val_loss failed to improve for {ConfigTraining.patience} epochs.')
            break

        print('Epoch ', epoch + 1, '^')
    tensorboard_writer.close()
    return best_val_loss
