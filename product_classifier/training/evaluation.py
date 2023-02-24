from typing import Tuple, List

from tqdm import tqdm
import itertools
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from product_classifier.dataset.dataset import AmazonDataset
from product_classifier.models.product_classifier import ProductClassifier


def _get_labels_logits_and_predictions(net: ProductClassifier, data_loader: DataLoader) -> Tuple[Tensor, Tensor, Tensor]:
    net.eval()
    all_labels = torch.empty(0, dtype=torch.int)
    all_predictions = torch.empty(0, dtype=torch.int)
    all_logits = torch.empty(0, dtype=torch.int)

    with torch.no_grad():
        for x, y in tqdm(data_loader):
            logits = net(x)
            all_logits = torch.cat((all_logits, logits), dim=0)
            all_predictions = torch.cat((all_predictions, torch.argmax(logits, dim=1)), dim=0)
            all_labels = torch.cat((all_labels, y), dim=0)
    return all_labels, all_logits, all_predictions


def run_evaluation(net: ProductClassifier, dataloader: DataLoader) -> Tuple[float, float]:
    all_labels, all_logits, all_predictions = _get_labels_logits_and_predictions(net, dataloader)

    accuracy = torch.mean(torch.eq(all_predictions, all_labels).float()).item()
    loss = F.nll_loss(F.log_softmax(all_logits, dim=1), all_labels).item()
    return accuracy, loss


def display_class_distribution_chart(dataset: AmazonDataset, split_name: str):
    class_names = []
    proportions = []
    for i in range(len(dataset.class_distribution)):
        class_names.append(list(dataset.class_distribution.keys())[i])
        proportions.append(list(dataset.class_distribution.values())[i])
    x_pos = [i for i, _ in enumerate(class_names)]

    plt.bar(x_pos, proportions, color='green')
    plt.rcParams.update({'font.size': 8})
    plt.xlabel(f'Classes ({len(dataset.class_distribution)} total)')
    plt.ylabel("Proportion")
    plt.title(f'Class distribution of {split_name} (dataset size: {len(dataset)})')

    plt.xticks(x_pos, class_names)
    plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='center')
    plt.tight_layout(pad=2)
    plt.show()


def display_confusion_matrix(net: ProductClassifier, dataloader: DataLoader, class_names: List[str]) -> None:
    all_labels, all_logits, all_predictions = _get_labels_logits_and_predictions(net, dataloader)
    c_matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(len(class_names))))
    _plot_confusion_matrix(c_matrix, class_names)


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(26, 26))
    plt.rcParams['font.size'] = '18'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=3)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
