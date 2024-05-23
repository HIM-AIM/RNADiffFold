# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([1 - alpha, alpha])
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        at = self.alpha.gather(0, targets.data.view(-1)).reshape(inputs.shape)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def bce_loss(preds, targets):
    device = preds.device
    pos_weight = torch.Tensor([300]).to(device)
    criterion_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion_weighted(preds, targets)


def calculate_mattews_correlation_coefficient(preds, targets):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = torch.sum(preds * targets)
    tn = torch.sum((1 - preds) * (1 - targets))
    fp = torch.sum(preds * (1 - targets))
    fn = torch.sum((1 - preds) * targets)
    mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc.cpu().item()


# return auc scores
def calculate_auc(targets, preds):

    return roc_auc_score(targets.reshape(-1).cpu().numpy(), preds.reshape(-1).cpu().numpy(), average='micro')


def calculate_auc_fpr_tpr(targets, preds):
    # return auc scores, fpr, tpr
    fpr, tpr, thresholds = roc_curve(targets.reshape(-1).cpu().numpy(), preds.reshape(-1).cpu().numpy())
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr


def evaluate_f1_precision_recall(preds, targets, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(preds) * torch.Tensor(targets))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(preds)).sum()
    true_p = targets.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return precision, recall, f1_score


def rna_evaluation(preds, targets, eps=1e-11):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = torch.sum(preds * targets)
    tn = torch.sum((1 - preds) * (1 - targets))
    fp = torch.sum(preds * (1 - targets))
    fn = torch.sum((1 - preds) * targets)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps) # accuracy
    prec = (tp + eps) / (tp + fp + eps)  # precision
    recall = (tp + eps) / (tp + fn + eps)  # recall
    sens = (tp + eps) / (tp + fn + eps)  # senstivity
    spec = (tn + eps) / (tn + fp + eps)  # spec

    F1 = 2 * ((prec * sens) / (prec + sens))
    MCC = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return accuracy, prec, recall, sens, spec, F1, MCC.cpu().item()


def loglik_nats(model, x):
    """Compute the log-likelihood in nats."""
    return - model.log_prob(x).mean()


def loglik_bpd(model, x, data_fcn_2, data_seq_raw, set_max_len, contact_masks):
    """Compute the log-likelihood in bits per dim."""
    return - model.log_prob(x, data_fcn_2, data_seq_raw, set_max_len, contact_masks).sum() / (math.log(2) * x.shape.numel())


def elbo_nats(model, x):
    """
    Compute the ELBO in nats.
    Same as .loglik_nats(), but may improve readability.
    """
    return loglik_nats(model, x)


def elbo_bpd(model, x, data_fcn_2, data_seq_raw, set_max_len, contact_masks):
    """
    Compute the ELBO in bits per dim.
    Same as .loglik_bpd(), but may improve readability.
    """
    return loglik_bpd(model, x, data_fcn_2, data_seq_raw, set_max_len, contact_masks)
