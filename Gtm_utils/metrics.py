import torch

from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_tfpn(edge_predictions, edge_labels, threshold=0.5):
 
    with torch.no_grad():
        probs = torch.sigmoid(edge_predictions)
        preds = (probs >= threshold).float()

        preds = preds.view(-1)
        labels = edge_labels.view(-1)

        TP = torch.sum((preds == 1) & (labels == 1)).item()
        TN = torch.sum((preds == 0) & (labels == 0)).item()
        FP = torch.sum((preds == 1) & (labels == 0)).item()
        FN = torch.sum((preds == 0) & (labels == 1)).item()

    return TP, TN, FP, FN



def calculate_metrics(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    if total == 0:
      
        return 0.0, 0.0, 0.0, 0.0

    try: 
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN) )
    except ZeroDivisionError:
        f1 = 0
    

    accuracy = (TP + TN) / total
    
    return accuracy, precision, recall, f1


def calculate_metrics_inverse(TP, TN, FP, FN): 
   
    total = TP + TN + FP + FN
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

 
    TP_inv, TN_inv = TN, TP
    FP_inv, FN_inv = FN, FP

    try: 
      
        precision = TP_inv / (TP_inv + FP_inv)
    except ZeroDivisionError:
        precision = 0
    try:
      
        recall = TP_inv / (TP_inv + FN_inv)
    except ZeroDivisionError:
        recall = 0
    try:
      
        f1 = TP_inv / (TP_inv + 0.5 * (FP_inv + FN_inv) )
    except ZeroDivisionError:
        f1 = 0
    

    accuracy = (TP_inv + TN_inv) / total
    
    return accuracy, precision, recall, f1


def get_precision_recall_curve(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    return precision, recall, thresholds


def get_precision_recall_curve_inverse(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = 1 - preds
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, preds, pos_label=0)
    return precision, recall, thresholds


# Actually computes average_precision_score instead of AUC-PC
def get_aps(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    auc_pc = average_precision_score(labels, preds)
    return auc_pc


# Actually computes average_precision_score instead of AUC-PC
def get_aps_inverse(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = 1 - preds
    labels = labels.cpu().numpy()
    auc_pc = average_precision_score(labels, preds, pos_label=0)
    return auc_pc
