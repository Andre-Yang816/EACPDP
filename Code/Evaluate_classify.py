import math

def AUC(label, pre):
    pos = []
    neg = []
    auc = 0
    for index,l in enumerate(label):
        if l == 0:
            neg.append(index)
        else:
            pos.append(index)
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    if len(pos)==0 or len(neg)==0:
        return 0
    else:
        return auc * 1.0 / (len(pos)*len(neg))

def calAUC(prob,labels):
    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc = 0
    if posNum * negNum != 0:
        auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc
def evaluate_classify(Yt,pred):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for i in range(len(Yt)):
        if Yt[i] == 0 and pred[i] == 0:
            TN += 1
        elif Yt[i] == 0 and pred[i] > 0:
            FP += 1
        elif Yt[i] > 0 and pred[i] == 0:
            FN += 1
        elif Yt[i] > 0 and pred[i] > 0:
            TP += 1
    if TN + FP + FN + TP != 0:
        Accuracy = (TN + TP) / (TN + FP + FN + TP)
    else:
        Accuracy = 0.0
    if FN + TP != 0:
        Recall = TP / (TP + FN)
    else:
        Recall = 0.0
    if FP + TP != 0:
        Precision = TP / (TP + FP)
    else:
        Precision = 0.0
    if Precision + Recall !=0:
        F1_measure = 2 * Precision * Recall / (Precision + Recall)
    else:
        F1_measure = 0.0
    auc = AUC(Yt,pred)
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0:
        mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    else:
        mcc = 0.
    return Precision,Recall,F1_measure