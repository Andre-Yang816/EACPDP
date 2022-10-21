from pandas import np
import warnings
warnings.filterwarnings("ignore")
def ranking_label_loc(Yt,pred,loc,percentage=0.2):
    density = pred / loc
    M = len(Yt)
    N = sum(Yt)
    L = sum(loc)
    K = sum([1 if i >= 1 else 0 for i in Yt])
    m = 0

    l = sum(loc) * percentage

    sort_axis = np.argsort(density)[::-1]
    sorted_Yt = np.array(Yt)[sort_axis]
    sorted_loc = np.array(loc)[sort_axis]
    sum_l = 0
    for i in range(len(sorted_loc)):
        sum_l += sorted_loc[i]
        if sum_l >= l:
            m = i+1
            break
    n = sum(sorted_Yt[:m])
    k = sum([1 if sorted_Yt[i] >= 1 else 0 for i in range(m)])
    Recall = k / K
    Precision = k / m
    Pofb = n / N
    PMI = m / M
    IFA = 0
    for i in range(m):
        if sorted_Yt[i] > 0:
            break
        else:
            IFA += 1
    popt = Popt(Yt, loc, sort_axis)
    if Precision + Recall != 0:
        F1 = (2*Precision * Recall)/(Precision + Recall)
    else:
        F1 = 0
    return Precision, Recall, F1, Pofb, PMI,popt, IFA

def ranking_prob_loc(Yt,prob,loc,percentage=0.2):
    density = prob / loc
    M = len(Yt)
    N = sum(Yt)
    L = sum(loc)
    K = sum([1 if i >= 1 else 0 for i in Yt])
    m = 0
    l = sum(loc) * percentage
    sort_axis = np.argsort(density)[::-1]
    sorted_Yt = np.array(Yt)[sort_axis]
    sorted_loc = np.array(loc)[sort_axis]
    sum_l = 0
    for i in range(len(sorted_loc)):
        sum_l += sorted_loc[i]
        if sum_l >= l:
            m = i + 1
            break
    n = sum(sorted_Yt[:m])
    k = sum([1 if sorted_Yt[i] >= 1 else 0 for i in range(m)])
    Recall = k / K
    Precision = k / m
    Pofb = n / N
    PMI = m / M
    IFA = 0
    for i in range(m):
        if sorted_Yt[i] > 0:
            break
        else:
            IFA += 1

    popt = Popt(Yt, loc, sort_axis)
    if Precision + Recall != 0:
        F1 = (2*Precision * Recall)/(Precision + Recall)
    else:
        F1 = 0
    return Precision, Recall, F1, Pofb, PMI,popt, IFA

def ranking_cbs(Yt,prob,Loc,lamd=0.5,percentage=0.2):
    '''
    :param Yt: true label, here is dichotomous, element value is 0/1
    :param prob: model prediction label probability
    :param Loc: number of lines of code per instance
    :param percentage: percentage of lines of code to check
    :return: 6 metrics
    '''
    yt_defect = []
    yt_non_defect = []
    loc_defect = []
    loc_non_defect = []
    pred_defect = []
    pred_non_defect = []
    for i in range(len(prob)):
        if prob[i] >= lamd:
            yt_defect.append(Yt[i])
            loc_defect.append(Loc[i])
            pred_defect.append(prob[i])
        else:
            yt_non_defect.append(Yt[i])
            loc_non_defect.append(Loc[i])
            pred_non_defect.append(prob[i])

    density_defect = []
    density_non_defect = []
    for i in range(len(loc_defect)):
        if loc_defect[i] == 0:
            density_defect.append(0)
        else:
            density_defect.append(pred_defect[i] / loc_defect[i])
    for i in range(len(loc_non_defect)):
        if loc_non_defect[i] == 0:
            density_non_defect.append(0)
        else:
            density_non_defect.append(pred_non_defect[i] / loc_non_defect[i])

    sort_axis_defect = np.argsort(density_defect)[::-1]
    sort_axis_non_defect = np.argsort(density_non_defect)[::-1]

    sorted_yt_defect = np.array(yt_defect)[sort_axis_defect]
    sorted_yt_non_defect = np.array(yt_non_defect)[sort_axis_non_defect]

    sorted_loc_defect = np.array(loc_defect)[sort_axis_defect]
    sorted_loc_non_defect = np.array(loc_non_defect)[sort_axis_non_defect]

    yt_sorted = np.hstack((sorted_yt_defect, sorted_yt_non_defect))
    loc_sorted = np.hstack((sorted_loc_defect, sorted_loc_non_defect))

    M = len(Yt)
    N = sum(Yt)
    L = sum(Loc)
    K = sum([1 if i >= 1 else 0 for i in Yt])
    m = 0

    l = L * percentage
    sum_l = 0
    for i in range(len(loc_sorted)):
        sum_l += loc_sorted[i]
        if sum_l >= l:
            m = i+1
            break

    n = sum(yt_sorted[:m])
    k = sum([1 if yt_sorted[i] >= 1 else 0 for i in range(m)])
    Recall = k / K
    Precision = k / m
    Pofb = n / N
    PMI = m / M
    IFA = 0
    for i in range(m):
        if yt_sorted[i] == 1:
            break
        else:
            IFA += 1
    sort_axis = np.hstack((sort_axis_defect, sort_axis_non_defect))
    popt = Popt(Yt, Loc, sort_axis)
    if Precision + Recall != 0:
        F1 = (2*Precision * Recall)/(Precision + Recall)
    else:
        F1 = 0
    return Precision, Recall, F1, Pofb, PMI,popt, IFA

def ranking_prob(Yt,prob,loc,percentage=0.2):
    density = prob
    M = len(Yt)
    N = sum(Yt)
    L = sum(loc)
    K = sum([1 if i >= 1 else 0 for i in Yt])
    m = 0

    l = sum(loc) * percentage

    sort_axis = np.argsort(density)[::-1]
    sorted_Yt = np.array(Yt)[sort_axis]
    sorted_loc = np.array(loc)[sort_axis]

    sum_l = 0
    for i in range(len(sorted_loc)):
        sum_l += sorted_loc[i]
        if sum_l >= l:
            m = i + 1
            break

    n = sum(sorted_Yt[:m])
    k = sum([1 if sorted_Yt[i] >= 1 else 0 for i in range(m)])
    Recall = k / K
    Precision = k / m
    Pofb = n / N
    PMI = m / M
    IFA = 0
    for i in range(m):
        if sorted_Yt[i] > 0:
            break
        else:
            IFA += 1
    if Precision + Recall != 0:
        F1 = (2*Precision * Recall)/(Precision + Recall)
    else:
        F1 = 0
    popt = Popt(Yt, loc, sort_axis)

    return Precision, Recall, F1, Pofb, PMI,popt, IFA

def Popt(Yt,loc,pred_index):
    Yt = [1 if Yt[i] >= 1 else 0 for i in range(len(Yt))]
    N = sum(Yt)

    xcost = loc
    xcostsum = sum(xcost)
    optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(xcost, Yt)]
    optimal_index = list(np.argsort(optimal_index))
    optimal_index = optimal_index
    optimal_index.reverse()

    optimal_X = [0]
    optimal_Y = [0]
    for i in optimal_index:
        optimal_X.append(xcost[i] / xcostsum + optimal_X[-1])
        optimal_Y.append(Yt[i] / N + optimal_Y[-1])

    wholeoptimal_auc = 0.
    prev_x = 0
    prev_y = 0
    for x, y in zip(optimal_X, optimal_Y):
        if x != prev_x:
            wholeoptimal_auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    pred_X = [0]
    pred_Y = [0]
    for i in pred_index:
        pred_X.append(xcost[i] / xcostsum + pred_X[-1])
        pred_Y.append(Yt[i] / N + pred_Y[-1])

    wholepred_auc = 0.
    prev_x = 0
    prev_y = 0
    for x, y in zip(pred_X, pred_Y):
        if x != prev_x:
            wholepred_auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    optimal_index.reverse()
    mini_X = [0]
    mini_Y = [0]
    for i in optimal_index:
        mini_X.append(xcost[i] / xcostsum + mini_X[-1])
        mini_Y.append(Yt[i] / N + mini_Y[-1])

    wholemini_auc = 0.
    prev_x = 0
    prev_y = 0
    for x, y in zip(mini_X, mini_Y):
        if x != prev_x:
            wholemini_auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    wholemini_auc = 1 - (wholeoptimal_auc - wholemini_auc)
    wholenormOPT = ((1 - (wholeoptimal_auc - wholepred_auc)) - wholemini_auc) / (1 - wholemini_auc)
    return wholenormOPT
