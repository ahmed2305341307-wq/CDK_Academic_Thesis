import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from scipy.stats import binomtest

def metrics(y,p,prob):
    ci=binomtest(int(np.sum(np.asarray(y)==p)),len(y)).proportion_ci(); tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,p),precision=precision_score(y,p,zero_division=0),recall=recall_score(y,p,zero_division=0),specificity=tn/(tn+fp),f1=f1_score(y,p),roc_auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob),brier=brier_score_loss(y,prob),accuracy_ci_low=ci.low,accuracy_ci_high=ci.high)
