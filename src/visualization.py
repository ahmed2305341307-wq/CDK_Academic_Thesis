import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
from .config import FIG

def save(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=150);plt.close('all')
def eda(X,y):
    y.value_counts().sort_index().rename(index={0:'Non-CKD',1:'CKD'}).plot.bar(title='Training class distribution');save('class_distribution')
    X.isna().mean().sort_values().plot.barh(figsize=(8,7),title='Training missing fraction');save('missing_values')
    corr=X.select_dtypes('number').corr();plt.figure(figsize=(8,7));plt.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');plt.xticks(range(len(corr)),corr.columns,rotation=90);plt.yticks(range(len(corr)),corr.columns);plt.colorbar();save('correlation')
    X.select_dtypes('number').hist(figsize=(12,9));save('feature_distributions')
def evaluation(y,preds,table,cv,ab):
    for name,(p,prob) in preds.items():
        ConfusionMatrixDisplay.from_predictions(y,p,display_labels=['Non-CKD','CKD']);plt.title(name);save('confusion_'+name.replace(' ','_'))
    for name,display in [('roc',RocCurveDisplay),('precision_recall',PrecisionRecallDisplay),('calibration',CalibrationDisplay)]:
        _,ax=plt.subplots()
        for model,(_,prob) in preds.items(): display.from_predictions(y,prob,name=model,ax=ax)
        save(name)
    table.set_index('model')[['accuracy','precision','recall','f1','roc_auc']].plot.bar(figsize=(10,5),ylim=(0,1.05));save('comparison')
    cv.boxplot(column='accuracy',by='model',rot=20);plt.suptitle('');save('cv_boxplot')
    ab[ab.model=='Stacking'].groupby('scenario').accuracy.mean().plot.bar(ylim=(0,1.05),title='Feature-set comparison: stacking CV');save('ablation')
