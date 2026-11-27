import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from .config import *
def explain(model,X,y,with_shap=False):
    r=permutation_importance(model,X,y,scoring='roc_auc',n_repeats=10,random_state=SEED,n_jobs=1)
    df=pd.DataFrame({'feature':X.columns,'mean':r.importances_mean,'sd':r.importances_std}).sort_values('mean');df.to_csv(MET/'permutation_importance.csv',index=False)
    df.plot.barh(x='feature',y='mean',xerr='sd',legend=False,title='Stacking: holdout permutation importance');plt.tight_layout();plt.savefig(FIG/'permutation_importance.png',dpi=150);plt.close()
    if with_shap:
        import shap
        rf=model.named_estimators_['random_forest'].best_estimator_;prep=rf.named_steps['prep'];z=prep.transform(X)
        exp=shap.TreeExplainer(rf.named_steps['model'])(z)
        if exp.values.ndim==3: exp=exp[:,:,1]
        exp.feature_names=list(prep.get_feature_names_out());shap.plots.beeswarm(exp,show=False);plt.tight_layout();plt.savefig(FIG/'shap_rf.png',dpi=150);plt.close()
