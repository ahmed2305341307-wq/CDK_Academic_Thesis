from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .preprocessing import preprocess
from .config import SEED

def make_models(cols):
    specs={'Random Forest':(RandomForestClassifier(n_estimators=60,random_state=SEED,n_jobs=1),{'model__max_depth':[3,6]}),'SVM':(SVC(probability=True,random_state=SEED),{'model__C':[.1,1,10]}),'XGBoost':(XGBClassifier(n_estimators=60,learning_rate=.05,random_state=SEED,n_jobs=1,eval_metric='logloss'),{'model__max_depth':[2,3]})}
    models={n:GridSearchCV(Pipeline([('prep',preprocess(cols)),('model',m)]),g,scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=SEED),n_jobs=1,error_score='raise') for n,(m,g) in specs.items()}
    return {**models,'Stacking':StackingClassifier(estimators=[(n.lower().replace(' ','_'),m) for n,m in models.items()],final_estimator=LogisticRegression(),cv=StratifiedKFold(3,shuffle=True,random_state=SEED),stack_method='predict_proba',n_jobs=1)}
