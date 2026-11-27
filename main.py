import argparse,json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold
from src.config import *
from src.preprocessing import load_data
from src.model_training import make_models
from src.model_evaluation import metrics
from src.visualization import eda,evaluation
from src.explainable_ai import explain
from src.statistical_analysis import paired_tests

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shap',action='store_true');parser.add_argument('--repeats',type=int,default=2);args=parser.parse_args()
    for p in [FIG,MET,REP,ROOT/'models',ROOT/'data/processed']:p.mkdir(parents=True,exist_ok=True)
    X,y=load_data();tr,te=train_test_split(X.index,test_size=.2,stratify=y,random_state=SEED)
    pd.DataFrame({'row':X.index,'split':['train' if i in tr else 'test' for i in X.index]}).to_csv(MET/'split.csv',index=False)
    eda(X.loc[tr],y.loc[tr]);X.assign(classification=y).to_csv(ROOT/'data/processed/cleaned.csv',index=False)
    folds=RepeatedStratifiedKFold(n_splits=5,n_repeats=args.repeats,random_state=SEED);cv=[];ab=[];results=[];preds={};params={}
    for scenario,cols in [('screening',SCREEN),('full_features',NUM+CAT)]:
        A=X.loc[tr,cols];b=y.loc[tr]
        for fold,(i,j) in enumerate(folds.split(A,b),1):
            models=make_models(cols)
            if scenario=='full_features':models={'Stacking':models['Stacking']}
            for name,m in models.items():
                m.fit(A.iloc[i],b.iloc[i]);p=m.predict(A.iloc[j]);prob=m.predict_proba(A.iloc[j])[:,1]
                row={'scenario':scenario,'fold':fold,'model':name,**metrics(b.iloc[j],p,prob)};ab.append(row)
                if scenario=='screening':cv.append(row)
            print(f'{scenario} fold {fold} complete',flush=True)
    ab=pd.DataFrame(ab);ab.to_csv(MET/'ablation.csv',index=False);cv=pd.DataFrame(cv);cv.to_csv(MET/'cross_validation.csv',index=False)
    cv.groupby('model')[['accuracy','f1','roc_auc']].agg(['mean','std']).to_csv(MET/'cv_summary.csv')
    A=X.loc[tr,SCREEN];B=X.loc[te,SCREEN]
    for name,m in make_models(SCREEN).items():
        m.fit(A,y.loc[tr]);p=m.predict(B);prob=m.predict_proba(B)[:,1];preds[name]=(p,prob)
        results.append({'model':name,**metrics(y.loc[te],p,prob)});joblib.dump(m,ROOT/'models'/(name.lower().replace(' ','_')+'.joblib'))
        if name=='Stacking': stack=m;params[name]={n:e.best_params_ for n,e in m.named_estimators_.items()}
        else: params[name]=m.best_params_
        print(name,results[-1],flush=True)
    table=pd.DataFrame(results);table.to_csv(MET/'test_metrics.csv',index=False)
    pred=pd.DataFrame({'row':te,'actual':y.loc[te].to_numpy()})
    for n,(p,prob) in preds.items():pred[n+'_prediction']=p;pred[n+'_probability']=prob
    pred.to_csv(MET/'predictions.csv',index=False)
    evaluation(y.loc[te],preds,table,cv,ab);paired_tests(y.loc[te],preds);explain(stack,B,y.loc[te],args.shap)
    (REP/'run.json').write_text(json.dumps({'seed':SEED,'repeats':args.repeats,'screening_features':SCREEN,'train':len(tr),'test':len(te),'shap':args.shap,'best_params':params},indent=2))
if __name__=='__main__':main()
