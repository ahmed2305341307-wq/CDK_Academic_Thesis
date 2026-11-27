from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from .config import MET
def paired_tests(y,preds):
    rows=[]
    for a,b in combinations(preds,2):
        ca=preds[a][0]==np.asarray(y);cb=preds[b][0]==np.asarray(y); n01=int((~ca&cb).sum());n10=int((ca&~cb).sum())
        rows.append(dict(model_a=a,model_b=b,a_only_correct=n10,b_only_correct=n01,p=binomtest(n10,n10+n01,.5).pvalue if n10+n01 else 1.))
    df=pd.DataFrame(rows).sort_values('p');df['holm_p']=np.minimum(1,np.maximum.accumulate(df.p.to_numpy()*np.arange(len(df),0,-1)));df.to_csv(MET/'mcnemar.csv',index=False)
