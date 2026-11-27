from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED=42
SCREEN=['age','bp','bgr','htn','dm','cad','appet','pe','ane']
NUM=['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
CAT=['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
FIG=ROOT/'results/figures'; MET=ROOT/'results/metrics'; REP=ROOT/'results/reports'
