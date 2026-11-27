from src.preprocessing import load_data,preprocess
from src.config import SCREEN
from sklearn.model_selection import train_test_split

def test_data():
    X,y=load_data();assert len(X)==400;assert set(y)=={0,1};assert 'id' not in X and 'classification' not in X

def test_preprocessing():
    X,y=load_data();a,b=train_test_split(X[SCREEN],random_state=42);p=preprocess(SCREEN);p.fit(a);assert p.transform(b).shape[0]==len(b)
