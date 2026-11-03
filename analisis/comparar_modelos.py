import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score, precision_score
R="../../models/"
X=pd.concat([pd.read_csv(R+"X_train.csv",index_col=0),pd.read_csv(R+"X_test.csv",index_col=0)]).astype(str)
y=pd.concat([pd.read_csv(R+"y_train.csv",index_col=0).iloc[:,0],pd.read_csv(R+"y_test.csv",index_col=0).iloc[:,0]])
print(X.shape, y.mean().round(3))
for c in [c for c in X if c.endswith("_bin")]:
    print(c, X.assign(y=y).groupby(c)["y"].agg(["mean","size"]).round(2).to_dict())
SENS=["ORIENTACION_CAT","ES_GRUPO_ETNICO","GRUPO_ETNICO_CAT","VICTIMA_VIOLENCIA","CONDICION_DISCAPACIDAD","IMC_CAT","HABITOS_RIESGO","VIVE_PERSONAS_PATOLOGIAS"]
def prep(cols): return ColumnTransformer([("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cols)])
models={
 "Reg. logística (L2)":LogisticRegression(C=1,class_weight="balanced",max_iter=5000),
 "Reg. logística L1":LogisticRegression(penalty="l1",solver="liblinear",C=0.1,class_weight="balanced",max_iter=5000),
 "Árbol de decisión":DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,min_samples_split=40,class_weight="balanced",random_state=0),
 "Random Forest":RandomForestClassifier(n_estimators=400,max_depth=8,max_features=0.3,max_samples=0.7,min_samples_leaf=3,min_samples_split=16,class_weight="balanced_subsample",random_state=0,n_jobs=-1),
 "Gradient Boosting":HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=200,class_weight="balanced",random_state=0),
}
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42)
rows=[]
for setname,cols in [("todas",list(X)),("sin_sensibles",[c for c in X if c not in SENS]),("sin_notas",[c for c in X if not c.endswith("_bin")])]:
  for n,m in models.items():
    p=Pipeline([("prep",prep(cols)),("clf",m)])
    r=cross_validate(p,X,y,cv=cv,scoring={"auc":"roc_auc","ap":"average_precision","rec":"recall","prec":"precision","f1":"f1"})
    pr=cross_val_predict(p,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba")[:,1]
    rows.append(dict(variables=setname,modelo=n,ROC_AUC=r["test_auc"].mean(),AUC_sd=r["test_auc"].std(),PR_AUC=r["test_ap"].mean(),Recall=r["test_rec"].mean(),Precision=r["test_prec"].mean(),F1=r["test_f1"].mean(),Brier=brier_score_loss(y,pr)))
df=pd.DataFrame(rows).round(3); print(df.to_string()); df.to_csv("comparacion.csv",index=False)
