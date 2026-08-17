import io
import os
import sys
import json
import hmac
import time
import hashlib
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "artifacts"
REGISTRY = ROOT / "models" / "registry"
LOGS = ROOT / "logs"
MODELS.mkdir(parents=True, exist_ok=True)
REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS / "enterprise_platform.log"
logger = logging.getLogger("EnterprisePlatform")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

st.set_page_config(page_title="Enterprise Data Intelligence Platform", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ===================== HIGH-CONTRAST DARK ENTERPRISE UI ===================== */
:root{
  --bg:#0b0f14;
  --surface:#111820;
  --surface-2:#17212b;
  --surface-3:#1d2935;
  --border:#334352;
  --border-soft:#263442;
  --text:#f8fafc;
  --text-2:#e2e8f0;
  --muted:#a8b5c3;
  --accent:#38bdf8;
  --accent-2:#818cf8;
  --good:#34d399;
  --warn:#fbbf24;
  --bad:#fb7185;
}

/* App background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{background:var(--bg)!important;color:var(--text)!important}
[data-testid="stHeader"]{background:#0b0f14!important}
[data-testid="stToolbar"]{background:transparent!important}
.block-container{max-width:1500px!important;padding-top:1.5rem!important;padding-bottom:3rem!important}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"]{background:#0d141b!important;border-right:1px solid var(--border-soft)!important}
[data-testid="stSidebar"] *{color:var(--text-2)!important}
[data-testid="stSidebar"] hr{border-color:var(--border)!important}

/* Global typography - explicit colors so headings can NEVER disappear */
.stMarkdown,.stMarkdown p,.stMarkdown span,.stMarkdown li,
[data-testid="stText"],label,p{color:var(--text-2)!important}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,
.stMarkdown h5,.stMarkdown h6,
h1,h2,h3,h4,h5,h6{color:#ffffff!important;font-weight:800!important}
.main-title{font-size:2.35rem!important;font-weight:900!important;letter-spacing:-.7px;color:#ffffff!important}
.subtitle{color:#b8c5d1!important;margin-bottom:1rem}
.small{font-size:.78rem;color:#a8b5c3!important;text-transform:uppercase;font-weight:800}
.big{font-size:1.65rem;font-weight:900;color:#ffffff!important}

/* Seven section ribbons - strong contrast */
.ribbon{
  background:#172b3a!important;
  color:#ffffff!important;
  padding:13px 18px;
  border:1px solid #3b82a6;
  border-left:4px solid var(--accent);
  border-radius:9px;
  font-weight:850;
  font-size:1.05rem;
  margin:8px 0 14px;
  box-shadow:0 4px 18px rgba(0,0,0,.28);
}
.ribbon *{color:#ffffff!important}

/* Cards */
.card{
  background:var(--surface)!important;
  color:var(--text-2)!important;
  border:1px solid var(--border)!important;
  border-radius:12px;
  padding:16px;
  box-shadow:0 4px 16px rgba(0,0,0,.22)
}
.card *{color:var(--text-2)!important}
.good{background:#0b211b!important;border:1px solid #238c6b!important;border-radius:10px;padding:15px;color:#d1fae5!important}
.good *{color:#d1fae5!important}
.bad{background:#29151a!important;border:1px solid #b94a5b!important;border-radius:10px;padding:15px;color:#ffe4e6!important}
.bad *{color:#ffe4e6!important}
.warn{background:#2a210d!important;border:1px solid #a87816!important;border-radius:10px;padding:15px;color:#fef3c7!important}
.warn *{color:#fef3c7!important}
.logbox{background:#080c11!important;color:#67d7ff!important;border:1px solid #29455a;padding:12px;border-radius:8px;height:220px;overflow:auto;white-space:pre-wrap}

/* Tabs: readable in both selected/unselected states */
button[data-baseweb="tab"]{color:#b7c4d0!important;background:transparent!important;font-weight:700!important}
button[data-baseweb="tab"] p{color:inherit!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#ffffff!important;background:#172b3a!important;border-radius:7px 7px 0 0}
button[data-baseweb="tab"][aria-selected="true"] p{color:#ffffff!important}
div[data-baseweb="tab-highlight"]{background:#38bdf8!important;height:3px!important}

/* Inputs */
.stTextInput label,.stNumberInput label,.stTextArea label,.stSelectbox label,
.stMultiSelect label,.stFileUploader label,.stRadio label,.stCheckbox label,
.stDateInput label,.stTimeInput label{color:#e2e8f0!important;font-weight:700!important}
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stDateInput input,.stTimeInput input{
  background:#111820!important;color:#ffffff!important;border:1px solid #40515f!important;border-radius:8px!important
}
.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:#81909e!important}
[data-baseweb="select"]>div{background:#111820!important;color:#ffffff!important;border-color:#40515f!important}
[data-baseweb="select"] span{color:#ffffff!important}
[data-baseweb="popover"],[data-baseweb="menu"]{background:#151e27!important;border:1px solid #40515f!important}
[data-baseweb="menu"] *{color:#ffffff!important}
[data-baseweb="option"]{background:#151e27!important;color:#ffffff!important}
[data-baseweb="option"]:hover{background:#243545!important}
.stFileUploader{background:#111820!important;border:1px dashed #4c9dcc!important;border-radius:10px;padding:8px}
.stFileUploader section{background:transparent!important}
.stFileUploader *{color:#e2e8f0!important}

/* Buttons */
.stButton>button,.stDownloadButton>button{
  background:#173247!important;color:#ffffff!important;border:1px solid #4b9dcc!important;
  border-radius:8px!important;font-weight:750!important
}
.stButton>button p,.stDownloadButton>button p{color:#ffffff!important}
.stButton>button:hover,.stDownloadButton>button:hover{background:#21465f!important;border-color:#7dd3fc!important;color:#ffffff!important}
.stButton>button:disabled{background:#202a33!important;color:#7f8c98!important;border-color:#37434e!important}

/* Metrics */
div[data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;padding:12px;border-radius:10px;box-shadow:0 3px 12px rgba(0,0,0,.18)}
div[data-testid="stMetricLabel"]{color:#aebbc8!important}
div[data-testid="stMetricLabel"] *{color:#aebbc8!important}
div[data-testid="stMetricValue"]{color:#ffffff!important}
div[data-testid="stMetricValue"] *{color:#ffffff!important}
div[data-testid="stMetricDelta"] *{color:#dbe5ee!important}

/* Tables/dataframes */
div[data-testid="stDataFrame"]{background:#111820!important;border:1px solid var(--border)!important;border-radius:8px;overflow:hidden}
[data-testid="stDataFrame"] iframe{background:#111820!important}

/* Alerts */
div[data-testid="stAlert"]{background:#151e27!important;border:1px solid #40515f!important;color:#f1f5f9!important}
div[data-testid="stAlert"] *{color:#f1f5f9!important}

/* Expanders */
details{background:#111820!important;border:1px solid var(--border)!important;border-radius:10px!important}
details summary,details summary *{color:#ffffff!important;font-weight:700!important}

/* Progress */
div[data-testid="stProgressBar"]>div{background:#26333f!important}

/* Links */
a{color:#7dd3fc!important}
a:hover{color:#bae6fd!important}

/* Horizontal rule */
hr{border-color:#334352!important}

/* Scrollbars */
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-track{background:#0b0f14}
::-webkit-scrollbar-thumb{background:#3b4b59;border-radius:8px}
::-webkit-scrollbar-thumb:hover{background:#526575}
</style>
""", unsafe_allow_html=True)


def log(message, level="INFO"):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {level.upper():<7} | {message}"
    getattr(logger, level.lower(), logger.info)(message)
    st.session_state.setdefault("logs", []).append(line)
    st.session_state.logs = st.session_state.logs[-300:]


def read_csv_bytes(data):
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except Exception as e:
            last_error = e
    else:
        raise ValueError(f"CSV encoding could not be detected: {last_error}")
    sample = text[:20000]
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
    sep = max(counts, key=counts.get)
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def default_bank_path():
    for p in [ROOT/"bank-full.csv", ROOT/"bank.csv", ROOT/"data"/"bank-full.csv", ROOT/"data"/"bank.csv", ROOT/"data"/"raw"/"bank-full.csv", ROOT/"data"/"raw"/"bank.csv"]:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_default():
    p = default_bank_path()
    if p is None:
        raise FileNotFoundError("bank-full.csv was not found. Put the real bank CSV in the project root, data/, or data/raw/.")
    return read_csv_bytes(p.read_bytes()), str(p)


def norm_col(c):
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")

ALIASES = {
    "customer_age":"age","client_age":"age","account_balance":"balance","average_balance":"balance",
    "contact_duration_seconds":"duration","contact_duration":"duration","campaign_contacts":"campaign",
    "previous_contacts":"previous","days_since_previous_contact":"pdays","contact_channel":"contact",
    "marital_status":"marital","housing_loan":"housing","personal_loan":"loan","credit_default":"default",
    "education_level":"education","previous_campaign_outcome":"poutcome","previous_outcome":"poutcome",
    "contact_day":"day","contact_month":"month","target":"y","label":"y","response":"y","outcome":"y",
    "subscribed":"y","subscription":"y","deposit":"y","term_deposit":"y","termdeposit":"y"
}


def canonicalize(df):
    df = df.copy()
    rename = {}
    used = set(df.columns)
    for c in list(df.columns):
        n = norm_col(c)
        target = ALIASES.get(n, n)
        if target != c and target not in used:
            rename[c] = target
            used.add(target)
    if rename:
        df = df.rename(columns=rename)
    return df, rename


def target_report(df):
    if "y" not in df.columns:
        return {"valid":False,"reason":"Target column 'y' is missing.","counts":{},"positive_rate":0.0}
    if df.empty:
        return {"valid":False,"reason":"Dataset contains zero rows.","counts":{},"positive_rate":0.0}
    if df["y"].isna().any():
        return {"valid":False,"reason":f"Target column contains {int(df['y'].isna().sum())} missing values.","counts":{},"positive_rate":0.0}
    s=df["y"].astype(str).str.strip().str.lower().replace({"0":"no","0.0":"no","false":"no","n":"no","1":"yes","1.0":"yes","true":"yes","y":"yes"})
    classes=set(s.unique())
    if classes != {"no","yes"}:
        return {"valid":False,"reason":f"Target must contain yes/no or 0/1. Found: {sorted(classes)[:10]}","counts":s.value_counts().to_dict(),"positive_rate":0.0}
    counts=s.value_counts().to_dict(); rate=counts.get("yes",0)/len(s)
    return {"valid":True,"counts":counts,"positive_rate":rate,"imbalanced":rate<.2 or rate>.8,"reason":"Target is valid."}

BANK_ANCHORS={"age","job","balance","housing","loan","contact","duration","campaign"}

def relevance(baseline,candidate):
    b={norm_col(c) for c in baseline.columns if c!="y"}; c={norm_col(x) for x in candidate.columns if x!="y"}; shared=sorted(b&c); added=sorted(c-b); removed=sorted(b-c)
    coverage=len(shared)/max(1,len(b)); anchor=len(set(shared)&BANK_ANCHORS)/len(BANK_ANCHORS)
    related=len(shared)>=5 and coverage>=.30 and anchor>=.50
    return {"related":related,"shared":shared,"added":added,"removed":removed,"coverage":coverage,"anchor":anchor}


def gate_dataset(baseline,candidate):
    reasons=[]
    if candidate.empty: reasons.append("CSV is empty.")
    if len(candidate)<100: reasons.append(f"Insufficient data: {len(candidate):,} rows. Minimum 100 rows.")
    tr=target_report(candidate)
    if not tr["valid"]: reasons.append(tr["reason"])
    rel=relevance(baseline,candidate)
    if not rel["related"]: reasons.append("Wrong or unrelated bank dataset: insufficient shared bank-domain features.")
    miss=float(candidate.isna().sum().sum()/max(1,candidate.size))
    if miss>.60: reasons.append(f"Too much missing data: {miss:.1%}.")
    constant=[c for c in candidate.columns if candidate[c].nunique(dropna=False)<=1]
    if len(constant)>=max(4,int(candidate.shape[1]*.60)): reasons.append("Too many constant/non-informative columns.")
    status="REJECT" if reasons else ("WARNING" if miss>.20 or len(candidate)<1000 else "ACCEPT")
    return {"status":status,"reasons":reasons or ["Dataset passed defensive checks."],"target":tr,"relevance":rel,"missing_ratio":miss,"constant":constant}


def fingerprint(df):
    payload={"columns":list(df.columns),"dtypes":{c:str(df[c].dtype) for c in df.columns},"rows":len(df)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def schema_diff(base,cand):
    b=set(base.columns); c=set(cand.columns); preserved=sorted(b&c); added=sorted(c-b); removed=sorted(b-c); types={}
    for col in preserved:
        if str(base[col].dtype)!=str(cand[col].dtype): types[col]={"baseline":str(base[col].dtype),"candidate":str(cand[col].dtype)}
    unseen={}
    for col in preserved:
        if base[col].dtype==object and cand[col].dtype==object:
            old=set(base[col].dropna().astype(str).unique()); new=set(cand[col].dropna().astype(str).unique()); diff=sorted(new-old)
            if diff: unseen[col]=diff[:50]
    return {"same":not added and not removed and not types,"added":added,"removed":removed,"preserved":preserved,"types":types,"unseen":unseen,"row_delta":len(cand)-len(base),"column_delta":len(cand.columns)-len(base.columns),"fingerprint":fingerprint(cand)}


def profile(df):
    miss=int(df.isna().sum().sum()); dup=int(df.duplicated().sum()); numeric=df.select_dtypes(include="number"); outlier_rows=set(); by={}
    for col in numeric.columns:
        s=pd.to_numeric(df[col],errors="coerce"); q1=s.quantile(.25); q3=s.quantile(.75); iqr=q3-q1
        if pd.isna(iqr) or iqr<=0: continue
        lo=q1-1.5*iqr; hi=q3+1.5*iqr; mask=((s<lo)|(s>hi)).fillna(False); outlier_rows.update(df.index[mask].tolist()); by[col]={"outliers":int(mask.sum()),"lower":float(lo),"upper":float(hi)}
    ratio=miss/max(1,df.size); score=max(0,min(100,100-min(50,ratio*70)-min(20,dup/max(1,len(df))*50)))
    return {"missing":miss,"duplicates":dup,"numeric":len(numeric.columns),"categorical":len(df.columns)-len(numeric.columns),"outliers":len(outlier_rows),"by":by,"score":score}


def feature_engineer(df):
    d=df.copy()
    if "age" in d: d["age_group"]=pd.cut(pd.to_numeric(d["age"],errors="coerce"),[-np.inf,25,30,40,50,60,np.inf],labels=["<=25","26-30","31-40","41-50","51-60","61+"]).astype(object)
    if "balance" in d:
        b=pd.to_numeric(d["balance"],errors="coerce"); d["balance_log"]=np.sign(b)*np.log1p(np.abs(b))
    if "campaign" in d:
        c=pd.to_numeric(d["campaign"],errors="coerce"); d["campaign_log"]=np.log1p(np.maximum(c,0)); d["campaign_intensity"]=pd.cut(c,[-np.inf,2,5,np.inf],labels=["low","medium","high"]).astype(object)
    if "previous" in d: d["previous_contact"]=(pd.to_numeric(d["previous"],errors="coerce")>0).astype(int)
    if "pdays" in d: d["previously_contacted"]=(pd.to_numeric(d["pdays"],errors="coerce")!=-1).astype(int)
    if "balance" in d: d["zero_balance"]=(pd.to_numeric(d["balance"],errors="coerce")==0).astype(int)
    if "housing" in d and "loan" in d: d["loan_burden"]=((d["housing"].astype(str).str.lower()=="yes")&(d["loan"].astype(str).str.lower()=="yes")).astype(int)
    if "contact" in d: d["contact_unknown"]=(d["contact"].astype(str).str.lower()=="unknown").astype(int)
    if "poutcome" in d: d["previous_success"]=(d["poutcome"].astype(str).str.lower()=="success").astype(int)
    return d


def preprocessor(X):
    nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in X.columns if c not in nums]; transformers=[]
    if nums: transformers.append(("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats:
        try: enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        except TypeError: enc=OneHotEncoder(handle_unknown="ignore",sparse=False)
        transformers.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",enc)]),cats))
    if not transformers: raise ValueError("No usable feature columns remain after removing target.")
    return ColumnTransformer(transformers,remainder="drop")


def model_suite(y):
    pos=max(1,int((y==1).sum())); neg=max(1,int((y==0).sum())); weight=neg/pos
    models={
        "Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=180,max_depth=14,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1),
        "Extra Trees":ExtraTreesClassifier(n_estimators=180,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=140,learning_rate=.06,max_depth=3,random_state=42),
        "HistGradient Boosting":HistGradientBoostingClassifier(max_iter=160,learning_rate=.07,max_leaf_nodes=31,random_state=42),
        "AdaBoost":AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,random_state=42),n_estimators=140,learning_rate=.06,random_state=42),
    }
    if XGBClassifier is not None: models["XGBoost"]=XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric="logloss",tree_method="hist",random_state=42,n_jobs=-1,scale_pos_weight=weight)
    if LGBMClassifier is not None: models["LightGBM"]=LGBMClassifier(n_estimators=180,num_leaves=31,learning_rate=.05,random_state=42,n_jobs=-1,verbosity=-1,class_weight="balanced")
    if CatBoostClassifier is not None: models["CatBoost"]=CatBoostClassifier(iterations=180,depth=6,learning_rate=.05,verbose=False,random_seed=42,allow_writing_files=False,auto_class_weights="Balanced")
    return models


def train_models(df, objective):
    engineered=feature_engineer(df); X=engineered.drop(columns=["y"]); y=engineered["y"].astype(str).str.lower().map({"no":0,"yes":1,"0":0,"1":1})
    if y.isna().any() or y.nunique()!=2: raise ValueError("Target is not valid binary yes/no data.")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    models=model_suite(ytr); results={}; failures={}; pipelines={}
    progress=st.progress(0,"Benchmarking all available algorithms...")
    for i,(name,clf) in enumerate(models.items(),1):
        try:
            pipe=Pipeline([("preprocessor",preprocessor(Xtr)),("model",clf)])
            started=time.perf_counter(); pipe.fit(Xtr,ytr); elapsed=time.perf_counter()-started; pred=pipe.predict(Xte)
            if hasattr(pipe,"predict_proba"): prob=pipe.predict_proba(Xte)[:,1]
            elif hasattr(pipe,"decision_function"):
                raw=pipe.decision_function(Xte); prob=1/(1+np.exp(-np.clip(raw,-30,30)))
            else: prob=pred.astype(float)
            metrics={"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),"roc_auc":float(roc_auc_score(yte,prob)),"seconds":float(elapsed),"confusion_matrix":confusion_matrix(yte,pred).tolist()}
            safe="".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_"); path=MODELS/f"{safe}.pkl"; joblib.dump(pipe,path); metrics["artifact"]=str(path); results[name]=metrics; pipelines[name]=pipe; log(f"MODEL {name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}")
        except Exception as e:
            failures[name]=str(e); log(f"MODEL FAILED {name}: {e}","WARNING")
        progress.progress(i/max(1,len(models)),f"Benchmarking {i}/{len(models)}")
    progress.empty()
    if not results: raise RuntimeError("All candidate algorithms failed. See logs.")
    def key(item):
        m=item[1]
        if objective=="Accuracy": return (m["accuracy"],m["f1"],m["roc_auc"])
        if objective=="F1": return (m["f1"],m["accuracy"],m["roc_auc"])
        if objective=="ROC-AUC": return (m["roc_auc"],m["f1"],m["accuracy"])
        return ((m["accuracy"]+m["f1"]+m["roc_auc"])/3,m["f1"],m["accuracy"])
    champ,metrics=max(results.items(),key=key); champ_pipe=pipelines[champ]; champ_path=MODELS/"champion_pipeline.pkl"; joblib.dump(champ_pipe,champ_path)
    return {"models":results,"failures":failures,"champion":champ,"metrics":metrics,"pipeline":champ_pipe,"features":list(X.columns),"artifact":str(champ_path)}


def governance(result, baseline_metrics=None):
    m=result["metrics"]; reasons=[]
    if m["accuracy"]<.60: reasons.append(f"Accuracy {m['accuracy']:.4f} is below 0.6000.")
    if m["f1"]<.25: reasons.append(f"F1 {m['f1']:.4f} is below 0.2500.")
    if m["roc_auc"]<.65: reasons.append(f"ROC-AUC {m['roc_auc']:.4f} is below 0.6500.")
    if baseline_metrics and m["roc_auc"]<float(baseline_metrics.get("roc_auc",0))-.05: reasons.append("ROC-AUC regression exceeded 0.05.")
    return {"decision":"REJECT" if reasons else "PROMOTE","reasons":reasons or ["Candidate passed production quality gates."]}


def importance(pipe):
    model=pipe.named_steps["model"]
    try: names=list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception: names=[]
    vals=None
    if hasattr(model,"feature_importances_"): vals=np.asarray(model.feature_importances_)
    elif hasattr(model,"coef_"): vals=np.abs(np.asarray(model.coef_)[0])
    elif hasattr(model,"get_feature_importance"):
        try: vals=np.asarray(model.get_feature_importance())
        except Exception: pass
    if vals is None: return pd.DataFrame(columns=["feature","importance"])
    if len(names)!=len(vals): names=[f"feature_{i}" for i in range(len(vals))]
    return pd.DataFrame({"feature":names,"importance":vals}).sort_values("importance",ascending=False).head(20).reset_index(drop=True)


def authenticate(username,password):
    expected_user=os.getenv("EDIP_ADMIN_USER","admin"); expected_pass=os.getenv("EDIP_ADMIN_PASSWORD","99")
    return hmac.compare_digest(str(username).strip(),expected_user) and hmac.compare_digest(str(password).strip(),expected_pass)


# ---------------- SESSION ----------------
for key,val in {"authenticated":False,"baseline_df":None,"baseline_path":None,"active_df":None,"active_name":"Not loaded","active_kind":"default","gate":None,"schema":None,"training":None,"prediction":None,"logs":[]}.items():
    st.session_state.setdefault(key,val)

if st.session_state.baseline_df is None:
    try:
        base,path=load_default(); base,_=canonicalize(base); st.session_state.baseline_df=base; st.session_state.baseline_path=path; st.session_state.active_df=base.copy(); st.session_state.active_name=Path(path).name; st.session_state.active_kind="default"; st.session_state.gate=gate_dataset(base,base); st.session_state.schema=schema_diff(base,base); log(f"BOOT: loaded default dataset {base.shape}")
    except Exception as e:
        st.error(str(e)); log(f"BOOT FAILED: {e}","ERROR"); st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Enterprise Control Center")
    st.markdown("### 🔐 Authentication")
    if not st.session_state.authenticated:
        with st.form("login"):
            user=st.text_input("Username")
            pw=st.text_input("Password",type="password")
            submitted=st.form_submit_button("Login",width="stretch")
        if submitted:
            if authenticate(user,pw): st.session_state.authenticated=True; log("AUTH: administrator authenticated"); st.rerun()
            else: st.error("Invalid credentials."); log("AUTH: failed login attempt","WARNING")
    else:
        st.success("Authenticated administrator")
        if st.button("Logout",width="stretch"): st.session_state.authenticated=False; log("AUTH: administrator logged out"); st.rerun()
    st.markdown("---")
    st.markdown("### 📥 Main Dataset")
    st.download_button("⬇️ Download Main Bank CSV",st.session_state.baseline_df.to_csv(index=False).encode(),"bank-full.csv","text/csv",width="stretch")
    st.caption(f"Baseline: {len(st.session_state.baseline_df):,} rows × {len(st.session_state.baseline_df.columns)} columns")
    st.markdown("---")
    st.markdown("### 📤 Updated Dataset")
    upload=st.file_uploader("Upload updated CSV",type=["csv"],disabled=not st.session_state.authenticated)
    if not st.session_state.authenticated: st.caption("Login required for CSV updates.")
    if upload is not None and st.session_state.authenticated:
        h=hashlib.sha256(upload.getvalue()).hexdigest()
        if st.session_state.get("upload_hash")!=h:
            st.session_state.upload_hash=h
            try:
                candidate,_=canonicalize(read_csv_bytes(upload.getvalue())); gate=gate_dataset(st.session_state.baseline_df,candidate); st.session_state.active_df=candidate; st.session_state.active_name=upload.name; st.session_state.active_kind="candidate"; st.session_state.gate=gate; st.session_state.schema=schema_diff(st.session_state.baseline_df,candidate); st.session_state.training=None; st.session_state.prediction=None; log(f"INGESTION: {upload.name} {candidate.shape}; gate={gate['status']}"); st.rerun()
            except Exception as e:
                st.error(f"CSV ingestion failed: {e}"); log(f"INGESTION FAILED: {e}","ERROR")
    if st.button("↩️ Reset to Default Prediction",width="stretch"):
        st.session_state.active_df=st.session_state.baseline_df.copy(); st.session_state.active_name=Path(st.session_state.baseline_path).name; st.session_state.active_kind="default"; st.session_state.gate=gate_dataset(st.session_state.baseline_df,st.session_state.baseline_df); st.session_state.schema=schema_diff(st.session_state.baseline_df,st.session_state.baseline_df); st.session_state.training=None; st.session_state.prediction=None; st.session_state.pop("upload_hash",None); log("RESET: default dataset restored"); st.rerun()
    st.markdown("---")
    objective=st.selectbox("Champion objective",["Balanced","Accuracy","F1","ROC-AUC"])
    st.markdown("### 🧾 Runtime Logs")
    st.markdown(f"<div class='logbox'>{'<br>'.join(st.session_state.logs[-30:]) or 'No logs yet.'}</div>",unsafe_allow_html=True)

# ---------------- HEADER ----------------
df=st.session_state.active_df
# Defensive session-state repair. Every rerun gets a complete gate/schema object.
# This also repairs state created by older versions of the application.
try:
    current_gate = st.session_state.get("gate")
    if not isinstance(current_gate, dict):
        raise ValueError("gate state is missing or invalid")
    required_gate_keys = {"target", "relevance", "reasons", "status", "missing_ratio", "constant"}
    if not required_gate_keys.issubset(current_gate.keys()):
        raise ValueError("gate state is incomplete")
    if not isinstance(current_gate.get("target"), dict):
        raise ValueError("target gate state is invalid")
except Exception:
    try:
        st.session_state.gate = gate_dataset(st.session_state.baseline_df, df)
        log("STATE REPAIR: rebuilt complete dataset gate", "WARNING")
    except Exception as e:
        st.session_state.gate = {
            "status": "REJECT",
            "reasons": [f"Dataset gate could not be rebuilt: {e}"],
            "target": {"valid": False, "reason": str(e), "counts": {}, "positive_rate": 0.0, "imbalanced": False},
            "relevance": {"related": False, "shared": [], "added": [], "removed": [], "coverage": 0.0, "anchor": 0.0},
            "missing_ratio": 1.0,
            "constant": []
        }
gate = st.session_state.gate
# Backfill optional keys so every section remains crash-safe.
gate["target"] = gate.get("target") if isinstance(gate.get("target"), dict) else {
    "valid": False, "reason": "Target validation unavailable.", "counts": {}, "positive_rate": 0.0, "imbalanced": False
}
gate["relevance"] = gate.get("relevance") if isinstance(gate.get("relevance"), dict) else {
    "related": False, "shared": [], "added": [], "removed": [], "coverage": 0.0, "anchor": 0.0
}
gate["reasons"] = gate.get("reasons") or ["Dataset validation could not be completed."]
gate["status"] = gate.get("status") or "REJECT"
gate["missing_ratio"] = float(gate.get("missing_ratio", 1.0))
gate["constant"] = gate.get("constant") or []

try:
    schema = st.session_state.schema
    if not isinstance(schema, dict) or not {"same", "added", "removed", "preserved", "types", "unseen", "row_delta", "column_delta", "fingerprint"}.issubset(schema.keys()):
        schema = schema_diff(st.session_state.baseline_df, df)
except Exception:
    schema = schema_diff(st.session_state.baseline_df, df)
st.session_state.schema = schema
q=profile(df)
st.markdown("<div class='main-title'>Enterprise Data Intelligence Platform</div>",unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Adaptive data intelligence • schema drift • multi-model learning • governance • real-time inference</div>",unsafe_allow_html=True)
if gate["status"]=="REJECT": st.markdown(f"<div class='bad'><b>🔴 NEW CSV ACTIVE — PREDICTION LOCKED</b><br>{st.session_state.active_name}<br>Reports below use this CSV only.<br><br>"+"<br>".join(gate["reasons"])+"</div>",unsafe_allow_html=True)
elif st.session_state.active_kind=="candidate": st.markdown(f"<div class='warn'><b>🟡 UPDATED CSV ACTIVE</b><br>{st.session_state.active_name}<br>All seven reports now use the updated CSV.</div>",unsafe_allow_html=True)
else: st.markdown(f"<div class='good'><b>🟢 DEFAULT BASELINE ACTIVE</b><br>{st.session_state.active_name}</div>",unsafe_allow_html=True)

k=st.columns(6)
for c,label,val in zip(k,["Rows","Columns","Missing Cells","Duplicates","Quality","Anomalies"],[f"{len(df):,}",len(df.columns),q["missing"],q["duplicates"],f"{q['score']:.1f}/100",q["outliers"]]):
    c.markdown(f"<div class='card'><div class='small'>{label}</div><div class='big'>{val}</div></div>",unsafe_allow_html=True)

tabs=st.tabs(["📂 1. Ingestion & Pre-Flight","📊 2. Deep Data Profiling","🔍 3. Anomaly Intelligence","🧬 4. Schema Drift & Registry","⚙️ 5. Feature Engineering Studio","🚀 6. Adaptive Retraining & Governance","🔮 7. Real-Time Inference & Explainability"])

with tabs[0]:
    st.markdown('<div class="ribbon">📂 1. Ingestion & Pre-Flight</div>',unsafe_allow_html=True)
    t=gate.get("target", {}) or {}
    c=st.columns(4)
    c[0].metric("Gate", gate.get("status", "REJECT"))
    c[1].metric("Rows", f"{len(df):,}")
    c[2].metric("Columns", len(df.columns))
    c[3].metric("YES Rate", f"{float(t.get('positive_rate', 0)):.2%}" if t.get("valid", False) else "N/A")
    st.dataframe(df.head(20),width="stretch")
    st.dataframe(pd.DataFrame([{"column":x,"dtype":str(df[x].dtype),"missing":int(df[x].isna().sum()),"missing_%":round(df[x].isna().mean()*100,2),"unique":int(df[x].nunique(dropna=True))} for x in df.columns]),width="stretch")
    if not t.get("valid"):
        st.error(t.get("reason", "Target validation failed."))

with tabs[1]:
    st.markdown('<div class="ribbon">📊 2. Deep Data Profiling</div>',unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").transpose().reset_index().rename(columns={"index":"column"}),width="stretch")
    nums=df.select_dtypes(include="number")
    if not nums.empty:
        st.markdown("### Numeric Distributions"); selected=st.selectbox("Feature",nums.columns,key="profile_feature"); st.line_chart(nums[selected].reset_index(drop=True).head(5000))
        st.markdown("### Correlation Matrix"); st.dataframe(nums.corr().round(3),width="stretch")

with tabs[2]:
    st.markdown('<div class="ribbon">🔍 3. Anomaly Intelligence</div>',unsafe_allow_html=True)
    st.write(f"Detected **{q['outliers']:,}** anomalous numeric records using IQR rules.")
    if q["by"]:
        at=pd.DataFrame([{"feature":x,**v} for x,v in q["by"].items()]).sort_values("outliers",ascending=False); st.dataframe(at,width="stretch")
        feat=st.selectbox("Inspect feature",at.feature.tolist(),key="anom_feature"); b=q["by"][feat]; s=pd.to_numeric(df[feat],errors="coerce"); mask=((s<b["lower"])|(s>b["upper"])).fillna(False); st.dataframe(df.loc[mask].head(100),width="stretch")
    else: st.success("No numeric IQR anomalies detected.")

with tabs[3]:
    st.markdown('<div class="ribbon">🧬 4. Schema Drift & Registry</div>',unsafe_allow_html=True)
    c=st.columns(5); c[0].metric("Schema", "MATCH" if schema["same"] else "DRIFT"); c[1].metric("Added",len(schema["added"])); c[2].metric("Removed",len(schema["removed"])); c[3].metric("Preserved",len(schema["preserved"])); c[4].metric("Row Delta",f"{schema['row_delta']:+,}")
    a,b=st.columns(2)
    with a:
        st.markdown("**Added columns**")
        st.write(schema.get("added") or "None")
        st.markdown("**Removed columns**")
        st.write(schema.get("removed") or "None")
    with b:
        st.markdown("**Type mutations**")
        st.json(schema.get("types") or {"type_mutations": []})
        st.markdown("**Unseen categories**")
        st.json(schema.get("unseen") or {"unseen_categories": []})
    st.code(json.dumps({"dataset":st.session_state.active_name,"fingerprint":schema["fingerprint"],"timestamp":datetime.now().isoformat(timespec="seconds")},indent=2))

with tabs[4]:
    st.markdown('<div class="ribbon">⚙️ 5. Feature Engineering Studio</div>',unsafe_allow_html=True)
    eng=feature_engineer(df); generated=[x for x in eng.columns if x not in df.columns]; st.metric("Generated Features",len(generated)); st.dataframe(pd.DataFrame({"engineered_feature":generated}),width="stretch"); st.dataframe(eng.head(20),width="stretch")

with tabs[5]:
    st.markdown('<div class="ribbon">🚀 6. Adaptive Retraining & Governance</div>',unsafe_allow_html=True)
    if not st.session_state.authenticated: st.warning("🔐 Administrator authentication is required for retraining.")
    elif gate["status"]=="REJECT": st.error("🔴 Retraining locked because the active CSV failed the defensive data gate.")
    elif st.button("🚀 Run Full Adaptive Model Benchmark",type="primary",width="stretch"):
        try:
            with st.spinner("Training and comparing all available algorithms..."):
                result=train_models(df,objective)
                # For a candidate dataset, compare against currently stored baseline only when available.
                base_metrics=None
                registry_file=REGISTRY/"model_registry.json"
                if registry_file.exists():
                    try: base_metrics=json.loads(registry_file.read_text(encoding="utf-8")).get("active_metrics")
                    except Exception: base_metrics=None
                gov=governance(result,base_metrics); result["governance"]=gov; st.session_state.training=result
                record={"timestamp":datetime.now().isoformat(timespec="seconds"),"dataset":st.session_state.active_name,"fingerprint":fingerprint(df),"champion":result["champion"],"metrics":result["metrics"],"governance":gov,"models":result["models"]}
                if gov["decision"]=="PROMOTE":
                    (REGISTRY/"model_registry.json").write_text(json.dumps({"active_model":result["champion"],"active_metrics":result["metrics"],"active_dataset":st.session_state.active_name,"history":[record]},indent=2,default=str),encoding="utf-8")
                    st.session_state.prediction_allowed=True
                    st.session_state.champion=result["pipeline"]
                else:
                    st.session_state.prediction_allowed=False; st.session_state.champion=None
                log(f"TRAINING END: champion={result['champion']} decision={gov['decision']}"); st.rerun()
        except Exception as e:
            st.error(f"Training failed safely: {e}"); st.code(traceback.format_exc()); log(f"TRAINING FAILED: {e}","ERROR")
    if st.session_state.training:
        r=st.session_state.training; rows=[]
        for name,m in r["models"].items(): rows.append({"Algorithm":name,"Accuracy":m["accuracy"],"Precision":m["precision"],"Recall":m["recall"],"F1":m["f1"],"ROC-AUC":m["roc_auc"],"Seconds":m["seconds"],"Status":"⭐ CHAMPION" if name==r["champion"] else "Candidate"})
        table=pd.DataFrame(rows).sort_values(["Accuracy","F1","ROC-AUC"],ascending=False); st.dataframe(table,width="stretch"); st.bar_chart(table.set_index("Algorithm")[["Accuracy","F1","ROC-AUC"]]);
        if r["governance"]["decision"]=="PROMOTE": st.success(f"🟢 PROMOTED: {r['champion']}")
        else: st.error("🔴 REJECTED — inference remains locked."); [st.warning(x) for x in r["governance"]["reasons"]]
        if r["failures"]: st.json(r["failures"])

with tabs[6]:
    st.markdown('<div class="ribbon">🔮 7. Real-Time Inference & Explainability</div>',unsafe_allow_html=True)
    if gate["status"]=="REJECT":
        st.error("🚫 INSUFFICIENT / WRONG / UNRELATED DATA — prediction unavailable. Reset to Default Prediction or upload a valid related bank CSV.")
    elif not st.session_state.get("prediction_allowed") or st.session_state.get("champion") is None:
        st.warning("🔒 Prediction is locked. Authenticate, train the current dataset, and promote a valid champion model.")
    else:
        pipe=st.session_state.champion; source=df.drop(columns=["y"],errors="ignore"); fields=list(source.columns); values={}; cols=st.columns(3)
        for i,col in enumerate(fields):
            with cols[i%3]:
                s=source[col]
                if pd.api.types.is_numeric_dtype(s):
                    n=pd.to_numeric(s,errors="coerce").dropna(); default=float(n.median()) if len(n) else 0.0; lo=float(n.min()) if len(n) else default-1; hi=float(n.max()) if len(n) else default+1; hi=max(hi,lo+1); default=float(np.clip(default,lo,hi)); values[col]=st.number_input(col.replace("_"," ").title(),min_value=lo,max_value=hi,value=default,key=f"inf_{col}")
                else:
                    opts=s.dropna().astype(str).value_counts().head(100).index.tolist() or ["unknown"]; values[col]=st.selectbox(col.replace("_"," ").title(),opts,key=f"inf_{col}")
        if st.button("🚀 Analyze Customer",type="primary",width="stretch"):
            try:
                customer_input=pd.DataFrame([values]); customer=feature_engineer(customer_input); expected=st.session_state.training["features"] if st.session_state.training else list(source.columns); 
                for c in expected:
                    if c not in customer: customer[c]=np.nan
                customer=customer[expected]; pred=int(pipe.predict(customer)[0]); prob=float(pipe.predict_proba(customer)[0,1]) if hasattr(pipe,"predict_proba") else float(pred); decision="YES" if pred==1 else "NO"; risk="LOW" if prob>=.75 else "MEDIUM" if prob>=.50 else "HIGH"; priority="HIGH" if prob>=.75 else "MEDIUM" if prob>=.50 else "LOW"; action="Prioritize customer for high-intent campaign follow-up." if prob>=.75 else "Include customer in standard marketing follow-up." if prob>=.50 else "Do not prioritize; retain for lower-frequency campaign treatment."; st.session_state.prediction={"decision":decision,"prob":prob,"risk":risk,"priority":priority,"action":action,"customer":customer,"profile":values.copy()}; log(f"INFERENCE: decision={decision}, probability={prob:.4f}"); st.rerun()
            except Exception as e:
                st.error("Prediction failed safely because the current schema is incompatible with the promoted model."); log(f"INFERENCE FAILED: {e}","ERROR")
        if st.session_state.prediction:
            p=st.session_state.prediction; c=st.columns(4)
            for box,label,val in zip(c,["Prediction","Probability","Risk","Priority"],[p["decision"],f"{p['prob']:.2%}",p["risk"],p["priority"]]): box.markdown(f"<div class='card'><div class='small'>{label}</div><div class='big'>{val}</div></div>",unsafe_allow_html=True)
            st.progress(p["prob"],text=f"YES probability: {p['prob']:.2%}"); st.info(p["action"]); st.markdown("### 👤 Customer Profile")
            profile=p.get("profile",{})
            labels={"age":"Age","job":"Job","marital":"Marital Status","education":"Education","balance":"Balance","housing":"Housing Loan","loan":"Personal Loan","default":"Default","contact":"Contact Method","month":"Campaign Month","day":"Day","campaign":"Campaign Contacts","pdays":"Days Since Previous Contact","previous":"Previous Contacts","poutcome":"Previous Outcome","duration":"Call Duration"}
            groups=[("Personal",["age","job","marital","education"]),("Financial",["balance","housing","loan","default"]),("Campaign",["contact","month","day","campaign","pdays","previous","poutcome","duration"])]
            for group,keys in groups:
                st.markdown(f"**{group}**")
                present=[k for k in keys if k in profile]
                if present:
                    pc=st.columns(min(4,max(1,len(present))))
                    for idx,k in enumerate(present):
                        v=profile.get(k)
                        if pd.isna(v) if not isinstance(v,(list,dict)) else False: v="—"
                        if isinstance(v,float) and np.isnan(v): v="—"
                        if k in {"housing","loan","default"} and str(v).lower() in {"yes","no"}: v=str(v).title()
                        if k=="balance" and isinstance(v,(int,float,np.integer,np.floating)) and not pd.isna(v): v=f"{v:,.0f}"
                        pc[idx%len(pc)].markdown(f"<div class='card'><div class='small'>{labels.get(k,k.replace('_',' ').title())}</div><div style='font-size:1.05rem;font-weight:700;color:#f8fafc;margin-top:6px'>{v}</div></div>",unsafe_allow_html=True)
            st.markdown("### 🤖 Model Information")
            r=st.session_state.training; c=st.columns(4); c[0].metric("Model",r["champion"]); c[1].metric("Accuracy",f"{r['metrics']['accuracy']:.4f}"); c[2].metric("F1",f"{r['metrics']['f1']:.4f}"); c[3].metric("ROC-AUC",f"{r['metrics']['roc_auc']:.4f}")
            imp=importance(pipe); st.markdown("### 📊 Top Model Features"); st.dataframe(imp,width="stretch"); st.bar_chart(imp.set_index("feature")["importance"] if not imp.empty else pd.Series(dtype=float)); st.markdown("### 📋 Final Decision Summary"); st.dataframe(pd.DataFrame({"Component":["Prediction","Probability","Risk","Priority","Action"],"Result":[p["decision"],f"{p['prob']:.2%}",p["risk"],p["priority"],p["action"]]}),width="stretch")

st.markdown("---")
st.caption("⚡ Enterprise Data & Decision Intelligence Platform • Adaptive Production Build")