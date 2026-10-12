import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

NY=ZoneInfo('America/New_York')
FEATURES=['gap','range_pct','body_pct','volume_ratio','prev_return','volatility']

def download_data(symbol='QQQ', period='30d'):
    import yfinance as yf
    df=yf.download(symbol,period=period,interval='5m',auto_adjust=True,progress=False,prepost=False)
    if df.empty: raise ValueError('No data downloaded; Yahoo intraday history is limited (typically 60 days).')
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df.rename(columns=str.lower)
    if df.index.tz is None: df.index=df.index.tz_localize('UTC')
    df.index=df.index.tz_convert(NY)
    return df[['open','high','low','close','volume']].dropna()

def build_days(df):
    rows=[]; sessions=[]
    for day,g in df.groupby(df.index.date):
        g=g.between_time('09:30','11:00').sort_index()
        opening=g.between_time('09:30','09:40') # three 5-min bars, 09:30-09:45
        after=g.between_time('09:45','10:55')
        if len(opening)<3 or len(after)<4: continue
        high=float(opening.high.max()); low=float(opening.low.min()); op=float(opening.open.iloc[0]); close=float(opening.close.iloc[-1]); volume=float(opening.volume.mean())
        prev=g.close.pct_change().dropna()
        rows.append({'date':str(day),'gap':(op/close-1), 'range_pct':(high-low)/op,'body_pct':(close-op)/op,'volume_ratio':volume/max(float(opening.volume.mean()),1), 'prev_return':float(prev.iloc[-1]) if len(prev) else 0,'volatility':float(prev.std()) if len(prev)>1 else 0})
        sessions.append((str(day),high,low,after))
    return pd.DataFrame(rows),sessions

def run_backtest(df,capital=10000,risk_fraction=.005,fee_bps=2,threshold=.55):
    feats,sessions=build_days(df)
    if len(feats)<12: raise ValueError('Need at least 12 complete market sessions; try a longer period or CSV history.')
    # Labels use only subsequent price action; training is strictly earlier days.
    labels=[]
    for _,hi,lo,after in sessions:
        end=float(after.close.iloc[-1]); midpoint=(hi+lo)/2
        labels.append(int(end>midpoint))
    feats['label']=labels
    records=[]; equity=float(capital)
    for i in range(10,len(feats)):
        train=feats.iloc[:i]
        if train.label.nunique()<2: continue
        model=HistGradientBoostingClassifier(max_iter=40,max_leaf_nodes=5,min_samples_leaf=3,random_state=42)
        model.fit(train[FEATURES],train.label)
        p=float(model.predict_proba(feats.iloc[[i]][FEATURES])[0,1])
        date,hi,lo,after=sessions[i]
        side=1 if p>=threshold else (-1 if p<=1-threshold else 0)
        if side==0: continue
        # Signal is formed at 09:45; enter only if subsequent bar crosses the opening range.
        trigger=hi if side==1 else lo
        hit=after[after.high>=trigger] if side==1 else after[after.low<=trigger]
        if hit.empty: continue
        entrybar=hit.iloc[0]; entrytime=hit.index[0]
        if side==1 and float(entrybar.open)>trigger: entry=float(entrybar.open)
        elif side==-1 and float(entrybar.open)<trigger: entry=float(entrybar.open)
        else: entry=trigger
        stop=lo if side==1 else hi
        distance=abs(entry-stop)
        if distance<=0: continue
        qty=min(equity/max(entry,1),equity*risk_fraction/distance)
        if qty<=0: continue
        target=entry+side*2*distance
        exitprice=float(after.close.iloc[-1]); reason='time'
        for _,bar in after.loc[entrytime:].iterrows():
            stophit=(bar.low<=stop if side==1 else bar.high>=stop)
            targethit=(bar.high>=target if side==1 else bar.low<=target)
            if stophit: exitprice=min(stop,float(bar.open)) if side==1 else max(stop,float(bar.open));reason='stop';break
            if targethit: exitprice=target;reason='target';break
        fees=qty*(entry+exitprice)*fee_bps/10000
        pnl=qty*side*(exitprice-entry)-fees
        equity+=pnl
        records.append({'date':date,'side':'LONG' if side==1 else 'SHORT','probability':round(p,3),'entry':round(entry,3),'exit':round(exitprice,3),'reason':reason,'pnl':round(pnl,2),'equity':round(equity,2)})
    return pd.DataFrame(records),feats

def load_csv(path):
    df=pd.read_csv(path)
    df.columns=[c.lower().strip() for c in df.columns]
    if 'datetime' not in df: raise ValueError('CSV requires datetime,open,high,low,close,volume columns (UTC timestamps).')
    df['datetime']=pd.to_datetime(df['datetime'],utc=True)
    df=df.set_index('datetime').sort_index().tz_convert(NY)
    return df[['open','high','low','close','volume']].astype(float)
