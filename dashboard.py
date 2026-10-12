import streamlit as st
import pandas as pd
from engine import download_data,load_csv,run_backtest
st.set_page_config(page_title='Powell AI Trader V2',layout='wide')
st.title('POWELL AI TRADER V2')
st.caption('Research / historical simulation only — no live orders or real-money trading')
symbol=st.selectbox('Market',['QQQ','SPY'])
capital=st.number_input('Simulated capital (USD)',min_value=100.,value=10000.,step=100.)
risk=st.slider('Max intended risk per trade (%)',0.1,2.0,0.5,0.1)/100
threshold=st.slider('Prediction confidence',0.50,0.80,0.55,0.01)
source=st.radio('Historical data',['Yahoo Finance (5-minute, limited history)','Upload CSV (UTC timestamps)'])
file=st.file_uploader('CSV: datetime,open,high,low,close,volume',type='csv') if source.startswith('Upload') else None
if st.button('Run walk-forward backtest'):
    try:
        df=load_csv(file) if file is not None else download_data(symbol)
        trades,features=run_backtest(df,capital,risk,threshold=threshold)
        st.write(f'Sessions processed: {len(features)} | Trades: {len(trades)}')
        if trades.empty: st.warning('No eligible trades; this is not evidence of profitability.')
        else:
            a,b,c=st.columns(3)
            a.metric('Net P&L (USD)',f'{trades.pnl.sum():,.2f}')
            b.metric('Win rate',f'{(trades.pnl>0).mean():.1%}')
            c.metric('Ending simulated equity',f'{trades.equity.iloc[-1]:,.2f}')
            st.line_chart(pd.concat([pd.Series([capital],name='equity'),trades.equity.reset_index(drop=True)],ignore_index=True))
            st.dataframe(trades,use_container_width=True)
            st.download_button('Export trades CSV',trades.to_csv(index=False),'backtest_trades.csv','text/csv')
    except Exception as e: st.error(str(e))
st.info('Yahoo intraday data is limited. For meaningful model validation, use extensive licensed historical 5-minute data and compare against a simple baseline. No live broker execution is included.')
