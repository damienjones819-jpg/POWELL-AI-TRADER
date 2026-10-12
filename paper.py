"""Read-only Alpaca paper-account connectivity check. NEVER submits orders."""
import os
from alpaca.trading.client import TradingClient

def check_paper_account():
    key=os.environ['ALPACA_API_KEY']; secret=os.environ['ALPACA_SECRET_KEY']
    client=TradingClient(key,secret,paper=True)
    account=client.get_account()
    return {'status':str(account.status),'equity':str(account.equity),'buying_power':str(account.buying_power)}

if __name__=='__main__': print(check_paper_account())
