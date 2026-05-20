"""
providers/universe_seeds.py -- Curated high-liquidity ticker lists

Used as the scan batch when the raw universe is alphabetical/unsorted.
These are the most-traded US stocks/ETFs by daily dollar volume.
Scanner still discovers NEW tickers via Alpaca/Polygon -- this just
ensures high-quality candidates fill the batch slots.
"""

# Top ~200 US stocks by avg daily volume (regularly updated)
US_LIQUID = [
    # Mega-cap tech
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL",
    "AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL","ADI",
    # Financials
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","V","MA","PYPL","SQ",
    # Healthcare
    "UNH","JNJ","PFE","MRK","ABBV","BMY","LLY","GILD","AMGN","BIIB","MRNA",
    # Consumer
    "WMT","COST","TGT","HD","LOW","NKE","SBUX","MCD","CMG","YUM",
    # Energy
    "XOM","CVX","COP","SLB","OXY","HAL","MPC","PSX","VLO","EOG",
    # Industrial/Defense
    "BA","CAT","HON","UPS","FDX","LMT","RTX","GE","MMM","DE",
    # Communication
    "NFLX","DIS","CMCSA","T","VZ","TMUS","SNAP","PINS","TWTR","RBLX",
    # High-momentum / growth
    "PLTR","COIN","HOOD","SOFI","RIVN","LCID","NIO","XPEV","LI","GRAB",
    "RKLB","IONQ","QUBT","ARQT","SMCI","ARM","MSTR",
    # ETFs (liquid)
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","XLF","XLE",
    "XLK","XLV","XLI","ARKK","SOXL","TQQQ","SPXL","UVXY","VXX",
    # Mid-cap momentum names
    "CRWD","DDOG","SNOW","NET","ZS","OKTA","PANW","FTNT","S","CYBR",
    "SHOP","SE","MELI","APP","TTD","ROKU","ZM","DOCN","BILL","HUBS",
    "UBER","LYFT","ABNB","DASH","DKNG","PENN","MGAM",
    "GME","AMC","BBBY","SPCE","WISH","CLOV","WKHS","RIDE","NKLA",
    # International ADRs (highly liquid)
    "BABA","TSM","ASML","NVO","SAP","TM","SONY","BIDU","JD","PDD",
]

# Top crypto by market cap (CoinGecko top-30 that yfinance supports reliably)
CRYPTO_LIQUID = [
    "BTC","ETH","SOL","BNB","XRP","ADA","AVAX","DOGE","DOT","MATIC",
    "LINK","LTC","BCH","ATOM","XLM","NEAR","ALGO","FIL","ETC","APT",
    "ARB","AAVE","MKR","GRT","IMX","MANA","SAND","AXS","HBAR","VET",
]

# Penny stocks with consistent volume
PENNY_LIQUID = [
    "SNDL","NAKD","EXPR","CLOV","WKHS","RIDE","NKLA","IDEANOMICS",
    "MULN","BBIG","PROG","CENN","EVGO","LCID","GOEV","ARVL",
    "VERB","SFUN","HCDI","BKYI","ABML","MMAT","TRCH",
]
