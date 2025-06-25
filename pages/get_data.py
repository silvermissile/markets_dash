import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

####################################
############## RATES ###############
####################################

def get_rates():
    '''
   Import Monthly US rates since 1982 from FRED St Louis
    '''
    start = '2000-01-01'
    tickers = ['GS30','GS10','GS5','GS3','GS2','GS1','GS6m','GS3m']
    df = pdr.get_data_fred(tickers,start)
    df.columns=['30Y','10Y','5Y','3Y','2Y','1Y','6M','3M']
    df.dropna(inplace=True)
    # Changing format from 1st day of the month to last day of the month
    df.index = df.index + pd.offsets.MonthEnd(0)
    print(f'rates as at: {df.index[-1]}')
    return df



####################################
########### EQUITIES ###############
####################################

####################################
# GET FUNCTIONS
####################################

# GETTING S&P 500 constituents from Wikipedia - Save to csv
def get_spx_cons(csv_path):
    '''
    Extract S&P 500 companies from wikipedia and store tickers and Sectors / Industries as df
    Then store as csv.
    '''
    URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(URL)[0]
    df['Symbol'] = df['Symbol'].str.replace('.','-')
    df = df.drop(['Headquarters Location','Date added','CIK','Founded'],axis=1)
    df = df.sort_values(by=['GICS Sector','GICS Sub-Industry'])
    df = df.set_index('Symbol')
    df.dropna(inplace=True)
    return df.to_csv(csv_path)

# GETTING S&P prices from yfinance - Save to csv
def get_prices(df):
    '''
    Dowload prices from yfinance from a list of tickers. returns df of prices written to a csv
    '''
    # local_path linked to drive via sinology
    local_path = '/Users/chloeguillaume/SynologyDrive/Google Drive/DATA_PUBLIC/us_markets_dash_data/spx.csv'

    # url public on google drive
    file_id = '1SoheVoh79lEo5HhVxR_p_XdLewRAgWdh'
    url_open = f'https://drive.google.com/uc?id={file_id}&export=download'
    url_save = f'https://drive.google.com/u/0/uc?id={file_id}&export=download'

    #file = pd.read_csv(local_path)

    tickers_list = df.index.tolist()
    start= '2020-12-31'
    prices_df = yf.download(tickers_list, start=start,interval='1d',)
    file = prices_df['Adj Close']

    return file.to_csv('spx.csv')


####################################
# LOAD FUNCTIONS
####################################

# load S&P 500 weights from IVV ETF stored in csv
def load_IVV_weight():
    '''
    Load weights from IVV Holdings csv => df_IVV
    link to IVV page:

    '''
    local_path = 'pages/IVV_holdings.csv'

    df_IVV = pd.read_csv(local_path,skiprows=8,header=1)
    df_IVV = df_IVV[df_IVV['Asset Class']=='Equity']
    df_IVV = df_IVV[['Ticker','Name','Sector','Asset Class','Weight (%)']]
    df_IVV = df_IVV.set_index('Ticker')
    df_IVV.index = df_IVV.index.str.replace('BRKB','BRK-B')
    df_IVV.index = df_IVV.index.str.replace('BFB','BF-B')
    df_IVV['Weight (%)'] = df_IVV['Weight (%)']/100
    return df_IVV

# load S&P 500 weights from IVV ETF stored in csv
def load_wiki_cons(csv_path):
    '''
    Load tickers, sectors, industries etc. from wiki csv file
    => df
    '''

    df = pd.read_csv(csv_path)
    df = df.set_index('Symbol')
    return df

####################################
# COMPUTE FUNCTIONS
####################################

# Computing Daily returns
def get_returns():
    '''
    Load prices from csv and compute daily stock returns.
    output returns_df
    '''

    local_path = 'pages/spx.csv'

    prices_csv = pd.read_csv(local_path).set_index('Date')
    prices_csv.index = pd.to_datetime(prices_csv.index)
    print(prices_csv.index[-1])

    # fwd fill last prices to missing daily prices (non-trading)
    daily_prices_csv = prices_csv.asfreq('D').ffill()
    returns_df = np.log(daily_prices_csv / daily_prices_csv.shift(1))
    ## EDIT
    last_date = returns_df.index[-1]
    print(last_date)

    last_month_end = pd.date_range(last_date, periods=1, freq='M').strftime('%Y-%m-%d')[0]
    last_month_end = last_date - pd.offsets.MonthEnd(1)
    print(last_month_end)
    returns_df = returns_df[returns_df.index <= last_month_end]

    return returns_df

# Computing resampled returns
def get_spx_returns(local_path = 'pages/spx.csv',freq='W'):
    '''
    Input daily prices df of prices. Use freq as 'D','B', 'W', or 'M' for daily, business, weekly, or monthly.
    Output returns_df in selected frequency
    '''
    prices_df = pd.read_csv(local_path).set_index('Date')
    prices_df.index = pd.to_datetime(prices_df.index)
    # fwd fill last prices to missing daily prices (non-trading)
    prices_df = prices_df.asfreq('D').ffill()

    prices_d = prices_df.dropna(axis=0,how='all')
    prices_d = prices_d.dropna(axis=1)
    prices_res = prices_d.resample(freq).last()
    prices_res = prices_res.dropna(axis=1)
    returns = prices_res.pct_change().dropna()

    ## Set to month end
    last_date = returns.index[-1]
    last_month_end = pd.date_range(last_date, periods=1, freq='M').strftime('%Y-%m-%d')[0]
    last_month_end = last_date - pd.offsets.MonthEnd(1)
    returns = returns[returns.index <= last_month_end]

    return returns


# Computing stock 1M, 3M, and YTD performance
def get_stock_perf(returns_df, df):
    '''
    计算股票在不同区间（1个月、3个月、2022年、2023年YTD）的累计收益率
    参数：
        returns_df: 行为日期（DatetimeIndex），列为股票代码，值为对数收益率
        df: 股票元信息表，索引为股票代码，包含如名称、行业等
    返回：
        stock_df: 合并了收益率区间统计的股票信息表
    '''
    # 确保returns_df的索引为datetime类型，便于按年份筛选
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)
    
    # 计算最近1个月（30天）的累计收益率，结果为每只股票一列
    df_ret_summ = pd.DataFrame(np.exp((returns_df[-30:]).sum())-1, columns=['1M'])
    # 计算最近3个月（90天）的累计收益率
    df_ret_summ['3M'] = np.exp(returns_df[-90:].sum())-1
    
    # 计算2022年全年累计收益率
    returns_2022 = returns_df[returns_df.index.year == 2022]
    if not returns_2022.empty:
        df_ret_summ['2022'] = np.exp(returns_2022.sum())-1
    else:
        df_ret_summ['2022'] = np.nan  # 若无2022年数据则为NaN
    
    # 计算2023年（YTD）累计收益率
    returns_2023 = returns_df[returns_df.index.year == 2023]
    if not returns_2023.empty:
        df_ret_summ['YTD'] = np.exp(returns_2023.sum())-1
    else:
        df_ret_summ['YTD'] = np.nan  # 若无2023年数据则为NaN
    
    # 设置索引名为'Symbol'，便于后续合并
    df_ret_summ.index.rename('Symbol', inplace=True)
    # 将收益率统计与股票元信息表按索引合并
    stock_df = df.join(df_ret_summ)
    return stock_df


# Computing sector ind returns
def get_sector_perf(returns_df, df, period='2022'):
    '''
    from df of daily returns for each stocks compute sector cum performance vs EW
    df: stock info df ,Symbol,Security,Sector,Sub-Industry,Weight
    returns_df: daily returns of all stocks, index is date, columns is Symbol
    '''
    # 调试输出（可选）
    returns_df.head(5).to_csv('returns_df.csv')
    print(returns_df.head(5))
    print(returns_df.index)
    print(returns_df.dtypes)

    df.head(5).to_csv('df.csv')
    print(df.head(5))
    print(df.index)
    print(df.dtypes)
    print(df.index)

    # 合并元信息和收益率数据
    returns = returns_df.T
    returns.index.rename('Symbol', inplace=True)
    returns = df.join(returns)
    # 只保留收益率数据（去除非收益率列）
    meta_cols = ['Security', 'Sector', 'Sub-Industry', 'Weight']
    date_cols = [col for col in returns.columns if col not in meta_cols]
    returns_data = returns[date_cols]

    # 行业分组均值
    sector_returns = returns_data.groupby(returns['Sector']).mean().T
    sector_returns.index = pd.to_datetime(sector_returns.index)
    print('sector_returns info:')
    print(sector_returns.head(5))
    print(sector_returns.index)
    print(sector_returns.dtypes)

    # 子行业分组均值
    ind_returns = returns_data.groupby(returns['Sub-Industry']).mean().T
    ind_returns.index = pd.to_datetime(ind_returns.index)
    print('ind_returns info:')
    print(ind_returns.head(5))
    print(ind_returns.index)
    print(ind_returns.dtypes)

    # 取2023年和2022年数据
    sector_returns_2023 = sector_returns[sector_returns.index.year == 2023]
    sector_returns_2022 = sector_returns[sector_returns.index.year == 2022]
    ind_returns_2023 = ind_returns[ind_returns.index.year == 2023]
    ind_returns_2022 = ind_returns[ind_returns.index.year == 2022]

    # 计算行业累计收益率（用于画线图）
    sector_cum_perf = (np.exp(sector_returns_2023.cumsum())) * 100
    # 设置2022年年末为基准点
    sector_cum_perf.loc[pd.to_datetime('2022-12-31')] = 100
    sector_cum_perf = sector_cum_perf.sort_index()

    # 计算区间收益率
    sector_df = pd.DataFrame(np.exp(sector_returns_2023.sum()) - 1, columns=['YTD'])
    sector_df['3M'] = np.exp(sector_returns_2023[-90:].sum()) - 1
    sector_df['2022'] = np.exp(sector_returns_2022.sum()) - 1

    ind_df = pd.DataFrame(np.exp(ind_returns_2023.sum()) - 1, columns=['YTD'])
    ind_df['3M'] = np.exp(ind_returns_2023[-90:].sum()) - 1
    ind_df['2022'] = np.exp(ind_returns_2022.sum()) - 1

    return sector_df, ind_df, sector_cum_perf

####################################
# FEATURE ENGINEERING
####################################
def join_dfs(df,df_IVV):
    df = df.join(df_IVV['Weight (%)'])
    df.sort_values(by='Weight (%)',inplace=True,ascending=False)
    df = df.rename(columns={'GICS Sector':'Sector','GICS Sub-Industry':'Sub-Industry','Weight (%)':'Weight'})
    df.dropna(inplace=True)
    df = df[df['Weight'] != 0] #to remove any possible 0% weight stock
    return df

##########################
###      ML Models     ###
##########################

### RUN PCA ###
def train_PCA(X, n_comp=3):
    """
    From returns df X, compute n_comp PCA and returns W,pca,X_proj,cum_var
    """
    # Set X

    # Standardize returns into X_scal
    scaler = StandardScaler()
    scaler.fit(X)
    X_scal = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    # Run PCA
    pca = PCA(n_comp)
    pca.fit(X_scal)
    # Get PCA loadings
    W = pca.components_
    W = pd.DataFrame(W.T,
                     index=X.columns,
                     columns=[f'PC{n+1}' for n in range(n_comp)])
    # Print cum explained variance by n_comp components
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f'Total explained variance:{np.round(cum_var.max(),2)} with {n_comp} PCs')

    X_proj = pca.transform(X_scal)
    X_proj = pd.DataFrame(X_proj, columns=[f'PC{n+1}' for n in range(n_comp)],index=X_scal.index)

    return W,pca,X_proj,cum_var


def get_kmean_clusters(X,k=11):
    """
    From X matrix of returns,
    train K-Means to cluster stocks in k clusters.
    Returns clusters labels assigned to each stock in a df
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X.T)
    labels = kmeans.labels_
    # Assign stocks to clusters
    clusters_k = pd.DataFrame(labels,index=X.columns,columns=['cluster'])
    clusters_k = clusters_k.sort_values(by='cluster')
    return clusters_k


def get_pcakmean_clusters(W,k=11):
    """
    From W matrix of PCA loadings for each stock,
    train K-Means to cluster stocks in k clusters.
    Returns clusters labels assigned to each stock in a df
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(W)
    labels = kmeans.labels_
    # Assign stocks to clusters
    clusters_k = pd.DataFrame(labels,index=W.index,columns=['cluster'])
    clusters_k = clusters_k.sort_values(by='cluster')
    return clusters_k
