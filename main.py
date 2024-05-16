import streamlit as st
import yfinance as yf
# from prophet import Prophet
# from prophet.plot import plot_plotly
from app.plot import plot_plotly
# from app.make_future import plot_plotly
from datetime import date
from plotly import graph_objs as go
import requests
from PIL import Image
import ta
from app.make_future import MakeFuture


# Set page configuration
img = Image.open('logo.jpeg')
st.set_page_config(page_title='Predict Crypto Price', page_icon=img, layout='wide')

# Custom CSS for styling
st.markdown("""
<style>
body {
    color: yellow;
    background-color: black;
}
h1, h2, h3, h4, h5, h6 {
    color: yellow;
}
</style>
""", unsafe_allow_html=True)

cryptos = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "SOL-USD", "ADA-USD", "XRP-USD", "LUNA1-USD",
           "HEX-USD", "AVAX-USD", "DOT-USD", "DOGE-USD", "SHIB-USD", "MATIC-USD", "ATOM-USD", "LTC-USD"]

selected = st.sidebar.selectbox("Navigation", ["Home", "Time Series Analysis", "Indicators", "Plan my purchase"])

start = '2011-01-01'
end = '2022-01-01'
today = date.today().strftime("%Y-%m-%d")

if selected == "Home":
    st.title("TILL DATE DATA")
    target_crypto = st.selectbox("ENTER THE CRYPTO TO BE PREDICTED", cryptos)

    def load_data(ticker):
        data = yf.download(ticker, start, today)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("loading data...")
    data = load_data(target_crypto)
    data_load_state.text("data loaded successfully....")

    st.subheader("DATA FROM 2011 TO CURRENT DAY")
    st.dataframe(data.tail(), width=None)

    def org_graph():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text="TILL DATE DATA", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    org_graph()

    st.header("Lets understand the data")
    st.subheader("Open")
    st.write("Open indicates the price of cryptocurrency at the beginning of the day")
    st.subheader("Close")
    st.write("Close indicates the price of cryptocurrency at the end of the day")
    st.subheader("Low")
    st.write("Low indicates the least price of the cryptocurrency recorded in the whole day")
    st.subheader("High")
    st.write("High indicates the highest price of the cryptocurrency recorded in the whole day")
    st.subheader("Volume")
    st.write("Volume indicates the amount of cryptocurrency recorded at the end of the day")

    max_price_date = data.loc[data['High'].idxmax()]['Date']
    min_price_date = data.loc[data['Low'].idxmin()]['Date']

    st.subheader("Highest and Lowest Prices")
    st.write(f"Highest price ({data['High'].max()}) occurred on: {max_price_date}")
    st.write(f"Lowest price ({data['Low'].min()}) occurred on: {min_price_date}")

if selected == "Time Series Analysis":
    target_crypto = st.sidebar.selectbox("ENTER THE CRYPTO TO BE PREDICTED", cryptos)

    def load_data(ticker):
        data = yf.download(ticker, start, today)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("loading data...")
    data = load_data(target_crypto)
    data_load_state.text("data loaded successfully....")

    timeline = ("WEEKS", "MONTHS", "YEARS")
    duration = st.sidebar.selectbox("ENTER THE DURATION TO PREDICT", timeline)
    N = st.sidebar.slider("ENTER NUMBER", 0, 10)

    if duration == "WEEKS":
        no_of_days = N * 7
    elif duration == "MONTHS":
        no_of_days = N * 30
    elif duration == "YEARS":
        no_of_days = N * 365

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = MakeFuture()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=no_of_days)
    forecast = m.predict(future)

    st.subheader('PREDICTED DATA')
    st.dataframe(forecast.tail(), width=None)
    st.subheader('FORECAST DATA')
    # m=MakeFuture()
    
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1, use_container_width=True)
    st.write("FORECAST COMPONENTS")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

if selected == "Indicators":
    
    target_crypto = st.sidebar.selectbox("ENTER THE CRYPTO TO BE ANALYZED", cryptos)

    def load_data(ticker):
        data = yf.download(ticker, start, today)
        return data
    def prepare_prophet_data(data):
        df = data.reset_index()
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        return df

    def forecast_with_prophet(data, periods):
        m = MakeFuture()
        m.fit(data)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast
    data_load_state = st.text("loading data...")
    data = load_data(target_crypto)
    data_load_state.text("data loaded successfully....")

    # Calculate RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

    # Calculate Moving Averages
    data['MA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['MA_200'] = ta.trend.sma_indicator(data['Close'], window=200)

    # Calculate Bollinger Bands
    data['BB_High'], data['BB_Mid'], data['BB_Low'] = ta.volatility.bollinger_hband(data['Close'], window=20), \
                                                       ta.volatility.bollinger_mavg(data['Close'], window=20), \
                                                       ta.volatility.bollinger_lband(data['Close'], window=20)

    st.subheader(f"Technical Indicators for {target_crypto}")
    st.dataframe(data.tail(), width=None)

    st.subheader("RSI (Relative Strength Index)")
    st.line_chart(data['RSI'], use_container_width=True)

    st.subheader("Moving Averages (50-day and 200-day)")
    st.line_chart(data[['Close', 'MA_50', 'MA_200']], use_container_width=True)

    st.subheader("Bollinger Bands")
    st.line_chart(data[['Close', 'BB_High', 'BB_Mid', 'BB_Low']], use_container_width=True)

    # Prediction based on indicators
    st.subheader("Prediction based on Indicators")

    # Here you can integrate your prediction model using the indicator values
    df_train = prepare_prophet_data(data)

    forecast = forecast_with_prophet(df_train, periods=7)  # Forecast for next 1 week
    st.subheader("Forecast based on Indicators for next 1 week")
    st.dataframe(forecast.tail(7), width=None)

if selected == "Plan my purchase":
    target_crypto = st.sidebar.selectbox("ENTER THE CRYPTO TO BE PREDICTED", cryptos).replace('-', "") + "T"

    def load_data(ticker):
        data = yf.download(ticker, start='2019-01-01')
        return data

    data_load_state = st.text("loading data...")
    data = load_data(target_crypto)
    data_load_state.text("data loaded successfully....")

    key = "https://api.binance.com/api/v3/ticker/price?symbol="
    url = key + target_crypto
    data = requests.get(url)
    data = data.json()
    currprice = float(data['price'])
    money = st.number_input("ENTER THE MONEY YOU WANT TO INVEST IN DOLLARS", min_value=10.00, step=1.00)
    amount = money / currprice
    st.write("YOU WILL HAVE ", amount, " OF ", target_crypto, " IN YOUR WALLET")
