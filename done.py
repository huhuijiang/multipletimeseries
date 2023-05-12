import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import AutoARIMA, ExponentialSmoothing, Prophet
from darts.dataprocessing.transformers import Scaler
from st_aggrid import AgGrid
from darts.metrics import mape, rmse, mse
from sklearn.metrics import mean_absolute_percentage_error
#from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Set up the app layout
st.set_page_config(page_title='Time Series Forecasting', page_icon=':chart_with_upwards_trend:', layout="wide")

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

# Set custom font and size for the page title
st.sidebar.markdown("<h1 style='text-align: center; font-size: 40px; font-family: Lucida Calligraphy, sans-serif;'>Multi-TS</h1>", unsafe_allow_html=True)

#Logo section
st.sidebar.image("fixed.png", use_column_width=True)

# Add a file uploader to allow users to upload their time series data
uploaded_file = st.sidebar.file_uploader('Upload a CSV file', type='csv')

#Footer area
      #Add a header and expander in side bar
        #st.sidebar.markdown('<p class="font">My First Photo Converter App</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
            st.write("""
                This app is a multiple time series forecasting tool built using the Darts library. With this app, users can upload a CSV file containing time series data and use the Exponential Smoothing, Auto ARIMA, and Prophet models to forecast future values.
            """)

            st.write("""
                Exponential Smoothing is a time series forecasting method that uses weighted averages of past observations to predict future values. Auto ARIMA is a machine learning algorithm that automatically selects the best parameters for an ARIMA model to forecast future values. Prophet is a forecasting library developed by Facebook that uses a decomposable time series model to produce forecasts
            """)

            st.write("""
                Darts is a high-level time series forecasting library that provides a range of models and tools for time series analysis and forecasting. It is built on top of popular Python libraries such as Pandas, Scikit-learn, and Tensorflow, and provides a simple and easy-to-use interface for working with time series data.
            """)
st.sidebar.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
st.sidebar.markdown("""
            Made at night by [@george_obaido](https://twitter.com/Geobaido)
            """,
            unsafe_allow_html=True,
            )

# Define the app header
st.header('Multiple Time Series Forecasting')

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Convert the "ds" column to a datetime column
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
    
    # Group the DataFrame by year and sum the passenger counts
    passenger_totals = df.groupby(df['ds'].dt.year)['value'].sum()

     # Create a bar chart of passenger totals by year
    fig, ax = plt.subplots()
    ax.bar(passenger_totals.index, passenger_totals.values)
    ax.set_title('Total Passengers by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Passengers')

    # Display the chart and table in two columns
    col1, col2, col3 = st.columns([10, 10, 5])
    with col1:
        st.pyplot(fig)

    with col2:
        # Create a line plot of passenger totals by year
        fig, ax = plt.subplots()
        ax.plot(passenger_totals.index, passenger_totals.values)
        ax.set_title('Total Passengers by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Passengers')
        st.pyplot(fig)

    with col3:
        #st.subheader("Dataset:")
        # Display the first 10 rows of the dataframe
        st.dataframe(df.head(8))
     
    if 'ds' not in df.columns:
        st.error('Error: The uploaded CSV file does not contain a column named "ds"')
    else:
        df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
        # Convert the DataFrame to a Darts TimeSeries object
        series = TimeSeries.from_dataframe(df, 'ds', 'value')

        # Split the series into training and validation sets
        train, val = series.split_before(pd.Timestamp('19580101'))

        #Exponential Smoothing
        model = ExponentialSmoothing()
        model.fit(train)
        prediction = model.predict(len(val))

        # Plot Exponential Smoothing forecast with heading
        # Define the app header
        st.header('Forecasting Models: Exponential Smoothing, AutoARIMA and Prophet')
        col1, col2, col3 = st.columns([10, 10, 10])
        with col1:
            st.subheader('Exponential Smoothing')
            fig = plt.figure()
            series.plot(label = 'actual')
            prediction.plot(label = 'forecast', lw = 3)
            plt.legend()
            plt.title('Actual vs Forecast')
            st.pyplot(fig)

        #Auto ARIMA
        model_aarima = AutoARIMA()
        model_aarima.fit(train)
        prediction_aarima = model_aarima.predict(len(val))

        # Plot Auto ARIMA forecast with heading
        with col2:
            st.subheader('Auto ARIMA')
            fig = plt.figure()
            series.plot(label = 'actual')
            prediction_aarima.plot(label = 'forecast', lw = 3)
            plt.legend()
            plt.title('Actual vs Forecast')
            st.pyplot(fig)

        #Prophet
        model_prophet = Prophet()
        model_prophet.fit(train)
        prediction_prophet = model_prophet.predict(len(val))

        # Plot Prophet forecast with heading
        with col3:
            st.subheader('Prophet')
            fig = plt.figure()
            series.plot(label = 'actual')
            prediction_prophet.plot(label = 'forecast', lw = 3)
            plt.legend()
            plt.title('Actual vs Forecast')
            st.pyplot(fig)

            # Plot the MAE plot for each model
            # Define the app header
        st.header('Error Metrics: MAPE, MSE, RMSE')
        col1, col2, col3 = st.columns([10, 10, 10])

        models = [ExponentialSmoothing(), Prophet(), AutoARIMA()]

        backtests = [model.historical_forecasts(series,
                            start=.5,
                            forecast_horizon=3)
             for model in models]
        
        with col1:
            fig = plt.figure()
            series.plot(label='actual')
            for i, m in enumerate(models):
                    err = mape(backtests[i], series)
                    backtests[i].plot(lw=3, label='{}, MAPE={:.2f}%'.format(m.__class__.__name__, err))
                    #backtests[i].plot(lw = 3, label = '{}\nMAPE = {:.2f}%\n'.format(m, err))
                    #backtests[i].plot(lw = 3, label = '{}, MAPE = {:.2f}%'.format(m, err))
                    
            plt.title('Backtest with 3-months forecast horizon (MAPE)')
            plt.legend()
            st.pyplot(fig)

        with col2:
            fig = plt.figure()
            series.plot(label='actual')
            for i, m in enumerate(models):
                    err = mse(backtests[i], series)
                    backtests[i].plot(lw=3, label='{}, MSE={:.2f}%'.format(m.__class__.__name__, err))
                    
            plt.title('Backtest with 3-months forecast horizon (MSE)')
            plt.legend()
            st.pyplot(fig)

        with col3:
            fig = plt.figure()
            series.plot(label='actual')
            for i, m in enumerate(models):
                    err = rmse(backtests[i], series)
                    backtests[i].plot(lw=3, label='{}, RMSE={:.2f}%'.format(m.__class__.__name__, err))
                    
            plt.title('Backtest with 3-months forecast horizon (RMSE)')
            plt.legend()
            st.pyplot(fig)


       


       
  
