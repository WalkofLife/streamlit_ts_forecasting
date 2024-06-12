import streamlit
import streamlit as st
import pandas as pd
from calendar import month_abbr
from datetime import date
import seaborn as sns
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import datetime as dt
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

st.write('# Forecasting Engine')

st.write("")

# SIde bar
st.sidebar.title("User Input")
st.sidebar.write("""
    This PoC uses the [champane monthly dataset](https://www.kaggle.com/datasets/piyushagni5/monthly-sales-of-french-champagne?resource=download).
You can try on your own monthly dataset by uploading the excel in the mentioned format:
[Example CSV Input File](https://github.com/WalkofLife/streamlit_ts_forecasting/blob/main/input_format_example.csv)
""")

# Collects user Input Features into Dataframe

uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    item_id_name = st.sidebar.selectbox( # Input for what is the timeseries indentifier
        label='Timeseries Identifier',
        options=input_df.columns.unique())

    datetime_identifier = st.sidebar.selectbox( # Input for what is the date column indentifier
        label='Date Column Name',
        options=input_df.columns.unique())

    target_identifier = st.sidebar.selectbox( # Input for what is the target column indentifier
        label='Target Column Name',
        options=input_df.columns.unique())

else:
    input_df = pd.read_csv('input_format_example.csv')
    item_id_name = 'item_id'
    datetime_identifier = 'timestamp'
    target_identifier = 'target'

input_df[datetime_identifier] = pd.to_datetime(input_df[datetime_identifier])

st.sidebar.header("Validating Error")
no_of_forecast = st.sidebar.slider('No of Months to Forecast', 1,5,3)
# training_till = st.sidebar.date_input(label='Training Till')

with st.sidebar.expander('Training Till'):
    max_year = input_df[datetime_identifier].dt.year.max()
    max_month = input_df[datetime_identifier].dt.month.max()
    min_year = input_df[datetime_identifier].dt.year.min()
    min_month = input_df[datetime_identifier].dt.month.min()
    year_selected = st.selectbox("Year", range(max_year, min_year, -1))
    month_abbr = month_abbr[1:] # Dict of 0 to 11 for Jan to Dec
    temp = 11
    report_month_str = st.radio(label="Month", options= month_abbr, index= int(max_month) - 1, horizontal=True)
    month_selected = month_abbr.index(report_month_str) + 1

    training_till = date(year_selected, month_selected, 1)
    st.write(training_till)

def create_line_chart(input_df, date_col, target_col, split_date):
    import matplotlib.dates as mdates
    sns.set_style("whitegrid")     # Set the aesthetic style of the plots
    palette = sns.color_palette("viridis", n_colors=1)    # Create a custom color palette
    fig, ax = plt.subplots(figsize=(12, 6)) # Create the figure and axis objects

    # Split the data into two subsets
    data_before = input_df[input_df[date_col].dt.date <= split_date]
    data_before = data_before[data_before[date_col].dt.date > data_before[date_col].dt.date.max() - relativedelta(years = 2)]
    data_after = input_df[input_df[date_col].dt.date > split_date]

    # For Training Period
    sns.lineplot(
        data=data_before,
        x= date_col,
        y= target_col,
        # hue="event",
        # style="event",
        color='orange',
        markers=True,
        dashes=False,
        # palette=palette,
        ax=ax,
        label='Training Data'
    )

    # For Validation Period
    sns.lineplot(
        data=data_after,
        x= date_col,
        y= target_col,
        # hue="event",
        # style="event",
        color='green',
        markers=True,
        dashes=False,
        # palette=palette,
        ax=ax,
        label='Training Data'
    )


    # Add titles and labels
    ax.set_title("Target Over Time", fontsize=20)
    ax.set_xlabel("Month-Year", fontsize=14)
    ax.set_ylabel("Target", fontsize=14)

    # Format the x-axis to show Year-Month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every month
    fig.autofmt_xdate()  # Rotate the x labels for better readability

    # Display the plot in Streamlit
    ax.legend()
    st.pyplot(fig)
create_line_chart(input_df, date_col = datetime_identifier,target_col = target_identifier ,split_date= training_till)