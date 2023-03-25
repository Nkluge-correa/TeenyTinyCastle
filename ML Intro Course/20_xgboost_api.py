import uvicorn
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import timedelta
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import StandardScaler

app = FastAPI()

scaler = StandardScaler()

"""
To run this script, you will need to install
`fastapi` and `uvicorn`. All other libraries
should be part of the requirements described
by the `requirements.txt` from this repository.
"""

FEATURES = ['difference_1', 'difference_2', 'difference_3',
            'difference_4', 'difference_5', 'difference_6',
            'difference_7', 'moving_average_week',
            'moving_average_two_weeks', 'difference_year',
            'day_of_week', 'day_of_year', 'quarter', 'month', 'year']

TARGET = ['sales']


def create_dataframe(list1, list2):
    """
    Create a Pandas DataFrame from two lists.

    Parameters:

        list1 (list): A list of dates.
        list2 (list): A list of sales figures.
    Returns:

        df (Pandas DataFrame): A DataFrame object containing 
        two columns: 'dates' and 'sales', with each column 
        representing the corresponding input list.
    """
    data = {'dates': list1, 'sales': list2}

    df = pd.DataFrame(data)

    return df


def create_sales_features(df):
    """
    Creates new features based on the `sales` column 
    of the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to create 
        features for.

    Returns:
        pandas.DataFrame: A new DataFrame with the original 
        columns and additional columns for each feature created.
    """
    df = df.copy()

    df.loc[df['sales'] > df.sales.mean() + (df.sales.std() * 3),
           'sales'] = df.sales.mean() + (df.sales.std() * 3)

    previous = df.sales.shift(1)
    df['difference_1'] = df.sales - previous

    df['moving_average_week'] = df.sales.rolling(window=7).mean()

    df['moving_average_two_weeks'] = df.sales.rolling(window=14).mean()

    for i in range(1, 7):
        column = 'difference_' + str(i+1)
        df[column] = df['difference_1'].shift(i)

    df['difference_year'] = df.sales - df.sales.shift(366)

    df = df.dropna()

    return df


def create_time_features(df):
    """
    Extracts various time-related features from a DataFrame 
    containing time-series data and returns the updated DataFrame.

    Args:
        - df (pandas.DataFrame): The DataFrame containing time-series 
        data to process. This DataFrame must have a 'dates' column 
        with datetime values.

    Returns:
        - pandas.DataFrame: The updated DataFrame with additional 
        time-related features added as columns.
    """

    df = df.copy()

    df = df.set_index('dates')

    df.index = pd.to_datetime(df.index)

    df['day_of_week'] = df.index.day_of_week
    df['day_of_year'] = df.index.day_of_year
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year

    df = df.sort_index()

    return df


def scale_dataset(df):
    """
    This function is used to scale all feature values 
    of a given dataset. It performs feature scaling 
    which transforms each feature so that it has a mean 
    of 0 and a standard deviation of 1.

    Parameters:

        - df : pandas DataFrame The input dataset that needs 
        to be scaled. It should contain columns for features 
        and a target column.

    Returns:

        - df : pandas DataFrame A new DataFrame with scaled 
        feature values and original product_id and sales columns. 
    """

    df = df.copy()

    features = df.drop(['sales'], axis=1)
    target = df[['sales']]

    scaler.fit(features)

    features = pd.DataFrame(
        scaler.transform(features),
        columns=features.columns,
        index=features.index)

    df = pd.concat([target, features], axis=1)

    return df


def train_model(df):
    """
    Trains a gradient boosting regression model on the input DataFrame.

    Args:
        df: A pandas DataFrame containing two columns, 'dates' and 'sales'.

    Returns:
        A trained XGBoost model.
    """
    model = xgb.XGBRegressor(n_estimators=1000, booster='gbtree',
                             max_depth=2,
                             learning_rate=0.1)

    train_df = create_sales_features(df)
    train_df = create_time_features(train_df)
    train_df = scale_dataset(train_df)

    x_features = train_df[FEATURES]
    y_target = train_df[TARGET]

    model.fit(x_features, y_target,
              eval_set=[(x_features, y_target)],
              verbose=100)

    return model


def generate_forecast(model, df, ahead):
    """
    Generates a forecast for future sales based 
    on a time-series dataframe.

    Parameters:
    -----------
    df: pandas.DataFrame
        A time-series dataframe with dates as index and sales as a column.
    ahead: int
        The number of future periods to forecast.

    Returns:
    --------
    pandas.DataFrame
        A dataframe with the forecasted sales for the next `ahead` periods.
    """

    df = df.set_index('dates')
    df.index = pd.to_datetime(df.index)

    for i in range(ahead):

        future_date = df.index.max() + timedelta(days=1)

        future_dates = pd.date_range(start=future_date.strftime("%Y-%m-%d"),
                                     end=future_date.strftime("%Y-%m-%d"))

        future_df = pd.DataFrame(index=future_dates)
        present_df = df[['sales']].copy()

        df_with_future = pd.concat([present_df, future_df]).reset_index().rename(
            columns={"index": "dates"}).fillna(0)
        df_with_future = create_sales_features(df_with_future)
        df_with_future = create_time_features(df_with_future)

        pred = model.predict(df_with_future.tail(1)[FEATURES])

        future_df['sales'] = abs(np.round(pred))

        df = pd.concat([present_df, future_df])

    return df.tail(ahead)


def api_call(list1, list2, ahead):
    """
    This function combines the other functions in this 
    module to generate a sales forecast

    Parameters:
    -----------
        `list1` and `list2` are lists of equal length containing 
        dates and sales values respectively.

        `ahead` is the number of time units to forecast ahead, 
        which should be a positive integer.

    Returns
    -----------

    A tuple of three items:
        - a list of forecasted dates in the format "YYYY-MM-DD"
        - a list of forecasted sales values
        - a dictionary of statistical values for the sales data, including mean, minimum,
            maximum, variance, and standard deviation.
    """
    df = create_dataframe(list1, list2)

    statistics = dict(mean=df.sales.mean(),
                      minimum=df.sales.min(),
                      maximum=df.sales.max(),
                      variance=df.sales.var(),
                      std=df.sales.std(),)

    model = train_model(df)

    forecast = generate_forecast(model, df, abs(ahead))

    return list(forecast.index.strftime("%Y-%m-%d")), list(forecast.sales), statistics


class InputData(BaseModel):
    product: str
    dates: List[str]
    sales: List[float]
    ahead: int


@app.get("/", response_class=HTMLResponse)
async def read_items():
    html_content = """
    <!DOCTYPE html>
    <html>

    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teeny-Tiny API 🏰</title>
    </head>

    <body>
    <h1 id="teeny-tiny-api-🏰">Teeny-Tiny API 🏰</h1>
    <p>This is an example of an ML model (<code>XGBRegressor</code>), with a particular pre-processing technique and configuration, that can be used for regression/forecasting tasks, served as an API.</p>
    <h2 id="how-to-send-requests">How to send <code>requests</code></h2>
    <pre class=" language-python"><code class="prism  language-python">
    <span class="token keyword">import</span> requests
    <span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd

    <span class="token comment"># The data you are going to use</span>
    df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'time_series_data.csv'</span><span class="token punctuation">)</span>

    <span class="token comment"># The API endpoint URL</span>
    url <span class="token operator">=</span> <span class="token string">"https://teeny-tiny-api.onrender.com/predict"</span>

    <span class="token comment"># Define the input data</span>
    data <span class="token operator">=</span> <span class="token punctuation">{</span>
        <span class="token string">"product"</span><span class="token punctuation">:</span> df<span class="token punctuation">.</span>product_id<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token comment"># name of the product</span>
        <span class="token string">"dates"</span><span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>dates<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token comment"># the list of dates</span>
        <span class="token string">"sales"</span><span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>sales<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token comment"># the list of sales</span>
        <span class="token string">"ahead"</span><span class="token punctuation">:</span> <span class="token number">15</span> <span class="token comment"># how many days ahead we want to look</span>
    <span class="token punctuation">}</span>

    <span class="token comment"># Send a POST request with the input data</span>
    response <span class="token operator">=</span> requests<span class="token punctuation">.</span>post<span class="token punctuation">(</span>url<span class="token punctuation">,</span> json<span class="token operator">=</span>data<span class="token punctuation">)</span>

    <span class="token comment"># Done!</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span>response<span class="token punctuation">.</span>json<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

    </code></pre>
    <p>Return to the <a href="https://github.com/Nkluge-correa/teeny-tiny_castle">castle</a>.</p>

    </body>

    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
async def predict(data: InputData):
    days, sales, statistics = api_call(data.dates, data.sales, data.ahead)
    return {data.product:
            {"dates": days,
             "sales": sales,
             "statistics": statistics}}

if __name__ == "__main__":
    uvicorn.run(app)
