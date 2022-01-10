import os
from matplotlib import pyplot as plt
from numpy import zeros
from numpy.core.fromnumeric import sort
import pandas as pd

import seaborn as sns
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics 

import helpers


# Plot time of day plots the total count of events during the given time periods. Expects a datagrame with a Time period column.
def plot_time_of_day(df):
    data = helpers.sort_periods(df)
    line = sns.lineplot(x=data['Time period'], y=data['event_datetime'],sort = False,hue = data['event'])
    line.set_title('Total activity of the day')
    line.set_ylabel('Count')
    line.set_xlabel('Time period')
    plt.show()


# Get manova test runs a monova test by testing the types of events vs Time period. 
def get_manova_test(df_with_events):
    sorted_events = {'view' : 0, 'addtocart' : 1, 'transaction' : 2}
    df_with_events['Event value'] = df_with_events['event'].map(sorted_events)
    df_with_events.sort_values('Event value')
    df_with_sorted_events = helpers.sort_periods(df_with_events)
    helpers.sort_events_by_binary(df_with_sorted_events, 'view')
    helpers.sort_events_by_binary(df_with_sorted_events,'addtocart')
    helpers.sort_events_by_binary(df_with_sorted_events,'transaction')
    df_with_sorted_events = df_with_sorted_events.reset_index(drop =True)
    helpers.get_time_of_day(df_with_sorted_events)
    fit = MANOVA.from_formula('view + addtocart + transaction ~ Q("Time period")', data=data_with_sorted_events)
    print(fit.mv_test())


# Plot boxplot shows a boxplot for Hour vs Event value. 
def plot_boxplot(csv):
    #fig, axs = plt.subplots(ncols=3)
    sns.boxplot(data=csv, x='Hour', y='Event value')
    plt.show()


# Get log reg runs a logistic regression over a binary transaction dependent value for the df.
def get_log_reg(df):
    helpers.sort_events_by_binary(df, 'transaction')
    df = helpers.get_time_of_day(df)
    df['Month'] = df['date'].map(lambda x: x.month)
    df['day'] = df['date'].map(lambda x: x.day)
    df['Minute'] = df['event_datetime'].map(lambda x: x.minute)
    df['day_of_week'] = df['date'].map(helpers.get_day)
    sorted_days = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
    df['day_of_week_value'] = df['day_of_week'].map(sorted_days)
    y = df.transaction
    X = df.drop(['visitorid', 'transactionid', 'date', 'timestamp', 'event', 'event_datetime', 'transaction', 'itemid'], axis = 'columns')
    model=sm.Logit(y,csv.day_of_week_value)
    result = model.fit(method='newton')
    # print(result.params)
    # print(result.summary())
    X.head()


# Get mn log runs a multinomial logistic regression over the df
def get_mn_log_reg(df):
    sorted_events = {'view' : 0, 'addtocart' : 1, 'transaction' : 2}
    df['event_value'] = df['event'].map(sorted_events)
    y = df.event_value
    df['Month'] = df['date'].map(lambda x: x.month)
    df['day'] = df['date'].map(lambda x: x.day)
    df['Minute'] = df['event_datetime'].map(lambda x: x.minute)
    df['weekday'] = df['date'].map(lambda x: x.weekday())

    X = df.drop(['visitorid', 'transactionid', 'Time period', 'date', 'timestamp', 'event', 'event_datetime', 'itemid', 'event_value', 'Minute'], axis = 'columns')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size= 0.3)
    model = sm.MNLogit(y_train, sm.add_constant(X_train))
    result = model.fit()
    # print(result.summary())
    # print(result.summary2())

    model1 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
    preds = model1.predict(X_test)
    print('Accuracy Score:', metrics.accuracy_score(y_test, preds))  

# Get time of day bargraph returns a count bargraph over the dataframes's time periods 
def get_time_of_day_bargraph(df_with_time_of_day):
    sns.countplot(x='Time period', hue= 'event', data=df_with_time_of_day)
    plt.show()


# Get day of week bargraph returns a count bargraph over the dataframe's days of the week
def get_day_of_week_bargraph(df_with_day_of_week):
    df_with_day_of_week['day_of_week'] = df_with_day_of_week['date'].map(helpers.get_day)
    sns.countplot(x='day_of_week', hue= 'event', data=df_with_day_of_week)
    plt.show()


if __name__ == "__main__":
    csv = helpers.read_events()
    csv = helpers.get_time_of_day(csv)
    #data = get_manova_test(csv)
    # X = get_log_reg(csv)
    # plot_time_of_day(csv)
    #get_time_of_day_bargraph(csv)
    get_mn_log_reg(csv)