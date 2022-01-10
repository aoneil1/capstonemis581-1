import os
import calendar
import pandas as pd


# Read events takes the events csv and returns an events dataframe  
def read_events():
    # Remember to change file location
    for dirname, _, filenames in os.walk('/Users/unicornkitty/Downloads/archive'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if 'events' in os.path.join(dirname, filename):
                df_events = pd.read_csv(path)
                break 

    return df_events


# Sort periods sorts a dataframe by Time period value
def sort_periods(df_with_time_periods):
    data = pd.DataFrame(df_with_time_periods.groupby(by=['Time period','event'])['event_datetime'].count()).reset_index()
    sorted_periods = {'Dawn': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    data['Time period value'] = data['Time period'].map(sorted_periods)
    data = data.sort_values('Time period value').drop('Time period value',axis=1)
    data_with_sorted_periods = data.reset_index(drop =True)
    return data_with_sorted_periods


# Get day is a helper function that provides a lambda expression that returns week day names. 
def get_day(x):
    day = calendar.day_name[x.weekday()]
    return day


# Get time periods defines an hour of day within the following time periods of day. 
def get_time_periods(hour):
    ## Define each time of day
    if hour >= 4 and hour < 6:
        return 'Dawn'
    elif hour >= 6 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 16:
        return 'Afternoon'
    elif hour >= 16 and hour < 22:
        return 'Evening'
    else:
        return 'Night'


# Get time of day assumes a dataframe with a timestamp column. Returns a dataframe with date, datetime, Time period, and Hour columns.
def get_time_of_day(df_with_timestamp):
    df_with_timestamp = df_with_timestamp.assign(date=pd.Series(df_with_timestamp.datetime.fromtimestamp(i/1000).date()
    for i in df_with_timestamp.timestamp))
    df_with_timestamp = df_with_timestamp.assign(event_datetime=pd.Series(df_with_timestamp.datetime.fromtimestamp(i/1000).time()
    for i in df_with_timestamp.timestamp))
    df_with_timestamp.head()
    df_with_timestamp['Hour'] = df_with_timestamp['event_datetime'].map(lambda x: x.hour)
    df_with_timestamp['Time period'] = df_with_timestamp['Hour'].map(get_time_periods)
    return df_with_timestamp

    
# Sort events by binary maps a dataframe to events by 0 and 1 dependent on the valid_event. 
def sort_events_by_binary(df_with_events, valid_event):
    categories = ['view', 'addtocart', 'transaction']
    categories.remove(valid_event)
    ones = {valid_event : 1}
    zeros = {key: 0 for key in categories}
    df_with_events[valid_event] = df_with_events['event'].map({**zeros, **ones})
    return df_with_events