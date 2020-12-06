import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import pytz
import datetime
import pandas as pd
import numpy as np
from suntime import Sun, SunTimeException

import plotly.io as plt_io

# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

# set the paper_bgcolor and the plot_bgcolor to a new color
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
plt_io.templates["custom_dark"]['layout']['modebar']['bgcolor'] = 'rgba(0,0,0,0)'
plt_io.templates["custom_dark"]['layout']['modebar']['bgcolor'] = 'rgba(0,0,0,0)'

# you may also want to change gridline colors if you are modifying background
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'


def sun_is_up(date_and_time):
    date_and_time = date_and_time.replace(tzinfo=pytz.utc)
    latitude = 51.21
    longitude = 4.42
    sun = Sun(latitude, longitude)
    date = date_and_time.date()

    sunrise = sunset = 0

    try:
        sunrise = sun.get_local_sunrise_time(date)
        sunset = sun.get_local_sunset_time(date)
    except SunTimeException as e:
        print("Error: {0}.".format(e))

    return sunrise + datetime.timedelta(minutes=15) <= date_and_time <= sunset - datetime.timedelta(minutes=15)


def create_solar_dataset():
    solar = pd.read_csv('https://raw.githubusercontent.com/ffabi/VisualAnalysis/main/data/PV_Elec_Gas3.csv', ',')

    solar = solar.rename(columns={"Unnamed: 0": "date"})
    solar = solar.rename(columns={"kWh electricity/day": "grid_usage"})
    solar = solar.rename(columns={"Gas/day": "Gas_mxm"})

    # create separate year, month, day values
    solar["date"] = pd.DatetimeIndex(pd.to_datetime(solar["date"], format="%d/%m/%Y"))
    solar['year'] = pd.to_numeric(solar['date'].dt.strftime('%Y'))
    solar['month'] = pd.to_numeric(solar['date'].dt.strftime('%m'))
    solar['day'] = pd.to_numeric(solar['date'].dt.strftime('%d'))
    solar["month_name"] = solar["date"].apply(lambda x: x.month_name())

    # calculate daily power
    solar = solar.set_index("date")
    solar2 = solar.shift(periods=1, freq='D', axis=0)
    solar['Cumulative_solar_power_shift'] = solar2.loc[:, 'Cumulative_solar_power']
    solar['daily_produced_energy'] = solar['Cumulative_solar_power'].values - solar[
        'Cumulative_solar_power_shift'].values

    solar = solar[(solar["year"] >= 2012)]

    solar = solar.drop(['Cumulative_solar_power'], axis=1)
    solar = solar.drop(['Cumulative_solar_power_shift'], axis=1)
    solar = solar.reset_index()

    solar["is_hot"] = solar["Gas_mxm"] <= 3
    solar["is_hot"] = solar["is_hot"].astype(int)

    solar["consumption"] = solar["grid_usage"] + solar["daily_produced_energy"]

    solar = solar.sort_values(by="date").reset_index(drop=True)

    return solar


def create_weather_dataset():
    weather = pd.read_csv("https://raw.githubusercontent.com/ffabi/VisualAnalysis/main/data/weather_in_Antwerp.csv",
                          ";")

    weather = weather.rename(columns={"Unnamed: 0": "datetime", "barometer": "pressure"})

    weather = weather[weather["temp"].notna()]
    # weather = weather[weather["visibility"].notna()]

    weather["hour"] = weather["clock"].str.split(":", n=1, expand=True)[0].astype(int)
    weather["minute"] = weather["clock"].str.split(":", n=1, expand=True)[1].astype(int)
    weather["datetime"] = pd.to_datetime(
        dict(year=weather["year"], month=weather["month"], day=weather["day"], hour=weather["hour"],
             minute=weather["minute"]))

    weather["pressure"] = weather["pressure"].apply(
        lambda x: x.replace(" mbar", "") if isinstance(x, str) else x).astype(float)
    weather["humidity"] = weather["humidity"].apply(lambda x: x.replace("%", "") if isinstance(x, str) else x).astype(
        float)
    weather["temp"] = weather["temp"].apply(lambda x: x.replace("°C", "") if isinstance(x, str) else x).astype(float)
    weather["wind"] = weather["wind"].apply(lambda x: x.replace(" km/h", "") if isinstance(x, str) else x)
    weather["wind"] = weather["wind"].apply(lambda x: 0 if (isinstance(x, str) and x == "No wind") else x).astype(float)
    weather["visibility"] = weather["visibility"].fillna("99\xa0km").apply(
        lambda x: x.replace("\xa0km", "") if isinstance(x, str) else x).astype(int)

    rain_regex = "rain|precip|snow|drizzle|shower|storm|thunder|sprinkles|sleet"
    cloud_regex = "cloud|fog|overcast|haze"

    weather["is_rainy"] = weather["weather"].str.contains(f"({rain_regex})", regex=True, case=False)
    weather["is_cloudy"] = weather.is_rainy | weather["weather"].str.contains(f"({cloud_regex})", regex=True,
                                                                              case=False) & ~weather[
        "weather"].str.contains("(scattered|passing)", regex=True, case=False)
    weather["is_clear"] = ~weather.is_cloudy

    weather["date"] = weather["datetime"].dt.date

    return weather


def create_dataset():
    solar = create_solar_dataset()
    weather = create_weather_dataset()

    # use weather data only after sunrise and before sunset
    daylight_weather = weather[list(map(sun_is_up, weather["datetime"]))]

    # daily_weather = daylight_weather.resample('D', on='datetime').mean().reset_index()

    c = {'temp': "mean",
         'wind': "mean", 'humidity': "mean", 'pressure': "mean",
         'visibility': "mean", 'is_rainy': "mean",
         'is_cloudy': "mean", 'is_clear': "mean",
         "hour": "count", "month": "mean"
         }
    daily_weather = daylight_weather.resample('D', on='datetime').agg(c).reset_index()
    daily_weather["hour"] /= 2

    daily_weather = daily_weather.rename(columns={"datetime": "date", "hour": "daylight_hours"})

    # no weather data on 31th of each month, so drop it from solar as well
    dropped_solar = solar[solar["day"] != 31]
    dropped_solar = dropped_solar[(dropped_solar["year"] <= 2019)]
    dropped_solar = dropped_solar.drop(columns=["month", "day"], axis=1)

    merged = pd.merge(daily_weather, dropped_solar, on="date")
    merged["unix_time"] = merged["date"].astype(int) / 10 ** 9

    merged = merged[merged["temp"].notna()]
    merged = merged[merged["daily_produced_energy"].notna()]

    return merged


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="Home Assignment")

df = create_dataset()
years = list(map(int, df['year'].unique()))
years += [max(years) + 1]

color = 'rgb(0,214,150)'

template = "custom_dark"

available_indicators = {
    "is_cloudy": "Cloud %",
    "temp": "Temperature (\u00b0C)"
}
color_scales = {
    "is_cloudy": px.colors.sequential.Blues_r,
    "temp": px.colors.sequential.Bluered
}

navbar = dbc.NavbarSimple(
    children=[],
    brand="Home Assignment",
    brand_href="#",
    color="primary",
    dark=True,
    fixed="top"
)

dropdown = dbc.FormGroup(
    [
        html.Div("dummy", className="mt-5"),
        dbc.Label("Color base", html_for="color-base", className="mt-2"),
        dcc.Dropdown(
            id='color-base',
            options=[{'label': v, 'value': k} for k, v in available_indicators.items()],
            value='temp',
            clearable=False
        ),
    ]
)

year_slider = dbc.FormGroup(
    [
        dbc.Label("Select year range", html_for="year-slider"),
        dcc.RangeSlider(
            id='year-slider',
            min=min(years),
            max=max(years),
            value=[min(years), max(years)],
            marks={year: str(year) for year in years},
            allowCross=False,
            pushable=1
        ),
    ]
)

clearness_slider = dbc.FormGroup(
    [
        dbc.Label("Clearness %", html_for="clearness-slider"),
        dcc.RangeSlider(
            id='clearness-slider',
            min=0,
            max=1,
            step=0.01,
            value=[0, 1],
            marks={
                0: {'label': '0%'},
                0.5: {'label': '50%'},
                1: {'label': '100%'}
            }
        ),
    ]
)

form = dbc.Form([dropdown, year_slider, clearness_slider])

app.layout = html.Div([
    navbar,
    dbc.Container(children=[
        dbc.Row(dbc.Col(form)),

        dbc.Row([
            dbc.Col(dcc.Graph(id='weather-conditions'), width=3),
            dbc.Col(dcc.Graph(id='monthly-consumption'), width=9)
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='energy-vs-sunny-hours')),
            dbc.Col(dcc.Graph(id='energy-vs-temp'))
        ]),

        dbc.Row(dbc.Col(dcc.Graph(id='monthly-production')))
    ]),
])


@app.callback(
    Output('weather-conditions', 'figure'),
    Input('clearness-slider', 'value'),
    Input('year-slider', 'value'))
def update_weather_conditions(clearness_values, year_values):
    dff = df[df['year'] >= year_values[0]][df['year'] < year_values[1]]
    dff = dff[dff['is_clear'] >= clearness_values[0]][dff['is_clear'] <= clearness_values[1]]

    dff = dff[["is_clear", "is_cloudy", "is_rainy"]].agg("mean").reset_index()
    dff.columns = ["index", "values"]
    dff["labels"] = ["Clear", "Cloudy", "Rainy"]
    dff["colors"] = ['gold', 'lightgrey', 'blue']

    title = f"Weather conditions"

    fig = px.pie(
        dff,
        values='values',
        names='labels',
        color='labels',
        template=template,
        color_discrete_map={label: color for label, color in zip(dff["labels"], dff["colors"])}
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, title_text=title, margin=dict(l=2, r=2),)

    return fig


@app.callback(
    Output('energy-vs-sunny-hours', 'figure'),
    Input('color-base', 'value'),
    Input('clearness-slider', 'value'),
    Input('year-slider', 'value'))
def update_energy_vs_daylight_hours(color_base, clearness_values, year_values):
    dff = df[df['year'] >= year_values[0]][df['year'] < year_values[1]]
    dff = dff[dff['is_clear'] >= clearness_values[0]][dff['is_clear'] <= clearness_values[1]]

    title = "Daily produced energy vs daylight hours"

    jittered = dff.copy()
    jittered["daylight_hours"] += np.random.normal(0, 0.1, len(jittered["daylight_hours"]))

    fig = px.scatter(
        jittered,
        x="daylight_hours", y="daily_produced_energy",
        title=title,
        template=template,
        trendline="lowess",
        opacity=0.55,
        color=color_base,
        color_continuous_scale=color_scales[color_base],
    )

    fig.update_xaxes(title_text="Daylight hours", rangemode="nonnegative")
    fig.update_yaxes(title_text="Daily produced energy (kWh)", rangemode="nonnegative")

    # fig.update_layout(legend_title_text='Cloudy %')

    return fig


@app.callback(
    Output('energy-vs-temp', 'figure'),
    Input('color-base', 'value'),
    Input('clearness-slider', 'value'),
    Input('year-slider', 'value'))
def update_energy_vs_temperature(color_base, clearness_values, year_values):
    dff = df[df['year'] >= year_values[0]][df['year'] < year_values[1]]
    dff = dff[dff['is_clear'] >= clearness_values[0]][dff['is_clear'] <= clearness_values[1]]

    title = "Produced energy vs temperature"

    dff["daily_produced_energy_normalized"] = dff["daily_produced_energy"] / dff["daylight_hours"] * dff["is_cloudy"]

    fig = px.scatter(
        dff,
        title=title,
        x="temp", y="daily_produced_energy_normalized",
        trendline="lowess",
        template=template,
        opacity=0.55,
        color=color_base,
        color_continuous_scale=color_scales[color_base],
    )

    fig.update_xaxes(title_text=f"Temperature (\u00b0C)", rangemode="nonnegative")
    fig.update_yaxes(title_text="Produced energy (kW)", rangemode="nonnegative")

    # fig.update_layout(legend_title_text='Cloudy %')

    return fig


@app.callback(
    Output('monthly-production', 'figure'),
    Input('clearness-slider', 'value'),
    Input('year-slider', 'value'))
def update_monthly_production(clearness_values, year_values):
    dff = df[df['year'] >= year_values[0]][df['year'] < year_values[1]]
    dff = dff[dff['is_clear'] >= clearness_values[0]][dff['is_clear'] <= clearness_values[1]]

    title = "Solar energy production by month"

    fig = px.violin(
        dff,
        title=title,
        x="month_name", y="daily_produced_energy",
        template=template,
        box=True,
        points="all"
    )

    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Daily produced energy (kWh)")
    fig.update_traces(marker_color=color)

    return fig


@app.callback(
    Output('monthly-consumption', 'figure'),
    Input('clearness-slider', 'value'),
    Input('year-slider', 'value'))
def update_monthly_consumption(clearness_values, year_values):
    dff = df[df['year'] >= year_values[0]][df['year'] < year_values[1]]
    dff = dff[dff['is_clear'] >= clearness_values[0]][dff['is_clear'] <= clearness_values[1]]

    title = "Summary of daily solar energy production, total energy consumption and gas consumption by month"

    fig = go.Figure()

    grouped_mean = dff.groupby("month_name", as_index=False).mean()
    grouped_mean = grouped_mean.sort_values(by="month")

    fig.add_bar(
        x=grouped_mean["month_name"],
        y=grouped_mean["grid_usage"],
        name="Grid usage (kWh)"
    )

    fig.add_bar(
        x=grouped_mean["month_name"],
        y=grouped_mean["Gas_mxm"],
        name="Gas usage (m\u00b3)"
    )

    fig.add_bar(
        x=grouped_mean["month_name"],
        y=grouped_mean["daily_produced_energy"],
        name="Daily produced energy (kWh)")

    fig.update_layout(template=template, title=title)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="kWh / m\u00b3")

    return fig


if __name__ == "__main__":
    debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
    app.run_server(host="0.0.0.0", port=8050, debug=True)
