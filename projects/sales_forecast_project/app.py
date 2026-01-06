import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

# ---------- Configuration ----------
DATA_PATH = "train.csv"   # adjust if needed
DATE_COL = 'Order Date'
SALES_COL = 'Sales'

# ---------- Load & preprocess ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=[DATE_COL])
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors='coerce').fillna(0.0)
    return df

raw_df = load_data()

# helper to get unique groups
def get_unique_values(level):
    if level == 'Overall':
        return ['Overall']
    if level == 'Category':
        return sorted(raw_df['Category'].dropna().unique().tolist())
    if level == 'Sub-Category':
        return sorted(raw_df['Sub-Category'].dropna().unique().tolist())
    if level == 'Product':
        prod_sales = raw_df.groupby('Product Name')[SALES_COL].sum().sort_values(ascending=False)
        return prod_sales.head(200).index.tolist()
    return []

# aggregate helper
def aggregate_monthly(df, level, group_value=None):
    d = df.copy()
    d['Order_Month'] = d[DATE_COL].dt.to_period('M').dt.to_timestamp()
    if level == 'Overall':
        gs = d.groupby('Order_Month')[SALES_COL].sum().reset_index()
        gs.columns = ['ds', 'y']
        return gs
    col = 'Category' if level=='Category' else ('Sub-Category' if level=='Sub-Category' else 'Product Name')
    gs = d[d[col]==group_value].groupby('Order_Month')[SALES_COL].sum().reset_index()
    gs.columns = ['ds', 'y']
    return gs

# model cache
MODEL_CACHE = {}
LAST_FORECAST = {}   # store last forecast for download

def fit_prophet(ts_df):
    ts_df = ts_df.sort_values('ds')
    if len(ts_df) < 3:
        idx = pd.date_range(end=ts_df['ds'].max(), periods=6, freq='M')
        tmp = pd.DataFrame({'ds': idx})
        tmp = tmp.merge(ts_df, on='ds', how='left').fillna(0.0)
        ts_df = tmp[['ds','y']]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(ts_df)
    return m

def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='M')
    return model.predict(future)

# ---------- Dash App ----------
app = dash.Dash(__name__)
server = app.server

levels = ['Overall','Category','Sub-Category','Product']

app.layout = html.Div(style={
    "background": "linear-gradient(to right, #e3f2fd, #f9f9f9)",
    "minHeight": "100vh",
    "padding": "25px"
}, children=[

    # Title Section
    html.Div([
        html.H1("📈 Sales Forecasting Dashboard",
                style={"textAlign": "center",
                       "color": "#1A5276",
                       "fontFamily": "Trebuchet MS, sans-serif",
                       "fontSize": "38px",
                       "marginBottom": "10px"}),
        html.P("Category • Sub-Category • Product Level",
               style={"textAlign": "center",
                      "color": "#566573",
                      "fontSize": "18px",
                      "marginBottom": "30px"})
    ]),

    # Controls Section
    html.Div([
        html.Div([
            html.Label("Aggregation Level:", style={"fontWeight": "bold", "color": "#2C3E50"}),
            dcc.Dropdown(
                id="level-dropdown",
                options=[{"label": i, "value": i} for i in levels],
                value="Overall",
                clearable=False,
                style={"borderRadius": "10px", "padding": "6px"}
            ),
        ], style={"flex": "1", "marginRight": "15px"}),

        html.Div([
            html.Label("Select Group:", style={"fontWeight": "bold", "color": "#2C3E50"}),
            dcc.Dropdown(id="group-dropdown", value="Overall",
                         clearable=False,
                         style={"borderRadius": "10px", "padding": "6px"}),
        ], style={"flex": "1"})
    ], style={"display": "flex",
              "gap": "15px",
              "backgroundColor": "white",
              "padding": "20px",
              "borderRadius": "15px",
              "boxShadow": "0px 3px 12px rgba(0,0,0,0.15)",
              "marginBottom": "20px"}),

    # Horizon Slider
    html.Div([
        html.Label("Forecast Horizon (months):",
                   style={"fontWeight": "bold", "color": "#2C3E50"}),
        dcc.Slider(id="horizon-slider", min=1, max=36, step=1, value=12,
                   marks={i: str(i) for i in [1, 6, 12, 24, 36]},
                   tooltip={"placement": "bottom", "always_visible": True}),
    ], style={"backgroundColor": "white",
              "padding": "20px",
              "borderRadius": "15px",
              "boxShadow": "0px 3px 12px rgba(0,0,0,0.15)",
              "marginBottom": "20px"}),

    # Buttons
    html.Div([
        html.Button("🔮 Run Forecast", id="run-button",
                    style={"background": "linear-gradient(45deg, #28a745, #2ecc71)",
                           "color": "white",
                           "fontWeight": "bold",
                           "padding": "14px 30px",
                           "border": "none",
                           "borderRadius": "15px",
                           "cursor": "pointer",
                           "fontSize": "17px",
                           "boxShadow": "0px 5px 12px rgba(0,0,0,0.2)",
                           "marginRight": "15px"}),
        dcc.Download(id="download-data"),
        html.Button("⬇️ Download CSV", id="download-button",
                    style={"background": "linear-gradient(45deg, #007bff, #3498db)",
                           "color": "white",
                           "fontWeight": "bold",
                           "padding": "14px 30px",
                           "border": "none",
                           "borderRadius": "15px",
                           "cursor": "pointer",
                           "fontSize": "17px",
                           "boxShadow": "0px 5px 12px rgba(0,0,0,0.2)"})
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    # Forecast Graph Card
    html.Div([
        dcc.Loading(
            id="loading-icon",
            type="circle",
            children=dcc.Graph(id="forecast-graph", style={"height": "70vh"})
        ),
        html.Div(id="info-div", style={"marginTop": "15px", "textAlign": "center", "color": "#2C3E50"})
    ], style={"backgroundColor": "white",
              "padding": "25px",
              "borderRadius": "20px",
              "boxShadow": "0px 5px 20px rgba(0,0,0,0.2)"}),

    # Footer
    html.Footer("🚀 Developed by SS Infotech | Sales Prediction with AI",
                style={"textAlign": "center",
                       "marginTop": "30px",
                       "color": "#5D6D7E",
                       "fontStyle": "italic",
                       "fontSize": "15px"})
])


# ---------- Callbacks ----------
@app.callback(
    Output('group-dropdown','options'),
    Output('group-dropdown','value'),
    Input('level-dropdown','value')
)
def update_groups(level):
    vals = get_unique_values(level)
    opts = [{'label':v,'value':v} for v in vals]
    default = vals[0] if len(vals)>0 else None
    return opts, default


@app.callback(
    Output('forecast-graph','figure'),
    Output('info-div','children'),
    Input('run-button','n_clicks'),
    State('level-dropdown','value'),
    State('group-dropdown','value'),
    State('horizon-slider','value')
)
def run_forecast(n_clicks, level, group_value, horizon):
    if not n_clicks:
        df = aggregate_monthly(raw_df, 'Overall')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='History'))
        fig.update_layout(title='Historical Monthly Sales (Overall)', xaxis_title='Month', yaxis_title='Sales')
        return fig, 'Click "Run Forecast" to fit model and get forecast.'

    ts = aggregate_monthly(raw_df, level, group_value if level!='Overall' else None)

    cache_key = f"{level}__{group_value}"
    model = MODEL_CACHE.get(cache_key)
    if model is None:
        model = fit_prophet(ts)
        MODEL_CACHE[cache_key] = model

    fcst = make_forecast(model, periods=horizon)

    hist = ts.copy(); hist['ds'] = pd.to_datetime(hist['ds'])
    plot_df = fcst[['ds','yhat','yhat_lower','yhat_upper']].copy()

    # Save for download
    LAST_FORECAST['data'] = plot_df.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines+markers', name='History'))
    fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['yhat_upper'], mode='lines',
                             name='Upper', line={'dash':'dash'}, showlegend=False))
    fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['yhat_lower'], mode='lines',
                             name='Lower', line={'dash':'dash'}, fill='tonexty',
                             fillcolor='rgba(0,0,0,0.05)', showlegend=False))
    title = f"Sales Forecast — {level}: {group_value} — Horizon: {horizon} months"
    fig.update_layout(title=title, xaxis_title='Month', yaxis_title='Sales')

    info = f"Model trained on {len(hist)} points. Latest history: {hist['ds'].max().strftime('%Y-%m-%d')}"
    return fig, info


@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    if "data" not in LAST_FORECAST or LAST_FORECAST['data'] is None:
        return None
    df = LAST_FORECAST['data']
    return dcc.send_data_frame(df.to_csv, "forecast.csv", index=False)


if __name__ == '__main__':
    app.run(debug=True, port=8050)
