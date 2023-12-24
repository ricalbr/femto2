import logging
import flask.cli
import threading
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc


flask.cli.show_server_banner = lambda *args: None
log = logging.getLogger('werkzeug').disabled = True

port = 5000

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME, dbc_css],
)
app_thread = threading.Thread(target=app.run, kwargs={'debug': False, 'port': port, 'use_reloader': False})
app_thread.start()

title = []
list_cells = []

header = html.H4(title, className="bg-primary text-white p-2 mb-2 text-center")

dropdown = html.Div(
    [
        dbc.Label("Cells", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            options=list_cells,
            value=list_cells,
            id="dropdown",
            multi=True,
            placeholder="Select a Cell",
            searchable=True,
        ),
    ],
    className="mb-4",
)

shutter_switch = html.Div(
    children=[
        html.Div(dbc.Label("Shutter closed lines", style={'font-weight': 'bold'})),
        html.Span(
            [
                dbc.Label(className="fa fa-eraser", html_for="switch"),
                dbc.Switch(id="switch", value=True, className="d-inline-block ms-1", persistence=True),
                dbc.Label(className="fa fa-pencil", html_for="switch"),
            ]
        ),
    ]
)

download = html.Div(
    children=[
        html.Div(dbc.Label("Export", style={'font-weight': 'bold'})),
        html.Div(dbc.Button("Download as HTML"), id="download"),
        dcc.Download(id='download_1'),
    ]
)

space = html.Br()

controls = dbc.Card([dropdown, shutter_switch, space, download], body=True)

tab1 = dbc.Tab(
    [dcc.Graph(id="device-2d", responsive=True, style={'width': '100%', 'height': '85vh'})],
    label="2D Plot",
    id='tab-2d',
)
tab2 = dbc.Tab(
    [dcc.Graph(id="device-3d", responsive=True, style={'width': '100%', 'height': '85vh'})],
    label="3D Plot",
    id='tab-3d',
)
tabs = dbc.Card(dbc.Tabs([tab1, tab2], id='tabs'))

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col([controls], width=2),
                dbc.Col([tabs], width=10),
            ]
        ),
    ],
    fluid=True,
    className="dbc dbc-ag-grid",
)
