from __future__ import annotations

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

title: list[str] = []
list_cells: list[str] = []

header = html.H4(title, className="bg-primary text-white fw-bold p-2 mb-2 text-center")

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

shutter_switch = dbc.Row(
    [
        dbc.Col(dbc.Label("Shutter closed lines", style={'font-weight': 'bold'})),
        dbc.Col(
            html.Span(
                [
                    dbc.Label(className="fa fa-eraser", html_for="switch"),
                    dbc.Switch(
                        id="switch",
                        value=True,
                        className="d-inline-block ms-1",
                        persistence=True,
                        input_style={'onColor': 'success'},
                        input_class_name='success',
                    ),
                    dbc.Label(className="fa fa-pencil", html_for="switch"),
                ]
            )
        ),
    ],
    align="center",
    justify="between",
    className="mb-4",
)


download = dbc.Row(
    [
        dbc.Col(dbc.Label("Export as HTML", style={'font-weight': 'bold'})),
        dbc.Col(
            [
                dbc.Button(
                    [dbc.Label(className="fa fa-download", html_for="download"), "  Download"], id="download", size="sm"
                ),
                dcc.Download(id='download_1'),
            ]
        ),
    ],
    align="center",
    justify="between",
    className="mb-4",
)

space = html.Br()

controls = dbc.Card(
    [
        dbc.CardHeader("Plot parameters", className="bg-secondary text-primary fw-bold text-center"),
        dbc.CardBody([dropdown, shutter_switch, space, download]),
    ],
    className="border-primary",
)

# img_path = pathlib.Path('.') / 'utils' / 'logo.png'
# encoded_img = base64.b64encode(open(img_path, 'rb').read())
# logo = dbc.Card(
#         [
#                 dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_img.decode())),
#         ],
#     style={"border": "none", "outline": "black", 'width':'33%'},)
logo = html.Div()

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
tabs = dbc.Card(
    dbc.Tabs(
        [tab1, tab2],
        id='tabs',
        active_tab="tab-0",
    )
)

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
