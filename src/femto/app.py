import dash
import logging
import flask.cli
import threading

flask.cli.show_server_banner = lambda *args: None
log = logging.getLogger('werkzeug').disabled = True

port = 5000

app = dash.Dash(__name__)
app_thread = threading.Thread(target=app.run_server, kwargs={'debug': False, 'port': port, 'use_reloader': False})
app_thread.start()

title = []
list_cell = []
app.layout = dash.html.Div(
    children=[
        dash.html.H1(
            id='main-title',
            children=title,
            style={'font-family': 'Arial', 'textAlign': 'center'},
        ),
        dash.html.Div(
            children=[
                dash.html.Div(
                    style={
                        "width": "5%",
                        'display': 'inline-block',
                        'margin-top': '10',
                    },
                ),
                dash.html.Div(
                    children=[
                        dash.html.Label(
                            ['Plot cells'],
                            style={
                                "font-weight": "bold",
                                "font-family": "arial",
                                "height": "5%",
                                'display': 'inline-block',
                                'margin-top': '10',
                            },
                        ),
                        dash.dcc.Dropdown(
                            id='dropdown',
                            options=list_cell,
                            value=list_cell,
                            multi=True,
                            placeholder="Select a Cell",
                            searchable=True,
                            style={
                                "font-weight": "regular",
                                "font-family": "Arial",
                                'margin-top': '10',
                            },
                        ),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                    },
                ),
                dash.html.Div(
                    children=[
                        dash.html.Label(
                            ['Show closed shutter lines:'],
                            style={
                                "font-weight": "bold",
                                "font-family": "arial",
                                "height": "5%",
                                'display': 'inline-block',
                                'margin-top': '10',
                            },
                        ),
                        dash.dcc.Checklist(
                            id='checklist',
                            options=[''],
                            value=[''],
                            style={
                                "font-weight": "regular",
                                "font-family": "Arial",
                            },
                        ),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                    },
                ),
            ]
        ),
        dash.dcc.Graph(
            id='figure-device',
            responsive=True,
            style={'width': '100%', 'height': '85vh'},
        ),
    ]
)
