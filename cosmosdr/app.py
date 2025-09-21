import numpy as np
import structlog
import time

from dash import (
    Dash,
    html,
    dcc,
    callback,
    Output,
    Input,
    State,
    no_update,
    callback_context,
)
import plotly.graph_objects as go

from signal_acquisition import signal_streamer
from signal_processing import get_frequency_space_np

# How often the stream plot redraws
UPDATE_FREQUENCY_HZ = 5
DEFAULT_FREQUENCY = 1090

logger = structlog.get_logger()

# Initialize the Dash app
app = Dash(__name__)

# Define the app styles
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .stream-button:hover {
                background-color: #3d3d3d !important;
                border-color: #ffffff !important;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .stream-button {
                transition: all 0.2s ease-in-out;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


class StreamParams:
    def __init__(self):
        self.center_freq = None
        self.sample_rate = None
        self.gain = None
        self.auto_gain = None
        self.active = False

    def update(self, center_freq=None, sample_rate=None, gain=None, auto_gain=None):
        if center_freq is not None:
            self.center_freq = center_freq
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if gain is not None:
            self.gain = gain
        if auto_gain is not None:
            self.auto_gain = auto_gain

    def are_valid(self):
        """Check if all parameters are valid for streaming"""
        if self.sample_rate is None or self.sample_rate < 1e5:
            return False
        if self.center_freq is None or self.center_freq < 10 or self.center_freq > 200:
            return False
        if not self.auto_gain and (
            self.gain is None or self.gain < 0 or self.gain > 50
        ):
            return False
        return True


# Initialize stream parameters
stream_params = StreamParams()


# Define base styles for light/dark mode
# Define theme colors and styles
COLORS = {
    "background": {
        "page": "#1a1a1a",
        "control_panel": "#2d2d2d",
        "input": "#3d3d3d",
        "slider": "#3d3d3d",
        "plot": "#1a1a1a",
        "grid": "#2d2d2d",
    },
    "text": {
        "primary": "#ffffff",
        "label": "#ffffff",
    },
    "border": {
        "input": "#5d5d5d",
    },
    "signal": "#21918c",
    "status": {
        "active": "#4CAF50",
        "inactive": "#666",
        "error": "#f44336",
    },
}

# Define base styles using the color configuration
STYLES = {
    "page": {
        "backgroundColor": COLORS["background"]["page"],
        "color": COLORS["text"]["primary"],
        "minHeight": "100vh",
        "padding": "2rem",
        "margin": "0",
        "position": "absolute",
        "top": "0",
        "left": "0",
        "right": "0",
        "bottom": "0",
    },
    "control_panel": {
        "backgroundColor": COLORS["background"]["control_panel"],
        "borderRadius": "8px",
        "padding": "1.5rem",
        "marginRight": "1rem",
        "width": "300px",
        "height": "calc(100vh - 4rem)",  # Full height minus padding
        "position": "fixed",
        "left": "2rem",
        "top": "2rem",
        "overflowY": "auto",
    },
    "main_content": {
        "marginLeft": "calc(300px + 3rem)",  # Control panel width + margin
        "paddingTop": "2rem",
    },
    "control_group": {
        "marginBottom": "2rem",
    },
    "label": {
        "color": COLORS["text"]["label"],
        "marginBottom": "0.5rem",
    },
    "input": {
        "backgroundColor": COLORS["background"]["input"],
        "color": COLORS["text"]["primary"],
        "borderColor": COLORS["border"]["input"],
    },
}


# Add any missing styles for dark mode components
STYLES["input_dark"] = {
    "backgroundColor": COLORS["background"]["input"],
    "color": COLORS["text"]["primary"],
    "borderColor": COLORS["border"]["input"],
    "borderRadius": "4px",
    "padding": "0.5rem",
}

STYLES["slider"] = {
    "color": COLORS["text"]["primary"],
}

STYLES["button"] = {
    "backgroundColor": COLORS["background"]["input"],
    "color": COLORS["text"]["primary"],
    "border": f"1px solid {COLORS['border']['input']}",
    "borderRadius": "4px",
    "padding": "0.5rem 1rem",
    "cursor": "pointer",
    "transition": "all 0.2s ease-in-out",
}


# Create base figure using the color configuration
def create_base_figure():
    # Create initial figure with dark theme
    fig = go.Figure()

    # Add initial trace with a single point to establish the plot
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="lines",
            name="Signal",
            line=dict(
                color=COLORS["signal"],
                width=1,
            ),
        )
    )

    # Set initial layout with dark theme
    fig.update_layout(
        template="plotly_dark",  # Use dark template as base
        title="Real-time Signal",
        xaxis_title="Sample",
        yaxis_title="Magnitude",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor=COLORS["background"]["plot"],
        paper_bgcolor=COLORS["background"]["plot"],
        font_color=COLORS["text"]["primary"],
        uirevision="constant",  # Keep zoom/pan state
    )

    # Axis styling
    grid_color = COLORS["background"]["grid"]
    axis_settings = dict(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        showline=True,
        linecolor=grid_color,
        showgrid=True,
        gridwidth=1,
        linewidth=1,
    )

    # Apply axis styling
    fig.update_xaxes(**axis_settings)
    fig.update_yaxes(
        **axis_settings,
        autorange=False,
        range=[0, 100],
    )

    return fig


# Define the layout
app.layout = html.Div(
    [
        # Side Control Panel
        html.Div(
            [
                html.H1(
                    "CosmoSDR",
                    id="header",
                    style={
                        "textAlign": "center",
                        "marginBottom": "2rem",
                        "color": COLORS["text"]["primary"],
                    },
                ),
                # Frequency Control
                html.Div(
                    [
                        html.Label(
                            "Center Frequency (MHz)",
                            id="freq-label",
                            style=STYLES["label"],
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Input(
                                        id="center-freq-input",
                                        type="number",
                                        value=DEFAULT_FREQUENCY,
                                        min=25,
                                        max=1300,
                                        step=0.1,
                                        style={**STYLES["input_dark"], "width": "100%"},
                                    ),
                                    style={"marginBottom": "0.5rem"},
                                ),
                                dcc.Slider(
                                    id="center-freq-slider",
                                    min=25,
                                    max=1300,
                                    value=DEFAULT_FREQUENCY,
                                    step=1,
                                    # step=(1300 - 25) / 9,  # 10 steps: 9 intervals
                                    marks={
                                        int(25 + i * ((1300 - 25) / 9)): {
                                            "label": str(
                                                int(25 + i * ((1300 - 25) / 9))
                                            ),
                                            "style": {
                                                "color": COLORS["text"]["primary"]
                                            },
                                        }
                                        for i in range(10)
                                    },
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                    className="dark-slider",
                                ),
                            ]
                        ),
                    ],
                    style=STYLES["control_group"],
                ),
                # Sample Rate Control
                html.Div(
                    [
                        html.Label(
                            "Sample Rate (Hz)", id="rate-label", style=STYLES["label"]
                        ),
                        dcc.Input(
                            id="sample-rate-input",
                            type="number",
                            value=2.4e6,
                            min=1e5,
                            style={**STYLES["input_dark"], "width": "100%"},
                        ),
                    ],
                    style=STYLES["control_group"],
                ),
                # Gain Control
                html.Div(
                    [
                        html.Label(
                            "SDR Gain (dB)", id="gain-label", style=STYLES["label"]
                        ),
                        dcc.Slider(
                            id="gain-slider",
                            min=0,
                            max=50,
                            value=25,
                            marks={
                                i: {
                                    "label": str(i),
                                    "style": {"color": COLORS["text"]["primary"]},
                                }
                                for i in range(0, 51, 10)
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="dark-slider",
                            disabled=True,
                        ),
                        html.Div(
                            dcc.Checklist(
                                id="auto-gain-check",
                                options=[{"label": "Auto", "value": "auto"}],
                                value=["auto"],
                                className="dark-check",
                                style={"color": COLORS["text"]["primary"]},
                                inputStyle={"marginRight": "0.5rem"},
                                labelStyle={"color": COLORS["text"]["primary"]},
                            ),
                            style={"marginTop": "0.5rem"},
                        ),
                    ],
                    style=STYLES["control_group"],
                ),
                # Stream Controls
                html.Div(
                    [
                        html.Button(
                            "Start Stream",
                            id="stream-control-button",
                            n_clicks=0,
                            style={
                                **STYLES["button"],
                                "width": "100%",
                                "marginBottom": "1rem",
                            },
                            className="stream-button",
                        ),
                        html.Div(
                            "Inactive",
                            id="stream-status",
                            style={
                                "display": "block",
                                "padding": "0.5rem",
                                "borderRadius": "4px",
                                "backgroundColor": COLORS["status"]["inactive"],
                                "color": COLORS["text"]["primary"],
                                "textAlign": "center",
                            },
                        ),
                        dcc.Store(id="stream-error-store", data=""),
                    ],
                    style=STYLES["control_group"],
                ),
                # Error display
                html.Div(
                    id="stream-error-display",
                    style={"color": COLORS["status"]["error"], "marginBottom": "1rem"},
                ),
            ],
            id="control-panel",
            style=STYLES["control_panel"],
        ),
        # Main Content Area
        html.Div(
            [
                # Main plot
                dcc.Graph(
                    id="signal-plot",
                    figure=create_base_figure(),  # Initialize with our base figure
                ),
            ],
            style=STYLES["main_content"],
        ),
        # Update interval
        dcc.Interval(
            id="signal-update-interval",
            interval=1000 / UPDATE_FREQUENCY_HZ,  # ms
            n_intervals=0,
            disabled=True,
        ),
        # Enable a ratchet mechanism to keep plot y-axis reasonable
        dcc.Store(id="y-axis-max", data=1.0),
    ],
    id="main-container",
    style=STYLES["page"],  # Apply dark theme to main container
)


# Callbacks
@callback(
    [Output("center-freq-input", "value"), Output("center-freq-slider", "value")],
    [Input("center-freq-input", "value"), Input("center-freq-slider", "value")],
    prevent_initial_call=True,
)
def sync_freq_controls(input_value, slider_value):
    # Get which input triggered the callback
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "center-freq-input":
        # Input box was changed
        new_value = input_value if input_value is not None else DEFAULT_FREQUENCY
        return new_value, new_value
    else:
        # Slider was changed
        return slider_value, slider_value


@callback(Output("gain-slider", "disabled"), Input("auto-gain-check", "value"))
def toggle_gain_control(auto_checked):
    return bool(auto_checked)


@callback(
    [
        Output("stream-control-button", "children"),
        Output("stream-status", "children"),
        Output("stream-status", "style"),
        Output("signal-update-interval", "disabled"),
        Output("stream-error-store", "data"),
    ],
    Input("stream-control-button", "n_clicks"),
    [
        State("center-freq-slider", "value"),
        State("sample-rate-input", "value"),
        State("gain-slider", "value"),
        State("auto-gain-check", "value"),
        State("stream-control-button", "children"),
    ],
    prevent_initial_call=True,
)
def toggle_stream(n_clicks, center_freq, sample_rate, gain, auto_gain, button_text):
    if not n_clicks:  # Skip if the button hasn't been clicked
        return no_update

    logger.info(
        "Stream toggle requested",
        button_text=button_text,
        center_freq=center_freq,
        sample_rate=sample_rate,
        gain=gain,
        auto_gain=auto_gain,
    )

    is_auto = "auto" in (auto_gain or [])
    sdr_gain = "auto" if is_auto else gain

    try:
        is_starting = button_text == "Start Stream"

        if is_starting:
            # Stop any existing stream first
            signal_streamer.stop_stream()

            # Start new stream with current parameters
            logger.info(
                "Starting stream",
                center_freq_mhz=center_freq * 1e6,
                sample_rate=sample_rate,
                sdr_gain=sdr_gain,
            )
            signal_streamer.start_stream(
                center_freq=center_freq * 1e6,
                sample_rate=sample_rate,
                sdr_gain=sdr_gain,
            )
            logger.info("Stream started successfully")

            # Update stream parameters
            stream_params.update(
                center_freq=center_freq,
                sample_rate=sample_rate,
                gain=gain,
                auto_gain=sdr_gain,
            )
            stream_params.active = True

            return [
                "Stop Stream",  # button text
                "Active",  # status text
                {  # status style
                    "display": "inline-block",
                    "padding": "0.5rem",
                    "borderRadius": "4px",
                    "backgroundColor": "#4CAF50",
                    "color": "white",
                },
                False,  # enable interval
                "",  # clear any errors
            ]
        else:
            signal_streamer.stop_stream()
            stream_params.active = False
            return [
                "Start Stream",  # button text
                "Inactive",  # status text
                {  # status style
                    "display": "inline-block",
                    "padding": "0.5rem",
                    "borderRadius": "4px",
                    "backgroundColor": "#666",
                    "color": "white",
                },
                True,  # disable interval
                "",  # clear any errors
            ]

    except Exception as e:
        logger.exception("Error managing stream")
        return [
            "Start Stream",  # button text
            "Error",  # status text
            {  # status style
                "display": "inline-block",
                "padding": "0.5rem",
                "borderRadius": "4px",
                "backgroundColor": "#f44336",
                "color": "white",
            },
            True,  # disable interval
            str(e),  # store error message
        ]


@callback(
    Output("signal-plot", "extendData"),
    Output("y-axis-max", "data"),
    # Output("signal-plot", "relayoutData"),
    Input("signal-update-interval", "n_intervals"),
    State("y-axis-max", "data"),
)
def stream_signal(n_intervals, y_axis_max):
    if not stream_params.active:
        return no_update

    start_time = time.time()

    signal_data = signal_streamer.current_signal
    if len(signal_data) == 0:
        logger.info("No signal data available")
        return no_update

    logger.info("Processing signal data", data_length=len(signal_data))

    freqs, s_fft, s_mag_fft, s_angle_fft = get_frequency_space_np(
        signal_data, stream_params.center_freq * 1e6, stream_params.sample_rate
    )
    processing_time = time.time() - start_time
    logger.info(
        "Signal processing complete",
        processing_time=processing_time,
        output_length=len(freqs),
    )

    x = freqs
    y = s_mag_fft

    plot_start_time = time.time()

    # Replace instead of extend by "resetting"
    # the third arg clears before extending, so we clear the previous frame's plot
    result = {"x": [x], "y": [y]}, [0], len(x)

    plot_time = time.time() - plot_start_time
    total_time = time.time() - start_time
    logger.info(
        "Plot update complete",
        plot_preparation_time=plot_time,
        total_callback_time=total_time,
        points_plotted=len(x),
    )

    # Update the y-axis max if the new data exceeds it
    candidate_new_y_max = float(np.max(y))
    if candidate_new_y_max > y_axis_max:
        next_y_max = candidate_new_y_max * 1.1  # Add 10% headroom
        logger.info("Updating y-axis max", new_y_max=next_y_max)
    else:
        next_y_max = y_axis_max
        logger.info("Retaining y-axis max", new_y_max=next_y_max)

    return result, next_y_max


# TODO: this y axis rescaling is not working, figure it out
@callback(
    Output("signal-plot", "relayoutData"),
    Input("y-axis-max", "data"),
)
def update_yaxis(y_axis_max):
    if not y_axis_max:
        return no_update
    return {
        "yaxis.range": [0, y_axis_max],
        "yaxis.autorange": False,
    }


@callback(
    Output("stream-control-button", "disabled"),
    [
        Input("center-freq-slider", "value"),
        Input("sample-rate-input", "value"),
        Input("gain-slider", "value"),
        Input("auto-gain-check", "value"),
    ],
)
def check_parameter_changes(center_freq, sample_rate, gain, auto_gain):
    """Validate all parameters"""
    # Check center frequency
    if center_freq is None or center_freq < 25 or center_freq > 1300:
        logger.info("Invalid center frequency", center_freq=center_freq)
        return True

    # Check sample rate
    if sample_rate is None or sample_rate < 1e5:
        logger.info("Invalid sample rate", sample_rate=sample_rate)
        return True

    # Check gain if not in auto mode
    is_auto = "auto" in (auto_gain or [])
    if not is_auto and (gain is None or gain < 0 or gain > 50):
        logger.info("Invalid gain setting", gain=gain, auto_gain=auto_gain)
        return True

    # All parameters are valid
    logger.info(
        "Parameters validated successfully",
        center_freq=center_freq,
        sample_rate=sample_rate,
        gain=gain,
        auto_gain=auto_gain,
    )
    return False


if __name__ == "__main__":
    # Add hidden div for cleanup
    app.layout.children.append(html.Div(id="_", style={"display": "none"}))

    app.run(debug=True, host="0.0.0.0")
