from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import numpy as np

# Initialize the Dash app
app = Dash(__name__)


# Create some placeholder data
def generate_placeholder_data(center_freq, sample_rate):
    n_points = 1000
    x = np.linspace(
        center_freq - sample_rate / 2, center_freq + sample_rate / 2, n_points
    )
    # Create a mock FFT signal with some peaks
    y = np.exp(-((x - center_freq) ** 2) / (sample_rate / 10) ** 2)  # Main peak
    y += 0.3 * np.exp(
        -((x - (center_freq - sample_rate / 4)) ** 2) / (sample_rate / 20) ** 2
    )  # Side peak
    y += 0.2 * np.exp(
        -((x - (center_freq + sample_rate / 4)) ** 2) / (sample_rate / 20) ** 2
    )  # Side peak
    y += np.random.normal(0, 0.05, n_points)  # Add some noise
    return x, y


# Define base styles for light/dark mode
def get_styles(is_dark):
    return {
        "page": {
            "backgroundColor": "#1a1a1a" if is_dark else "#ffffff",
            "color": "#ffffff" if is_dark else "#000000",
            "minHeight": "100vh",
            "padding": "2rem",
        },
        "control_panel": {
            "backgroundColor": "#2d2d2d" if is_dark else "#f0f0f0",
            "borderRadius": "8px",
            "padding": "1.5rem",
            "marginBottom": "2rem",
            "display": "flex",
            "alignItems": "center",
        },
        "label": {
            "color": "#ffffff" if is_dark else "#000000",
            "marginBottom": "0.5rem",
        },
        "slider": {"backgroundColor": "#3d3d3d" if is_dark else "#e0e0e0"},
        "input": {
            "backgroundColor": "#3d3d3d" if is_dark else "#ffffff",
            "color": "#ffffff" if is_dark else "#000000",
            "borderColor": "#5d5d5d" if is_dark else "#cccccc",
        },
    }


# Define the layout with light/dark mode toggle
app.layout = html.Div(
    [
        # Theme store
        dcc.Store(id="theme-store", data={"dark_mode": True}),
        # Header
        html.H1(
            "CosmoSDR",
            id="header",
            style={"textAlign": "center", "marginBottom": "2rem"},
        ),
        # Controls container
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Center Frequency (MHz)", id="freq-label"),
                        dcc.Slider(
                            id="center-freq-slider",
                            min=10,
                            max=200,
                            value=102.7,
                            marks={i: f"{i}" for i in range(10, 201, 30)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"flex": 1, "marginRight": "1rem"},
                ),
                html.Div(
                    [
                        html.Label("Sample Rate", id="rate-label"),
                        dcc.Input(
                            id="sample-rate-input",
                            type="number",
                            value=1e6,
                            min=1e5,
                            style={"width": "150px"},
                        ),
                    ],
                    style={"marginRight": "1rem"},
                ),
                html.Div(
                    [
                        html.Label("SDR Gain", id="gain-label"),
                        dcc.Checklist(
                            id="auto-gain-check",
                            options=[{"label": "Auto", "value": "auto"}],
                            value=["auto"],
                            className="dark-check",
                        ),
                        dcc.Slider(
                            id="gain-slider",
                            min=0,
                            max=50,
                            value=25,
                            marks={i: f"{i}" for i in range(0, 51, 10)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            disabled=True,
                        ),
                    ],
                    style={"flex": 1},
                ),
                html.Div(
                    [
                        html.Label("Theme", id="theme-label"),
                        dcc.Checklist(
                            id="dark-mode-switch",
                            options=[{"label": "Dark Mode", "value": "dark"}],
                            value=["dark"],
                            inline=True,
                            className="dark-check",
                        ),
                    ],
                    style={"marginLeft": "1rem"},
                ),
            ],
            id="control-panel",
        ),
        # Main plot
        dcc.Graph(id="signal-plot"),
    ],
    id="main-container",
)


# Callbacks
@callback(Output("gain-slider", "disabled"), Input("auto-gain-check", "value"))
def toggle_gain_control(auto_checked):
    return bool(auto_checked)


@callback(Output("theme-store", "data"), Input("dark-mode-switch", "value"))
def update_theme(dark_mode_value):
    # If 'dark' is in the value list, dark mode is enabled
    return {"dark_mode": "dark" in dark_mode_value}


@callback(
    [
        Output("main-container", "style"),
        Output("control-panel", "style"),
        Output("header", "style"),
        Output("freq-label", "style"),
        Output("rate-label", "style"),
        Output("gain-label", "style"),
        Output("theme-label", "style"),
        Output("sample-rate-input", "style"),
        Output("signal-plot", "figure"),
    ],
    [
        Input("center-freq-slider", "value"),
        Input("sample-rate-input", "value"),
        Input("gain-slider", "value"),
        Input("auto-gain-check", "value"),
        Input("theme-store", "data"),
    ],
)
def update_theme_and_plot(center_freq, sample_rate, gain, auto_gain, theme):
    is_dark = theme["dark_mode"]
    styles = get_styles(is_dark)

    # Convert MHz to Hz for center frequency
    center_freq_hz = center_freq * 1e6

    # Generate placeholder data
    x, y = generate_placeholder_data(center_freq_hz, sample_rate)

    # Create the figure
    fig = go.Figure()

    # Use viridis color for the line
    fig.add_trace(
        go.Scatter(
            x=x / 1e6,  # Convert Hz to MHz for display
            y=y,
            mode="lines",
            name="Signal FFT",
            line=dict(color="#21918c"),  # Viridis-like color
        )
    )

    # Update layout based on theme
    fig.update_layout(
        title="Signal FFT",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Magnitude",
        plot_bgcolor="#1a1a1a" if is_dark else "#ffffff",
        paper_bgcolor="#1a1a1a" if is_dark else "#ffffff",
        font_color="#ffffff" if is_dark else "#000000",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        colorway=["#21918c", "#3b528b", "#440154"],  # Viridis colors
    )

    # Update axes
    grid_color = "#2d2d2d" if is_dark else "#e0e0e0"
    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        showline=True,
        linecolor=grid_color,
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        showline=True,
        linecolor=grid_color,
    )

    return [
        styles["page"],
        styles["control_panel"],
        {
            "textAlign": "center",
            "marginBottom": "2rem",
            "color": "#ffffff" if is_dark else "#000000",
        },
        styles["label"],
        styles["label"],
        styles["label"],
        styles["label"],
        {**styles["input"], "width": "150px"},
        fig,
    ]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
