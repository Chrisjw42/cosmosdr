import plotly.graph_objects as go
from cosmosdr.config import COLORS


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
