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
