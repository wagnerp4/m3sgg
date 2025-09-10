"""Dark mode styling component for Streamlit app.

This module provides dark and light mode styling for the M3SGG Streamlit application.
"""

import streamlit as st


def apply_dark_mode_styles():
    """Apply dark mode styling to the Streamlit app.

    This function applies comprehensive dark mode styling including:
    - Main app background and text colors
    - Sidebar styling
    - Form elements (inputs, buttons, selectboxes)
    - Data visualization components
    - Alert and notification styling
    """
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 3rem;
                color: #4a9eff;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #2d3748;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #4a9eff;
                color: #e2e8f0;
            }
            .stAlert {
                margin-top: 1rem;
            }
            .stApp {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .stSidebar {
                background-color: #2d3748;
            }
            .stSelectbox > div > div {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stTextInput > div > div > input {
                background-color: #2d3748;
                color: #e2e8f0;
                border-color: #4a5568;
            }
            .stTextArea > div > div > textarea {
                background-color: #2d3748;
                color: #e2e8f0;
                border-color: #4a5568;
            }
            .stButton > button {
                background-color: #4a9eff;
                color: #1a202c;
                border: none;
            }
            .stButton > button:hover {
                background-color: #3182ce;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #2d3748;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4a9eff;
                color: #1a202c;
            }
            .stDataFrame {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stExpander {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stMarkdown {
                color: #e2e8f0;
            }
            .stSuccess {
                background-color: #2d5016;
                color: #9ae6b4;
            }
            .stError {
                background-color: #742a2a;
                color: #feb2b2;
            }
            .stWarning {
                background-color: #744210;
                color: #fbd38d;
            }
            .stInfo {
                background-color: #2c5282;
                color: #90cdf4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_light_mode_styles():
    """Apply light mode styling to the Streamlit app.

    This function applies clean light mode styling including:
    - Main app background and text colors
    - Metric card styling
    - Alert styling
    """
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #1f77b4;
            }
            .stAlert {
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme_styles():
    """Apply theme styles based on the current dark mode setting.

    This function checks the session state for dark mode preference
    and applies the appropriate styling.
    """
    if st.session_state.get("dark_mode", False):
        apply_dark_mode_styles()
    else:
        apply_light_mode_styles()
