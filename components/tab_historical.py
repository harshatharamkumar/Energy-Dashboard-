import streamlit as st
import plotly.express as px
from config import RENEWABLE_COL


def render(df_view):
    st.plotly_chart(
        px.line(df_view, x="Date", y=RENEWABLE_COL, title="Daily Renewable Generation"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(df_view, x="Date", y="Units_per_Clutch", title="Energy Units per Clutch"),
        use_container_width=True,
    )
