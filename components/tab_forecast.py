import streamlit as st
import plotly.graph_objects as go
from config import RENEWABLE_COL


def render(df_view, fdf):
    actual = df_view.set_index("Date")[RENEWABLE_COL]

    fig = go.Figure([
        go.Scatter(x=actual.index,  y=actual,                    mode="lines", name="Actual"),
        go.Scatter(x=fdf["Date"],   y=fdf["Renewable_Forecast"], mode="lines", name="Forecast",
                   line=dict(dash="dash")),
    ])
    fig.update_layout(title="Renewable Energy Forecast", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Energy Allocation Plan")
    st.dataframe(
        fdf.style.format({
            "Renewable_%":       "{:.1f}%",
            "Total_Energy_Cost": "₹{:,.0f}",
        }),
        use_container_width=True,
    )
