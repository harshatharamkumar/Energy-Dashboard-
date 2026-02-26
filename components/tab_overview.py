import streamlit as st
import plotly.graph_objects as go


def render(fdf):
    st.subheader("Energy & Cost Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Renewable Used (kWh)",     f"{fdf['Renewable_Used'].sum():,.0f}")
    c2.metric("Grid Required (kWh)",      f"{fdf['Grid_Required'].sum():,.0f}")
    c3.metric("Avg Renewable %",          f"{fdf['Renewable_%'].mean():.1f}%")
    c4.metric("Total Energy Cost (INR)",  f"₹{fdf['Total_Energy_Cost'].sum():,.0f}")

    fig = go.Figure([
        go.Bar(x=fdf["Date"], y=fdf["Grid_Cost"],      name="Grid Cost"),
        go.Bar(x=fdf["Date"], y=fdf["Renewable_Cost"], name="Renewable Cost"),
    ])
    fig.update_layout(barmode="stack", height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
