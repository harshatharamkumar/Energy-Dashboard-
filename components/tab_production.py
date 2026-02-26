import streamlit as st


def render(df_view, df_full, fdf):
    p1, p2, p3 = st.columns(3)
    p1.metric("Avg Units / Clutch",           f"{df_view['Units_per_Clutch'].mean():.2f}")
    p2.metric("Avg Diesel / Clutch (L)",      f"{df_view['Diesel_per_Clutch'].mean():.2f}")
    p3.metric("Avg Energy Cost / Clutch (₹)", f"{df_view['Energy_Cost_Per_Clutch'].mean():.2f}")

    st.subheader("AI Recommendation")
    _render_recommendation(fdf)


def _render_recommendation(fdf):
    avg_renewable = fdf["Renewable_Forecast"].mean()
    avg_demand    = fdf["Industrial_Demand"].mean()
    shortfall     = avg_demand - avg_renewable
    shortfall_pct = (shortfall / avg_demand * 100) if avg_demand > 0 else 0
    surplus       = -shortfall

    # CASE 1: Fully renewable
    if shortfall <= 0:
        surplus_kwh = surplus * len(fdf)
        st.success(
            f"✅ Renewable supply fully covers projected demand. "
            f"Estimated surplus of **{surplus_kwh:,.0f} kWh** over the forecast period — "
            f"consider selling excess back to the grid or charging battery storage."
        )
        return

    # CASE 2: Shortfall — tiered severity
    total_grid_cost = fdf["Grid_Cost"].sum()
    peak_shortfall  = (fdf["Industrial_Demand"] - fdf["Renewable_Forecast"]).clip(lower=0).max()
    worst_day       = (fdf["Industrial_Demand"] - fdf["Renewable_Forecast"]).clip(lower=0).idxmax()
    worst_date      = fdf.loc[worst_day, "Date"].strftime("%d %b")

    if shortfall_pct <= 20:
        severity_label, severity_fn = "🟡 Minor Shortfall",   st.warning
    elif shortfall_pct <= 50:
        severity_label, severity_fn = "🟠 Moderate Shortfall", st.warning
    else:
        severity_label, severity_fn = "🔴 Critical Shortfall", st.error

    severity_fn(
        f"**{severity_label}** — Renewable covers only "
        f"**{100 - shortfall_pct:.1f}%** of projected demand."
    )

    # Actionable recommendations
    st.markdown("**Recommended Actions:**")
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"**⏰ Shift Non-Critical Loads**\n\n"
            f"Schedule high-energy processes (heating, compressors, conveyors) "
            f"during peak solar hours to maximise renewable usage."
        )
        st.info(
            f"**🔋 Evaluate Battery Storage**\n\n"
            f"A buffer of **{peak_shortfall:,.0f} kWh** storage capacity would cover "
            f"the worst forecast day ({worst_date}). "
            f"Storage pays off when grid cost exceeds ₹{total_grid_cost:,.0f} over the period."
        )

    with col2:
        st.info(
            f"**⚡ Optimise Grid Procurement**\n\n"
            f"Projected grid cost: **₹{total_grid_cost:,.0f}** over {len(fdf)} days. "
            f"Negotiate off-peak grid contracts or TOD (Time-of-Day) tariffs to reduce cost."
        )
        st.info(
            f"**☀️ Expand Renewable Capacity**\n\n"
            f"Average daily shortfall is **{shortfall:,.0f} kWh**. "
            f"Adding solar/wind capacity to close this gap would reduce grid dependency long-term."
        )