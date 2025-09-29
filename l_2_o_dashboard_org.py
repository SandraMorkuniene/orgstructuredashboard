import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="L2O Profitability & Process Dashboard (Redesigned)", layout="wide")
st.title("Lead-to-Order (L2O) — Profitability & Process Dashboard")
st.caption("Prototype uses synthetic data.")

# ----------------------
# Data generation
# ----------------------
@st.cache_data
def generate_data(n=500, start_date="2025-01-01"):
    np.random.seed(42)
    start = datetime.fromisoformat(start_date)
    leads = []
    customers = [f"Cust_{i}" for i in range(1,11)]
    sectors = ["Retail","FMCG","Automotive","Electronics","Pharma"]
    regions = ["North","South","East","West"]
    routes = [f"City{a}-City{b}" for a in ["A","B","C","D"] for b in ["E","F","G","H"]]
    fleet_types = ["FTL","LTL","Reefer","Express"]
    teams = ["Team Alpha", "Team Beta", "Team Gamma"]
    managers = {"Team Alpha": "Manager A", "Team Beta": "Manager B", "Team Gamma": "Manager C"}
    sales_reps = {
    "Team Alpha": ["Rep1", "Rep2"],
    "Team Beta": ["Rep3", "Rep4"],
    "Team Gamma": ["Rep5", "Rep6"]
}
    
    for i in range(1, n+1):
        lead_date = start + timedelta(days=int(np.random.exponential(30)))
        team = np.random.choice(teams)
        rep = np.random.choice(sales_reps[team])
        manager = managers[team]
        customer = np.random.choice(customers, p=[0.15,0.15,0.12,0.12,0.1,0.08,0.08,0.08,0.07,0.05])
        sector = np.random.choice(sectors)
        region = np.random.choice(regions)
        route = np.random.choice(routes)
        distance_km = np.random.randint(50, 1500)
        weight_t = np.round(np.random.uniform(0.5, 25),1)
        service = np.random.choice(fleet_types, p=[0.4,0.3,0.2,0.1])
        # Lead to Quote latency
        prob = np.array([0.1,0.25,0.2,0.15,0.1,0.05,0.075,0.05])
        prob = prob / prob.sum()
        lead_to_quote_days = int(np.random.choice([0,1,2,3,4,5,7,10], p=prob))
        quote_date = lead_date + timedelta(days=lead_to_quote_days)

        base_cost = distance_km * (0.6 if service=="FTL" else 0.75 if service=="LTL" else 1.2 if service=="Reefer" else 1.5)
        overhead = base_cost * 0.12
        estimated_cost = base_cost + overhead + np.random.normal(0, 20)
        quoted_price = max(estimated_cost * np.random.uniform(1.08, 1.30), estimated_cost + 50)

        discount = 0.0
        if np.random.rand() < 0.18:
            discount = np.random.uniform(0.01, 0.25)
            quoted_price *= (1 - discount)

        expected_margin = (quoted_price - estimated_cost) / quoted_price
        negotiation_iters = np.random.poisson(0.6)

        approval_flag = True
        approval_level = "Auto"
        if expected_margin < 0.10:
            if np.random.rand() < 0.6:
                approval_flag = False
                approval_level = "Rejected"
            else:
                approval_flag = True
                approval_level = "Manager"
        elif expected_margin < 0.13:
            approval_level = "Manager"

        win_prob = np.clip(0.65 + (expected_margin - 0.12) - (lead_to_quote_days * 0.03), 0.05, 0.95)
        won = np.random.rand() < win_prob
        quote_to_order_days = int(np.random.choice([0,1,2,3,5,7], p=[0.05,0.4,0.25,0.15,0.1,0.05]))
        order_date = quote_date + timedelta(days=quote_to_order_days) if won else None

        actual_cost = estimated_cost + np.random.normal(0, estimated_cost*0.05)
        extra_cost = 0.0
        delay_flag = False
        extra_reason = None
        if won and np.random.rand() < 0.12:
            extra_cost = estimated_cost * np.random.uniform(0.05, 0.25)
            actual_cost += extra_cost
            delay_flag = True
            extra_reason = np.random.choice(["Delay","Empty_Return","Damage","Customs"])

        actual_revenue = quoted_price if won else 0.0
        actual_margin = (actual_revenue - actual_cost)/actual_revenue if won and actual_revenue>0 else None

        leads.append({
            "Lead_ID": f"L{i:05d}",
            "Lead_Date": lead_date.date(),
            "Sales_Team": team,
            "Sales_Rep": rep,
            "Manager": manager,
            "Customer": customer,
            "Customer_Sector": sector,
            "Region": region,
            "Route": route,
            "Distance_km": distance_km,
            "Weight_t": weight_t,
            "Service_Type": service,
            "Lead_to_Quote_Days": lead_to_quote_days,
            "Quote_Date": quote_date.date(),
            "Estimated_Cost": round(float(estimated_cost),2),
            "Quoted_Price": round(float(quoted_price),2),
            "Discount": round(float(discount),3),
            "Expected_Margin": round(float(expected_margin),3),
            "Negotiation_Iterations": int(negotiation_iters),
            "Approval_Flag": approval_flag,
            "Approval_Level": approval_level,
            "Quote_Won": won,
            "Quote_to_Order_Days": quote_to_order_days if won else None,
            "Order_Date": order_date.date() if order_date else None,
            "Planned_Revenue": round(float(quoted_price),2) if won else 0.0,
            "Planned_Cost": round(float(estimated_cost),2) if won else 0.0,
            "Planned_Margin": round(((quoted_price - estimated_cost)/quoted_price) if won else 0,3),
            "Actual_Revenue": round(float(actual_revenue),2) if won else 0.0,
            "Actual_Cost": round(float(actual_cost),2) if won else 0.0,
            "Actual_Margin": round(float(actual_margin),3) if (won and actual_revenue>0) else None,
            "Delay_Flag": delay_flag,
            "Extra_Cost_Reason": extra_reason
        })
    df = pd.DataFrame(leads)
    df["Lead_Month"] = pd.to_datetime(df["Lead_Date"]).dt.to_period("M").astype(str)
    df["Order_Month"] = pd.to_datetime(df["Order_Date"]).dt.to_period("M").astype(str)
    # extra_cost derived column for won orders
    df["Extra_Cost_Impact"] = df.apply(lambda r: (r["Actual_Cost"] - r["Planned_Cost"]) if (r["Quote_Won"] and pd.notnull(r["Actual_Cost"]) and pd.notnull(r["Planned_Cost"])) else 0.0, axis=1)
    return df

# ----------------------
# Load & Filters
# ----------------------
df = generate_data(800)
st.sidebar.header("Filters & Parameters")
date_min = st.sidebar.date_input("Leads since", value=pd.to_datetime(df["Lead_Date"]).min().date())
date_max = st.sidebar.date_input("Leads before", value=pd.to_datetime(df["Lead_Date"]).max().date())
selected_customers = st.sidebar.multiselect("Customer (multi)", options=sorted(df["Customer"].unique()), default=sorted(df["Customer"].unique()))
selected_regions = st.sidebar.multiselect("Region (multi)", options=sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
selected_service = st.sidebar.multiselect("Service Type (multi)", options=sorted(df["Service_Type"].unique()), default=sorted(df["Service_Type"].unique()))
margin_threshold = st.sidebar.slider("Alert Margin Threshold", min_value=0.0, max_value=0.3, value=0.12, step=0.01)

mask = (pd.to_datetime(df["Lead_Date"]) >= pd.to_datetime(date_min)) & (pd.to_datetime(df["Lead_Date"]) <= pd.to_datetime(date_max))
if selected_customers:
    mask &= df["Customer"].isin(selected_customers)
if selected_regions:
    mask &= df["Region"].isin(selected_regions)
if selected_service:
    mask &= df["Service_Type"].isin(selected_service)
fdf = df[mask].copy()

# safe helpers for empty filtered df
def safe_mean(series):
    if series.dropna().shape[0] == 0:
        return None
    return series.dropna().mean()

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Summary","Funnel & Margins","Process Efficiency","Root Causes","Alerts & Export"])

# ----------------------
# Tab 1 — Executive
# ----------------------
with tab1:
    st.subheader("📊 Executive KPIs ")

    pipeline_value = 0.0
    confirmed_revenue = 0.0
    actual_revenue = 0.0
    avg_expected_margin = None
    avg_actual_margin = None

    if not fdf.empty:
        pipeline_value = fdf["Quoted_Price"].sum()
        confirmed_revenue = fdf["Planned_Revenue"].sum()
        actual_revenue = fdf["Actual_Revenue"].sum()
        avg_expected_margin = safe_mean(fdf["Expected_Margin"])
        avg_actual_margin = safe_mean(fdf["Actual_Margin"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Pipeline (Quoted Potential)", f"${int(pipeline_value):,}")
    col2.metric("Planned Revenue (Won)", f"${int(confirmed_revenue):,}")
    col3.metric("Actual Revenue (Executed)", f"${int(actual_revenue):,}")
    col4.metric("Avg Actual Margin", f"{avg_actual_margin:.1%}" if avg_actual_margin is not None else "n/a")
    # Expected vs Actual margin over time
    #st.subheader("Margin Trend & Distribution")
    margins_time = fdf.groupby("Lead_Month").agg(
        expected_margin=("Expected_Margin","mean"),
        actual_margin=("Actual_Margin","mean"),
        planned_revenue=("Planned_Revenue","sum"),
        actual_revenue=("Actual_Revenue","sum")
    ).reset_index()
    fig_margin = go.Figure()
    fig_margin.add_trace(go.Scatter(x=margins_time["Lead_Month"], y=margins_time["expected_margin"], mode="lines+markers", name="Expected Margin"))
    fig_margin.add_trace(go.Scatter(x=margins_time["Lead_Month"], y=margins_time["actual_margin"], mode="lines+markers", name="Actual Margin"))
    fig_margin.update_layout(title="Expected vs Actual Margin (by Lead Month)", xaxis_title="Month", yaxis_title="Margin")
    st.plotly_chart(fig_margin, use_container_width=True)

    st.markdown("---")
    st.write("**Notes:** Pipeline uses quoted prices from the filtered set. Use the Funnel tab to inspect conversion and leakage." )

# ----------------------
# Tab 2 — Funnel & Margins
# ----------------------
with tab2:
    st.subheader("🔻 Funnel (counts & conversion %)")
    if fdf.empty:
        st.info("No data for the selected filters — adjust filters to see funnel and margins.")
    else:
        leads_count = len(fdf)
        quotes_count = fdf.shape[0]
        orders_count = fdf[fdf["Quote_Won"]==True].shape[0]

        funnel_df = pd.DataFrame({
            "stage": ["Leads","Quotes Sent","Orders Won"],
            "count": [leads_count, quotes_count, orders_count],
        })
        funnel_df["conversion_from_prev"] = funnel_df["count"].pct_change().fillna(1)
        funnel_df["conversion_label"] = (funnel_df["conversion_from_prev"]*100).apply(lambda x: f"{x:.0f}%")

    
        # Bar chart for absolute counts with conversion annotations
        fig = go.Figure()
        fig.add_trace(go.Bar(x=funnel_df["stage"], y=funnel_df["count"], text=funnel_df["count"], textposition='auto', name='Count'))
        # add conversion % as annotations above bars (except first)
        annotations = []
        for i, row in funnel_df.iterrows():
            if i>0:
                annotations.append(dict(x=row['stage'], y=row['count']+max(funnel_df['count'])*0.03, text=row['conversion_label'], showarrow=False))
        fig.update_layout(title='Funnel: Lead → Quote → Order (counts with conversion %)')
        fig.update_layout(annotations=annotations)
        st.plotly_chart(fig, use_container_width=True)

        if not fdf.empty:
            teams = fdf["Sales_Team"].unique()
            funnel_data = []
        
            for team in teams:
                team_df = fdf[fdf["Sales_Team"] == team]
                leads_count = len(team_df)
                quotes_count = team_df.shape[0]  # all quotes generated for the team
                orders_count = team_df[team_df["Quote_Won"] == True].shape[0]
                # Conversion rates
                conv_leads_to_quotes = quotes_count / leads_count if leads_count>0 else 0
                conv_quotes_to_orders = orders_count / quotes_count if quotes_count>0 else 0
                conv_leads_to_orders = orders_count / leads_count if leads_count>0 else 0
        
                funnel_data.extend([
                    {"Stage":"Leads","Sales_Team":team,"Count":leads_count,"Conversion":1.0},
                    {"Stage":"Quotes Sent","Sales_Team":team,"Count":quotes_count,"Conversion":conv_leads_to_quotes},
                    {"Stage":"Orders Won","Sales_Team":team,"Count":orders_count,"Conversion":conv_leads_to_orders}
                ])
        
            funnel_team_df = pd.DataFrame(funnel_data)
        
            # Stacked bar chart: x = Stage, y = Count, color = Sales_Team
            fig = px.bar(
                funnel_team_df,
                x="Stage",
                y="Count",
                color="Sales_Team",
                text="Count",
                barmode="stack",
                title="Funnel by Stage & Team (Counts & Conversion %)"
            )

    # Add conversion % annotations for each team (optional)
    annotations = []
    for idx, row in funnel_team_df.iterrows():
        if row["Stage"] == "Orders Won":  # show % above bar for Orders
            annotations.append(dict(
                x=row["Stage"],
                y=row["Count"],
                text=f"{row['Conversion']*100:.1f}%",
                showarrow=False,
                xanchor='center',
                yanchor='bottom'
            ))
    fig.update_layout(annotations=annotations)
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("**Conversion table**")
    st.dataframe(funnel_df.style.format({"count":"{:,}", "conversion_from_prev":"{:.1%}"}), use_container_width=True)

    st.subheader("📈 Margin Distributions by Team, Service & Region")
    if fdf.empty:
        st.info("No margin data to display.")
    else:
        # boxplot by service
        #fig_box_service = px.box(fdf, x='Service_Type', y='Expected_Margin', points='outliers', title='Expected Margin by Service Type')
        #st.plotly_chart(fig_box_service, use_container_width=True)
        fig_box_service_team = px.box(fdf, x='Service_Type', y='Expected_Margin', color='Sales_Team',
                              points='outliers', title='Expected Margin by Service Type & Team')
        st.plotly_chart(fig_box_service_team, use_container_width=True)
        

        # violin by region
        fig_violin_region = px.violin(fdf, x='Region', y='Expected_Margin', box=True, points='outliers', title='Expected Margin Distribution by Region')
        st.plotly_chart(fig_violin_region, use_container_width=True)
        
        discount_by_rep = fdf.groupby(["Sales_Rep", "Sales_Team"]).agg(
            avg_discount=("Discount","mean"),
            avg_expected_margin=("Expected_Margin","mean"),
            total_quotes=("Lead_ID","count")
        ).reset_index().sort_values("avg_discount", ascending=False)
        discount_by_rep["avg_discount_pct"] = discount_by_rep["avg_discount"] * 100
        
        fig_discount = px.scatter(
            discount_by_rep,
            x="avg_discount_pct",
            y="avg_expected_margin",
            size="total_quotes",
            color="Sales_Team",
            hover_data=["Sales_Rep","total_quotes"],
            title="💰 Discount Discipline by Rep: Avg Discount vs Expected Margin",
            labels={"avg_discount":"Average Discount %", "avg_expected_margin":"Average Expected Margin"}
        )
        fig_discount.update_layout(xaxis_tickformat=".2f")
        fig_discount.update_xaxes(dtick=0.5)
        st.plotly_chart(fig_discount, use_container_width=True)



# ----------------------
# Tab 3 — Process Efficiency
# ----------------------
with tab3:
    st.subheader("Cycle Times & Approvals")
    if fdf.empty:
        st.info("No data for process efficiency.")
    else:
        avg_lead_to_quote = safe_mean(fdf["Lead_to_Quote_Days"]) or 0
        avg_quote_to_order = safe_mean(fdf["Quote_to_Order_Days"]) or 0
        col1, col2 = st.columns(2)
        col1.metric("Avg Lead → Quote", f"{avg_lead_to_quote:.1f} days")
        col2.metric("Avg Quote → Order (won)", f"{avg_quote_to_order:.1f} days")

        st.subheader("⚖️ Approval Outcomes")
        #approval_counts = fdf.groupby("Approval_Level").size().reset_index(name="count")
        #st.bar_chart(approval_counts.set_index("Approval_Level"))
        approval_counts = fdf.groupby(["Approval_Level","Sales_Team"]).size().reset_index(name="count")

        fig = px.bar(
            approval_counts,
            x="Approval_Level",
            y="count",
            color="Sales_Team",
            title="⚖️ Approval Outcomes by Team",
            barmode="stack",  # stacked bars
            text="count"
        )
        fig.update_traces(textposition='auto')
        st.plotly_chart(fig, use_container_width=True)
        
    

        st.subheader("💸 Delay & Extra-Cost Impact by Reason")
        # extra cost impact grouped by reason
        extra_cost_df = fdf[fdf['Extra_Cost_Impact']>0].groupby('Extra_Cost_Reason').agg(count=('Lead_ID','count'), total_extra_cost=('Extra_Cost_Impact','sum')).reset_index().sort_values('total_extra_cost', ascending=False)
        if extra_cost_df.empty:
            st.info("No extra-cost events in the filtered set.")
        else:
            fig_extra = px.bar(extra_cost_df, x='Extra_Cost_Reason', y='total_extra_cost', text='count', title='Total Extra Cost Impact by Reason (sum of Actual - Planned)')
            st.plotly_chart(fig_extra, use_container_width=True)
            st.write(extra_cost_df.style.format({"total_extra_cost":"${:,.2f}"}))

# ----------------------
# Tab 4 — Root Causes (multi-select drilldowns)
# ----------------------
with tab4:
    st.subheader("🔍 Root Cause Drilldowns (multi-select)")
    if fdf.empty:
        st.info("No data for drilldowns.")
    else:
        slicer_customers = st.multiselect("Customer", options=sorted(fdf["Customer"].unique()), default=sorted(fdf["Customer"].unique()))
        slicer_routes = st.multiselect("Route", options=sorted(fdf["Route"].unique()), default=sorted(fdf["Route"].unique()))
        slicer_services = st.multiselect("Service Type", options=sorted(fdf["Service_Type"].unique()), default=sorted(fdf["Service_Type"].unique()))
        slicer_team = st.multiselect("Team", options=sorted(fdf["Sales_Team"].unique()), default=sorted(fdf["Sales_Team"].unique()))

        drill = fdf.copy()
        if slicer_customers:
            drill = drill[drill['Customer'].isin(slicer_customers)]
        if slicer_routes:
            drill = drill[drill['Route'].isin(slicer_routes)]
        if slicer_services:
            drill = drill[drill['Service_Type'].isin(slicer_services)]
        if slicer_team:
            drill = drill[drill['Sales_Team'].isin(slicer_team)]

        st.write(f"Filtered set: {len(drill):,} leads — showing top routes by volume")
        profit_by_route = drill.groupby("Route").agg(avg_expected_margin=("Expected_Margin","mean"), avg_actual_margin=("Actual_Margin","mean"), count=("Lead_ID","count"))\
                             .reset_index().sort_values("count", ascending=False).head(20)
        st.dataframe(profit_by_route.style.format({"avg_expected_margin":"{:.1%}","avg_actual_margin":"{:.1%}"}))

        st.write("Sales behavior (filtered)")
        sales_table = drill.groupby("Customer").agg(count_quotes=("Lead_ID","count"), avg_discount=("Discount","mean"), avg_expected_margin=("Expected_Margin","mean")).reset_index().sort_values("count_quotes", ascending=False)
        st.dataframe(sales_table.style.format({"avg_discount":"{:.2%}","avg_expected_margin":"{:.1%}"}))

# ----------------------
# Tab 5 — Alerts & Export
# ----------------------
with tab5:
    st.subheader("🚨 Alerts — prioritized & grouped")
    if fdf.empty:
        st.info("No data for alerts.")
    else:
        alerts = fdf[(fdf["Expected_Margin"] < margin_threshold) | (fdf["Discount"] > 0.15) | (fdf["Approval_Level"]=="Rejected")].copy()
        alerts = alerts.sort_values(["Expected_Margin","Discount"], ascending=[True, False])

        # conditional formatting using pandas Styler
        def highlight_alerts(row):
            styles = []
            # color Expected_Margin cell
            if pd.notnull(row['Expected_Margin']) and row['Expected_Margin'] < margin_threshold:
                styles.append('background-color: rgba(255,0,0,0.2)')
            else:
                styles.append('')
            # color Discount
            if row['Discount'] > 0.15:
                styles.append('background-color: rgba(255,165,0,0.25)')
            else:
                styles.append('')
            # Approval level
            if row['Approval_Level'] == 'Rejected':
                styles.append('background-color: rgba(128,128,128,0.2)')
            else:
                styles.append('')
            return styles

        if alerts.empty:
            st.write("No alerts in filtered set — good job!")
        else:
            display_cols = ["Lead_ID","Lead_Date","Customer","Route","Service_Type","Quoted_Price","Estimated_Cost","Expected_Margin","Discount","Approval_Level","Quote_Won"]
            styled = alerts[display_cols].style.format({"Quoted_Price":"${:,.2f}","Estimated_Cost":"${:,.2f}","Expected_Margin":"{:.1%}","Discount":"{:.1%}"})
            # apply per-row style: map to columns in order: Expected_Margin, Discount, Approval_Level -> we will apply with subset
            # pandas Styler row-wise apply returns list matching number of columns; to keep simple, style only specific columns
            def style_expected_margin(val):
                if pd.notnull(val) and val < margin_threshold:
                    return 'background-color: rgba(255,0,0,0.2)'
                return ''
            def style_discount(val):
                if val > 0.15:
                    return 'background-color: rgba(255,165,0,0.25)'
                return ''
            def style_approval(val):
                if val == 'Rejected':
                    return 'background-color: rgba(128,128,128,0.2)'
                return ''

            styled = styled.applymap(style_expected_margin, subset=['Expected_Margin'])
            styled = styled.applymap(style_discount, subset=['Discount'])
            styled = styled.applymap(style_approval, subset=['Approval_Level'])

            st.write("### Alerts (detailed)")
            st.dataframe(styled, use_container_width=True)

            # Grouped summary by customer & route
            st.write("### Alerts summary — group by Customer & Route")
            grouped_alerts = alerts.groupby(["Customer","Route"]).agg(alerts_count=("Lead_ID","count"), avg_expected_margin=("Expected_Margin","mean"), total_quoted=("Quoted_Price","sum")).reset_index().sort_values('alerts_count', ascending=False)
            st.dataframe(grouped_alerts.style.format({"avg_expected_margin":"{:.1%}", "total_quoted":"${:,.2f}"}), use_container_width=True)

            # Suggested actions
            st.markdown("### Suggested Actions")
            suggestions = []
            top_customers = alerts['Customer'].value_counts().head(3).to_dict()
            for c,r in top_customers.items():
                suggestions.append(f"- Review pricing & approval rules for **{c}** — {r} alerts flagged.")
            suggestions.append("- Investigate routes with repeated extra-cost impacts and update cost library.")
            suggestions.append("- Enforce manager approvals for quotes below threshold.")
            for s in suggestions:
                st.write(s)

            # Export: full filtered data and alerts-only
            @st.cache_data
            def to_csv_bytes(df_in):
                return df_in.to_csv(index=False).encode('utf-8')

            col_export_1, col_export_2 = st.columns(2)
            with col_export_1:
                st.download_button("Download filtered data (CSV)", to_csv_bytes(fdf), "l2o_filtered_data.csv", "text/csv")
            with col_export_2:
                st.download_button("Download alerts only (CSV)", to_csv_bytes(alerts), "l2o_alerts.csv", "text/csv")

st.markdown("---")
st.caption("Dashboard updated.")
