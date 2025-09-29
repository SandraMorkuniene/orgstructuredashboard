import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

st.set_page_config(page_title="L2O Profitability & Process Dashboard (Redesigned)", layout="wide")
st.title("Lead-to-Order (L2O) — Profitability & Process Dashboard")
st.caption("Prototype uses synthetic data.")

# -------------------- Synthetic Data Generator --------------------
@st.cache_data
def generate_data(n=5000, seed=42):
    np.random.seed(seed)

    customers = [f'Cust_{i}' for i in range(1, 21)]
    routes = [f'City{i}-City{j}' for i, j in zip(np.random.randint(1, 10, 15), np.random.randint(11, 20, 15))]
    services = ['FTL', 'PTL', 'Express']
    sectors = ['Automotive', 'Retail', 'Pharma', 'Electronics']
    approval_outcomes = ['Auto-Approved', 'Escalated', 'Rejected']
    delay_reasons = ['Weather', 'Customs', 'Documentation', 'Capacity', 'None']

    # Org structure
    reps = [f'Rep_{i}' for i in range(1, 13)]
    teams = ['Team_A', 'Team_B', 'Team_C']
    managers = ['Manager_1', 'Manager_2', 'Manager_3']
    rep_to_team = {rep: np.random.choice(teams) for rep in reps}
    team_to_manager = {team: managers[i] for i, team in enumerate(teams)}

    data = []
    for i in range(n):
        customer = np.random.choice(customers)
        route = np.random.choice(routes)
        service = np.random.choice(services)
        sector = np.random.choice(sectors)

        rep = np.random.choice(reps)
        team = rep_to_team[rep]
        manager = team_to_manager[team]

        est_cost = np.random.randint(500, 5000)
        discount = np.random.uniform(0, 0.2)
        expected_margin = np.random.uniform(0.1, 0.3)

        approval = np.random.choice(approval_outcomes, p=[0.6, 0.3, 0.1])
        approved = 1 if approval != 'Rejected' else 0

        actual_margin = expected_margin - np.random.uniform(-0.05, 0.1)
        actual_margin = max(0, actual_margin)

        delay = np.random.choice(delay_reasons, p=[0.1, 0.1, 0.1, 0.1, 0.6])

        data.append([
            i, customer, route, service, sector, rep, team, manager,
            est_cost, discount, expected_margin, actual_margin,
            approval, approved, delay,
            datetime.date.today() - datetime.timedelta(days=np.random.randint(0, 365))
        ])

    df = pd.DataFrame(data, columns=[
        'LeadID', 'Customer', 'Route', 'Service', 'Sector', 'Sales_Rep', 'Sales_Team', 'Manager',
        'Estimated_Cost', 'Discount', 'Expected_Margin', 'Actual_Margin',
        'Approval_Status', 'Approved', 'Delay_Reason', 'Date'
    ])
    return df

# -------------------- Dashboard --------------------
df = generate_data()

# Filters
with st.sidebar:
    st.header("Filters")
    selected_customers = st.multiselect("Customer", options=df['Customer'].unique())
    selected_routes = st.multiselect("Route", options=df['Route'].unique())
    selected_services = st.multiselect("Service", options=df['Service'].unique())
    selected_teams = st.multiselect("Sales Team", options=df['Sales_Team'].unique())
    date_range = st.date_input("Date range", [df['Date'].min(), df['Date'].max()])

fdf = df.copy()
if selected_customers:
    fdf = fdf[fdf['Customer'].isin(selected_customers)]
if selected_routes:
    fdf = fdf[fdf['Route'].isin(selected_routes)]
if selected_services:
    fdf = fdf[fdf['Service'].isin(selected_services)]
if selected_teams:
    fdf = fdf[fdf['Sales_Team'].isin(selected_teams)]
if date_range:
    fdf = fdf[(fdf['Date'] >= date_range[0]) & (fdf['Date'] <= date_range[1])]

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["Executive Summary", "Process Accountability", "Alerts"])

# --- Executive Summary ---
with tab1:
    st.subheader("KPIs by Team")
    if not fdf.empty:
        team_kpis = fdf.groupby('Sales_Team').agg({
            'LeadID': 'count',
            'Approved': 'mean',
            'Expected_Margin': 'mean',
            'Actual_Margin': 'mean',
            'Discount': 'mean'
        }).reset_index()
        team_kpis.rename(columns={
            'LeadID': 'Quotes', 'Approved': 'Approval Rate',
            'Expected_Margin': 'Avg Expected Margin', 'Actual_Margin': 'Avg Actual Margin', 'Discount': 'Avg Discount'
        }, inplace=True)
        st.dataframe(team_kpis)
    else:
        st.warning("No data after filtering.")

# --- Process Accountability ---
with tab2:
    st.subheader("Approval Patterns by Team/Rep")
    if not fdf.empty:
        approval_dist = fdf.groupby(['Sales_Team', 'Approval_Status']).size().reset_index(name='Count')
        fig = px.bar(approval_dist, x='Sales_Team', y='Count', color='Approval_Status', barmode='group', title="Approvals by Team")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Conversion Efficiency by Team")
        funnel = fdf.groupby('Sales_Team').agg({
            'LeadID': 'count',
            'Approved': 'sum'
        }).reset_index()
        funnel['Conversion %'] = funnel['Approved'] / funnel['LeadID'] * 100
        st.dataframe(funnel)

        st.subheader("Margin Gap by Team (Expected vs Actual)")
        margin_gap = fdf.groupby('Sales_Team').agg({
            'Expected_Margin': 'mean', 'Actual_Margin': 'mean'
        }).reset_index()
        margin_gap['Gap'] = margin_gap['Expected_Margin'] - margin_gap['Actual_Margin']
        fig_gap = px.bar(margin_gap, x='Sales_Team', y='Gap', color='Sales_Team', title="Margin Gap (pts)")
        st.plotly_chart(fig_gap, use_container_width=True)

        st.subheader("Discount Discipline by Rep")
        disc = fdf.groupby('Sales_Rep').agg({
            'Discount': 'mean', 'Actual_Margin': 'mean'
        }).reset_index()
        fig_disc = px.scatter(disc, x='Discount', y='Actual_Margin', text='Sales_Rep', title="Rep Discount vs Margin")
        fig_disc.update_traces(textposition='top center')
        st.plotly_chart(fig_disc, use_container_width=True)
    else:
        st.warning("No data after filtering.")

# --- Alerts ---
with tab3:
    st.subheader("Alerts by Team")
    if not fdf.empty:
        alerts = fdf[(fdf['Actual_Margin'] < 0.1) | (fdf['Discount'] > 0.15) | (fdf['Approval_Status'] == 'Rejected')]
        if not alerts.empty:
            st.dataframe(alerts[['Customer', 'Sales_Team', 'Sales_Rep', 'Route', 'Service', 'Expected_Margin', 'Actual_Margin', 'Discount', 'Approval_Status']])
        else:
            st.success("No alerts found.")
    else:
        st.warning("No data after filtering.")
