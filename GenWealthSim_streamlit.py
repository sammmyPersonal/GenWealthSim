import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Union
import json
import plotly.graph_objects as go


# ------------------ Backend Logic ------------------
@dataclass
class InvestmentBlock:
    name: str
    initial: float
    annual_growth: Union[float, Dict[int, float]]
    annual_spend_rate: Union[float, Dict[int, float]]
    contributions: Dict[int, float] = field(default_factory=dict)
    big_spend_events: Dict[int, float] = field(default_factory=dict)

def get_rate(rate: Union[float, Dict[int, float]], year: int) -> float:
    if isinstance(rate, dict):
        valid_years = [y for y in rate.keys() if y <= year]
        if not valid_years:
            return list(rate.values())[0]
        last_year = max(valid_years)
        return rate[last_year]
    return rate

def simulate_blocks(blocks: List[InvestmentBlock], years: int = 100, start_year: int = 0,
                    dt: float = 1.0, stochastic: bool = False, sigma: float = 0.05):
    years_range = list(range(start_year, start_year + years + 1))
    state = {b.name: b.initial for b in blocks}
    records = []

    for y in years_range:
        record = {'Year': y}
        for b in blocks:
            record[b.name] = state[b.name]
        record['Total'] = sum(state.values())
        records.append(record)

        if y == years_range[-1]:
            break

        for b in blocks:
            state[b.name] += b.contributions.get(y, 0.0)
            growth_rate = get_rate(b.annual_growth, y)
            if stochastic:
                growth_rate += np.random.normal(0, sigma)
            state[b.name] *= (1.0 + growth_rate) ** dt
            spend_rate = get_rate(b.annual_spend_rate, y)
            state[b.name] -= state[b.name] * spend_rate * dt
            state[b.name] -= b.big_spend_events.get(y, 0.0)
            if state[b.name] < 0:
                state[b.name] = 0.0

    return pd.DataFrame.from_records(records).set_index('Year')

def per_capita_wealth(df_total, family_members_dict):
    per_capita = pd.Series(index=df_total.index, dtype=float)
    for year in df_total.index:
        members = family_members_dict.get(year, list(family_members_dict.values())[-1])
        per_capita.loc[year] = df_total.loc[year, 'Total'] / members
    return per_capita

# ------------------ Streamlit UI ------------------
st.set_page_config(layout="wide")
st.title("Generational Wealth Simulator Dashboard")

# ------------------ Page Theme / CSS ------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp { 
        background-color: #f0f2f6; 
        color: #000000; 
    }

    /* Sidebar background */
    .css-1d391kg { 
        background-color: #e8eaf6; 
        color: #000000; 
    }

    /* Input boxes and text areas */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    .stTextArea>div>div>textarea {
        background-color: #ffffff; 
        color: #000000;
    }

    /* Textbox labels on main page → black */
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label {
        color: #111111 !important;
    }

    /* Textbox labels in sidebar → white */
    .stSidebar .stTextInput label,
    .stSidebar .stNumberInput label,
    .stSidebar .stTextArea label {
        color: #ffffff !important;
    }

    /* Expander headers */
    .st-expander header { 
        background-color: #d1d9ff; 
        color: #111111 !important; 
        font-weight: bold; 
    }

    /* Streamlit buttons */
    .stButton>button { 
        background-color: #4a90e2; 
        color: #000000;   /* Button text color */
    }

    /* Main page sliders → black labels */
    .stSlider label, 
    .stSlider span, 
    .stSlider div[data-baseweb="slider"] {
        color: #111111 !important;
    }

    /* Sidebar sliders → white labels */
    .stSidebar .stSlider label, 
    .stSidebar .stSlider span, 
    .stSidebar .stSlider div[data-baseweb="slider"] {
        color: #ffffff !important;
    }
</style>


    """, unsafe_allow_html=True
)

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Simulation Parameters")
years = st.sidebar.slider("Years to Simulate", 10, 200, 100)
sigma = st.sidebar.slider("Volatility (sigma)", 0.0, 0.2, 0.05, 0.01)
n_sim = st.sidebar.slider("Monte Carlo Simulations", 10, 2000, 1000, 10)

st.sidebar.header("Family Members")
num_members_start = st.sidebar.slider("Members Start Period", 1, 10, 4)
num_members_mid = st.sidebar.slider("Members Mid Period", 1, 10, 5)
num_members_end = st.sidebar.slider("Members End Period", 1, 10, 3)

family_members = {y: num_members_start for y in range(0, int(years/3))}
family_members.update({y: num_members_mid for y in range(int(years/3), int(2*years/3))})
family_members.update({y: num_members_end for y in range(int(2*years/3), years+1)})

st.sidebar.header("Investment Blocks")
if "blocks" not in st.session_state:
    st.session_state.blocks = [
        InvestmentBlock("Living_Expenses", 100_000_000.0, {0:0.03, 30:0.02}, {0:0.02,30:0.06}, {y:50000 for y in range(0,31)}, {20:200000}),
        InvestmentBlock("Emergency_Fund", 1_000_000.0, 0.015, 0.0, {}, {45:50000}),
        InvestmentBlock("Legacy_Fund", 4_000_000.0, {0:0.06,60:0.04}, {0:0,60:0.06}, {y:20000 for y in range(15,41,5)}, {80:500000}),
    ]

add_block = st.sidebar.button("Add Block")
remove_block = st.sidebar.button("Remove Last Block")
if add_block:
    st.session_state.blocks.append(InvestmentBlock(f"New_Block_{len(st.session_state.blocks)+1}", 1_000_000.0, 0.03, 0.02))
if remove_block and len(st.session_state.blocks) > 1:
    st.session_state.blocks.pop()

# ------------------ Investment Block Config ------------------
for i, block in enumerate(st.session_state.blocks):
    with st.expander(f"Block {i+1}: {block.name}", expanded=False):
        block.name = st.text_input(f"Block Name {i+1}", block.name)
        block.initial = st.number_input(f"Initial Value ({block.name})", min_value=0.0, value=float(block.initial), step=10000.0)

        # ------------------ Sliders for growth/spend ------------------

        st.subheader(f"{block.name} Parameters")
        
        # Row 1: Growth sliders (3 periods)
        st.markdown("**Growth Rates**")
        growth_col1, growth_col2, growth_col3 = st.columns(3)
        with growth_col1:
            growth_0_33 = st.slider(
                f"{block.name} Growth (0–33 yrs)",
                0.0, 0.2,
                float(get_rate(block.annual_growth, 0)),
                0.01
            )
        with growth_col2:
            growth_33_66 = st.slider(
                f"{block.name} Growth (33–66 yrs)",
                0.0, 0.2,
                float(get_rate(block.annual_growth, int(years/3))),
                0.01
            )
        with growth_col3:
            growth_66_100 = st.slider(
                f"{block.name} Growth (66–100 yrs)",
                0.0, 0.2,
                float(get_rate(block.annual_growth, int(2*years/3))),
                0.01
            )
        
        # Row 2: Spend sliders (3 periods)
        st.markdown("**Spending Rates**")
        spend_col1, spend_col2, spend_col3 = st.columns(3)
        with spend_col1:
            spend_0_33 = st.slider(
                f"{block.name} Spend (0–33 yrs)",
                0.0, 0.2,
                float(get_rate(block.annual_spend_rate, 0)),
                0.01
            )
        with spend_col2:
            spend_33_66 = st.slider(
                f"{block.name} Spend (33–66 yrs)",
                0.0, 0.2,
                float(get_rate(block.annual_spend_rate, int(years/3))),
                0.01
            )
        with spend_col3:
            spend_66_100 = st.slider(
                f"{block.name} Spend (66–100 yrs)",
                0.0, 0.2,
                float(get_rate(block.annual_spend_rate, int(2*years/3))),
                0.01
            )

        # Map growth sliders to their respective period start years
        block.annual_growth = {
            0: growth_0_33,
            int(years/3): growth_33_66,
            int(2*years/3): growth_66_100
        }
        
        # Map spend sliders to their respective period start years
        block.annual_spend_rate = {
            0: spend_0_33,
            int(years/3): spend_33_66,
            int(2*years/3): spend_66_100
        }


        # Optional JSON for fine control
        # st.markdown("**Optional JSON input for growth/spend rates (year:value)**")
        # growth_json = st.text_area(f"{block.name} Growth JSON", json.dumps(block.annual_growth))
        # spend_json = st.text_area(f"{block.name} Spend JSON", json.dumps(block.annual_spend_rate))
        # try:
        #     growth_dict = {int(k): float(v) for k,v in json.loads(growth_json).items()}
        #     spend_dict = {int(k): float(v) for k,v in json.loads(spend_json).items()}
        #     block.annual_growth = growth_dict
        #     block.annual_spend_rate = spend_dict
        # except:
        #     st.warning("Invalid JSON input for growth/spend rates")

        # Contributions and big spends
        # contrib_input = st.text_area(f"{block.name} Contributions (JSON year:value)", json.dumps(block.contributions), height=80)
        # try:
        #     block.contributions = {int(k): float(v) for k,v in json.loads(contrib_input).items()}
        # except:
        #     st.warning("Invalid JSON for contributions")
        # big_spend_input = st.text_area(f"{block.name} Big Spend Events (JSON year:value)", json.dumps(block.big_spend_events), height=80)
        # try:
        #     block.big_spend_events = {int(k): float(v) for k,v in json.loads(big_spend_input).items()}
        # except:
        #     st.warning("Invalid JSON for big spend events")

        st.subheader(f"{block.name} Contributions")
        
        # Initialise state
        if "contrib_rules" not in st.session_state:
            st.session_state.contrib_rules = []
        
        # Button to add a new contribution rule
        if st.button(f"➕ Add Contributions"):
            st.session_state.contrib_rules.append({"amount": 0.0, "start": 0, "years": 1})
        
        # Render contribution rules
        remove_indices = []
        for i, rule in enumerate(st.session_state.contrib_rules):
            st.markdown(f"**Rule {i+1}**")
            cols = st.columns(3)
            with cols[0]:
                rule["amount"] = st.number_input(f"Amount (Rule {i+1})", min_value=0.0, value=rule["amount"], step=1000.0, key=f"amount_{i}")
            with cols[1]:
                rule["start"] = st.number_input(f"Start Year (Rule {i+1})", min_value=0, max_value=100, value=rule["start"], step=1, key=f"start_{i}")
            with cols[2]:
                rule["years"] = st.number_input(f"Years (Rule {i+1})", min_value=1, max_value=100, value=rule["years"], step=1, key=f"years_{i}")
        
            # Remove button
            if st.button(f"❌ Remove{i+1}", key=f"remove_{i}"):
                remove_indices.append(i)
        
        # Remove selected rules
        for idx in sorted(remove_indices, reverse=True):
            st.session_state.contrib_rules.pop(idx)
        
        # Expand contribution rules into a year:value dict
        contributions = {}
        for rule in st.session_state.contrib_rules:
            for yr in range(rule["start"], rule["start"] + rule["years"]):
                contributions[yr] = contributions.get(yr, 0) + rule["amount"]
        
        block.contributions = contributions

        # Spend events block #
        st.subheader(f"{block.name} Big Spend Events")
        
        # Initialise state for each block
        if f"spend_rules_{block.name}" not in st.session_state:
            st.session_state[f"spend_rules_{block.name}"] = []
        
        # Button to add a new spend rule
        if st.button(f"➕ Add Big Spend", key=f"add_spend_{block.name}"):
            st.session_state[f"spend_rules_{block.name}"].append({"amount": 0.0, "year": 0})
        
        # Render spend rules
        remove_spend_indices = []
        for i, rule in enumerate(st.session_state[f"spend_rules_{block.name}"]):
            st.markdown(f"**Spend {i+1}**")
            cols = st.columns(2)
            with cols[0]:
                rule["amount"] = st.number_input(
                    f"Amount (Spend {i+1})",
                    min_value=0.0,
                    value=rule["amount"],
                    step=1000.0,
                    key=f"spend_amount_{block.name}_{i}"
                )
            with cols[1]:
                rule["year"] = st.number_input(
                    f"Year (Spend {i+1})",
                    min_value=0,
                    max_value=100,
                    value=rule["year"],
                    step=1,
                    key=f"spend_year_{block.name}_{i}"
                )
        
            # Remove button
            if st.button(f"❌ Remove Spend {i+1}", key=f"remove_spend_{block.name}_{i}"):
                remove_spend_indices.append(i)
        
        # Remove selected spend rules
        for idx in sorted(remove_spend_indices, reverse=True):
            st.session_state[f"spend_rules_{block.name}"].pop(idx)
        
        # Expand spend rules into year:value dict
        big_spend_events = {}
        for rule in st.session_state[f"spend_rules_{block.name}"]:
            big_spend_events[rule["year"]] = big_spend_events.get(rule["year"], 0) + rule["amount"]
        
        block.big_spend_events = big_spend_events



       


# ------------------ Run Simulation ------------------
df = simulate_blocks(st.session_state.blocks, years=years)

# Monte Carlo for Total Wealth
total_wealth_matrix = np.array([simulate_blocks(st.session_state.blocks, years=years, stochastic=True, sigma=sigma)['Total'].values for _ in range(n_sim)])
mean_total = total_wealth_matrix.mean(axis=0)
lower = np.percentile(total_wealth_matrix, 5, axis=0)
upper = np.percentile(total_wealth_matrix, 95, axis=0)

# ------------------ Generational Wealth Plot ------------------
fig = go.Figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, b in enumerate(st.session_state.blocks):
    fig.add_trace(go.Scatter(x=df.index, y=df[b.name], mode='lines', name=b.name,
                             line=dict(color=colors[i % len(colors)], width=3)))

fig.add_trace(go.Scatter(x=df.index, y=mean_total, mode='lines', name='Total Wealth (Mean)',
                         line=dict(color='#000000', width=4)))
fig.add_trace(go.Scatter(x=list(df.index)+list(df.index[::-1]),
                         y=list(upper)+list(lower[::-1]),
                         fill='toself', fillcolor='rgba(100,100,100,0.2)',
                         line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='5–95% CI'))

fig.update_layout(
    title=dict(
        text="Generational Wealth Over Time",
        font=dict(color="#111111", size=20)
    ),
    xaxis=dict(
        title=dict(text="Year", font=dict(color="#111111", size=16)),
        tickfont=dict(color="#111111")
    ),
    yaxis=dict(
        title=dict(text="Value ($)", font=dict(color="#111111", size=16)),
        tickfont=dict(color="#111111")
    ),
    plot_bgcolor='#f0f2f6',
    paper_bgcolor='#f0f2f6',
    font=dict(color="#111111"),
    hovermode="x unified",
    legend=dict(
        orientation="h",       # horizontal legend
        yanchor="top",
        y=-0.2,                # push below the x-axis
        xanchor="left",        # left-align
        x=0,                   # start from left
        bgcolor='rgba(255,255,255,0.5)',
        font=dict(color="#111111")
    )
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ Per-Capita Wealth Plot ------------------
pc_wealth = per_capita_wealth(df, family_members)
pc_fig = go.Figure()

pc_fig.add_trace(go.Scatter(
    x=pc_wealth.index,
    y=pc_wealth.values,
    mode='lines+markers',
    line=dict(color='#1f77b4', width=3),
    name="Per-Capita Wealth"
))

pc_fig.update_layout(
    title=dict(
        text="Per-Capita Wealth Over Time",
        font=dict(color="#111111", size=20)
    ),
    xaxis=dict(
        title=dict(text="Year", font=dict(color="#111111", size=16)),
        tickfont=dict(color="#111111")
    ),
    yaxis=dict(
        title=dict(text="Value per Person ($)", font=dict(color="#111111", size=16)),
        tickfont=dict(color="#111111")
    ),
    plot_bgcolor='#f0f2f6',
    paper_bgcolor='#f0f2f6',
    font=dict(color="#111111"),
    hovermode="x unified",
    legend=dict(bgcolor='rgba(255,255,255,0.5)', font=dict(color="#111111"))
)

st.plotly_chart(pc_fig, use_container_width=True)
