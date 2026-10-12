
from pathlib import Path
import html
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Restaurant Revenue Executive Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# FILE LOADING
# =========================================================
def find_file(filename: str) -> Path:
    candidates = [BASE_DIR / filename, Path.cwd() / filename]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@st.cache_resource
def load_model():
    return joblib.load(find_file("model.pkl"))


@st.cache_data
def load_data():
    return pd.read_csv(find_file("data.csv"))


try:
    model = load_model()
    data = load_data()
except FileNotFoundError:
    st.error("Place **model.pkl** and **data.csv** in the same folder as `app.py`.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to load the model/data: {exc}")
    st.stop()


# =========================================================
# CONSTANTS
# =========================================================
NUMERICAL_FEATURES = [
    "Average_Daily_Customers",
    "Average_Bill_Value",
    "Delivery_Order_Percentage",
    "Seating_Capacity",
    "Promotion_Spend",
    "Customer_Rating",
]
CATEGORICAL_FEATURES = ["Cuisine_Type", "Location_Type"]
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TEST_R2 = 0.9489704376207022
TEST_MAE = 135596.48376821354
TEST_RMSE = 172629.31243172596


# =========================================================
# HELPERS
# =========================================================
def rupee_full(value):
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.0f}"


def rupee_short(value):
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 10_000_000:
        return f"{sign}₹{v / 10_000_000:.2f} Cr"
    if v >= 100_000:
        return f"{sign}₹{v / 100_000:.2f} L"
    if v >= 1_000:
        return f"{sign}₹{v / 1_000:.1f} K"
    return f"{sign}₹{v:,.0f}"


def make_input_df(values):
    return pd.DataFrame([{
        "Average_Daily_Customers": float(values["Average_Daily_Customers"]),
        "Average_Bill_Value": float(values["Average_Bill_Value"]),
        "Delivery_Order_Percentage": float(values["Delivery_Order_Percentage"]),
        "Seating_Capacity": float(values["Seating_Capacity"]),
        "Promotion_Spend": float(values["Promotion_Spend"]),
        "Customer_Rating": float(values["Customer_Rating"]),
        "Cuisine_Type": str(values["Cuisine_Type"]),
        "Location_Type": str(values["Location_Type"]),
    }], columns=MODEL_FEATURES)


def predict(values):
    return float(model.predict(make_input_df(values))[0])


def revenue_position(prediction):
    revenue = data["Monthly_Revenue"]
    q25 = float(revenue.quantile(0.25))
    median = float(revenue.median())
    q75 = float(revenue.quantile(0.75))
    percentile = float((revenue <= prediction).mean() * 100)

    if prediction < q25:
        band = "Below Benchmark"
    elif prediction < median:
        band = "Developing"
    elif prediction < q75:
        band = "Strong"
    else:
        band = "Top Quartile"

    return {
        "q25": q25,
        "median": median,
        "q75": q75,
        "percentile": percentile,
        "band": band,
    }


def scenario_analysis(values):
    baseline = predict(values)
    mins = data[NUMERICAL_FEATURES].min()
    maxs = data[NUMERICAL_FEATURES].max()

    specs = [
        ("Increase daily customers by 10%", "Average_Daily_Customers", lambda x: x * 1.10, "Demand"),
        ("Increase average bill value by 5%", "Average_Bill_Value", lambda x: x * 1.05, "Pricing"),
        ("Increase delivery share by 10 pp", "Delivery_Order_Percentage", lambda x: x + 10, "Channel"),
        ("Increase promotion spend by 10%", "Promotion_Spend", lambda x: x * 1.10, "Marketing"),
        ("Improve customer rating by 0.2", "Customer_Rating", lambda x: x + 0.2, "Experience"),
        ("Add 10 seats", "Seating_Capacity", lambda x: x + 10, "Capacity"),
    ]

    rows = []
    for label, feature, fn, category in specs:
        changed = dict(values)
        original = float(changed[feature])
        new_value = fn(original)
        new_value = max(float(mins[feature]), min(float(maxs[feature]), new_value))

        if feature in {"Average_Daily_Customers", "Promotion_Spend", "Seating_Capacity"}:
            new_value = int(round(new_value))
        else:
            new_value = round(new_value, 2)

        changed[feature] = new_value
        pred = predict(changed)
        uplift = pred - baseline

        rows.append({
            "Scenario": label,
            "Category": category,
            "Predicted Revenue": pred,
            "Predicted Uplift": uplift,
            "Uplift %": (uplift / baseline * 100) if baseline else 0.0,
        })

    return pd.DataFrame(rows).sort_values(
        "Predicted Uplift", ascending=False
    ).reset_index(drop=True)


def manager_interpretation(prediction, pos):
    p = pos["percentile"]
    if pos["band"] == "Below Benchmark":
        return (
            f"The outlet is predicted to generate **{rupee_full(prediction)} per month**, "
            f"placing it around the **{p:.0f}th percentile**. This is below the lower-quartile "
            f"benchmark of **{rupee_full(pos['q25'])}**. Management should first identify the "
            "largest revenue gap and test one high-impact lever before increasing spending broadly."
        )
    if pos["band"] == "Developing":
        return (
            f"The outlet is predicted to generate **{rupee_full(prediction)} per month**, "
            f"around the **{p:.0f}th percentile**. It remains below the dataset median of "
            f"**{rupee_full(pos['median'])}**, so management should focus on one or two measurable "
            "growth initiatives and track whether actual revenue moves toward the median benchmark."
        )
    if pos["band"] == "Strong":
        return (
            f"The outlet is predicted to generate **{rupee_full(prediction)} per month**, "
            f"around the **{p:.0f}th percentile**, which is above the dataset median. Management can "
            "shift from recovery to selective scaling while protecting service quality and unit economics."
        )
    return (
        f"The outlet is predicted to generate **{rupee_full(prediction)} per month**, "
        f"around the **{p:.0f}th percentile**, at or above the upper-quartile benchmark of "
        f"**{rupee_full(pos['q75'])}**. Management should protect the current operating model and "
        "use controlled experiments for incremental growth."
    )


def priority_recommendations(values, prediction, pos, scenarios):
    recs = []

    if pos["band"] == "Below Benchmark":
        recs.append((
            "Recover core revenue first",
            "Focus on the highest model-implied lever and set a measurable 30-day improvement target."
        ))
    elif pos["band"] == "Developing":
        recs.append((
            "Close the gap to the median outlet",
            f"The current prediction is {rupee_full(pos['median'] - prediction)} below the dataset median."
        ))
    elif pos["band"] == "Strong":
        recs.append((
            "Scale selectively",
            "Prioritise growth initiatives that preserve customer experience and operating quality."
        ))
    else:
        recs.append((
            "Protect strong performance",
            "Optimise efficiency and test incremental changes rather than chasing volume blindly."
        ))

    for _, row in scenarios.head(2).iterrows():
        recs.append((
            row["Scenario"],
            f"Model-implied monthly uplift: {rupee_full(row['Predicted Uplift'])} ({row['Uplift %']:.1f}%)."
        ))

    rating_median = float(data["Customer_Rating"].median())
    if values["Customer_Rating"] < rating_median:
        recs.append((
            "Improve customer experience",
            f"Rating is below the dataset median of {rating_median:.1f}. Review service speed, consistency and complaints."
        ))
    else:
        delivery_median = float(data["Delivery_Order_Percentage"].median())
        if values["Delivery_Order_Percentage"] < delivery_median:
            recs.append((
                "Review delivery-channel opportunity",
                f"Delivery share is below the dataset median of {delivery_median:.1f}%."
            ))

    return recs[:4]


# =========================================================
# SIDEBAR INPUTS
# =========================================================
with st.sidebar:
    st.markdown("## Outlet Inputs")
    st.caption("Enter the current or planned outlet profile.")

    cuisines = sorted(data["Cuisine_Type"].dropna().unique().tolist())
    locations = sorted(data["Location_Type"].dropna().unique().tolist())

    values = {
        "Average_Daily_Customers": st.number_input(
            "Average Daily Customers",
            min_value=1,
            max_value=2000,
            value=int(round(data["Average_Daily_Customers"].median())),
            step=5,
        ),
        "Average_Bill_Value": st.number_input(
            "Average Bill Value (₹)",
            min_value=50.0,
            max_value=5000.0,
            value=float(round(data["Average_Bill_Value"].median(), 2)),
            step=10.0,
        ),
        "Delivery_Order_Percentage": st.slider(
            "Delivery Orders (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(data["Delivery_Order_Percentage"].median()),
            step=1.0,
        ),
        "Seating_Capacity": st.number_input(
            "Seating Capacity",
            min_value=1,
            max_value=1000,
            value=int(round(data["Seating_Capacity"].median())),
            step=5,
        ),
        "Promotion_Spend": st.number_input(
            "Monthly Promotion Spend (₹)",
            min_value=0,
            max_value=2_000_000,
            value=int(round(data["Promotion_Spend"].median())),
            step=5000,
        ),
        "Customer_Rating": st.slider(
            "Customer Rating",
            min_value=1.0,
            max_value=5.0,
            value=float(data["Customer_Rating"].median()),
            step=0.1,
        ),
        "Cuisine_Type": st.selectbox("Cuisine Type", cuisines),
        "Location_Type": st.selectbox("Location Type", locations),
    }

    st.divider()
    st.caption(
        "`Employee_Count` is excluded because it was removed after the VIF/multicollinearity correction."
    )


# =========================================================
# THEME-ADAPTIVE CSS
# Uses inherited Streamlit text/background colors instead of
# forcing a dark or light palette.
# =========================================================
st.markdown(
    """
<style>
.block-container {
    max-width: 1500px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}

/* Neutral adaptive surfaces: currentColor follows Streamlit's active theme */
.hero {
    padding: clamp(1.25rem, 2.4vw, 2rem);
    border-radius: 22px;
    background: linear-gradient(120deg, #0f3d78 0%, #2563eb 100%);
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, .12);
}
.hero-title {
    font-size: clamp(1.7rem, 3vw, 2.55rem);
    line-height: 1.08;
    font-weight: 850;
    letter-spacing: -0.035em;
}
.hero-subtitle {
    margin-top: .6rem;
    color: rgba(255,255,255,.88);
    font-size: clamp(.92rem, 1.4vw, 1.05rem);
    max-width: 930px;
    line-height: 1.55;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0,1fr));
    gap: .8rem;
    margin: .9rem 0 1.15rem 0;
}
.kpi-card,
.info-card,
.rec-card,
.signal,
.executive-box {
    background: color-mix(in srgb, currentColor 4%, transparent);
    border: 1px solid color-mix(in srgb, currentColor 15%, transparent);
    border-radius: 16px;
}
.kpi-card {
    padding: 1rem;
    min-width: 0;
}
.kpi-label {
    opacity: .62;
    font-size: .82rem;
    font-weight: 650;
    line-height: 1.3;
    min-height: 2.05em;
}
.kpi-value {
    font-size: clamp(1.3rem, 2.1vw, 2rem);
    line-height: 1.12;
    font-weight: 850;
    letter-spacing: -0.025em;
    margin-top: .5rem;
    overflow-wrap: anywhere;
}
.kpi-full {
    opacity: .58;
    font-size: .76rem;
    margin-top: .3rem;
    overflow-wrap: anywhere;
}

.badge {
    display: inline-block;
    margin-top: .45rem;
    padding: .2rem .5rem;
    border-radius: 999px;
    font-size: .74rem;
    font-weight: 750;
}
.badge-blue {
    background: rgba(37,99,235,.12);
    color: #2563eb;
}
.badge-green {
    background: rgba(22,163,74,.12);
    color: #16a34a;
}
.badge-red {
    background: rgba(220,38,38,.12);
    color: #dc2626;
}

.section-heading {
    font-size: clamp(1.05rem, 1.55vw, 1.2rem);
    font-weight: 800;
    margin: .45rem 0 .65rem 0;
}

.manager-box {
    padding: 1.05rem 1.1rem;
    border-radius: 16px;
    border: 1px solid color-mix(in srgb, currentColor 14%, transparent);
    border-left: 5px solid #2563eb;
    background: color-mix(in srgb, #2563eb 7%, transparent);
    line-height: 1.6;
    margin-bottom: .8rem;
}

.signal-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: .75rem;
}
.signal {
    padding: .9rem;
}
.signal-label {
    opacity: .62;
    font-size: .76rem;
    font-weight: 650;
}
.signal-value {
    font-size: clamp(1.05rem, 1.7vw, 1.4rem);
    font-weight: 800;
    margin-top: .3rem;
    overflow-wrap: anywhere;
}
.signal-sub {
    opacity: .58;
    font-size: .74rem;
    margin-top: .18rem;
}

.rec-card {
    padding: .95rem 1rem;
    margin-bottom: .65rem;
}
.rec-title {
    font-weight: 800;
    margin-bottom: .32rem;
}
.rec-body {
    opacity: .72;
    line-height: 1.5;
    font-size: .91rem;
}

.executive-box {
    padding: 1.05rem 1.1rem;
    line-height: 1.62;
    background: color-mix(in srgb, #2563eb 5%, transparent);
}
.footer-note {
    margin-top: .8rem;
    padding: .8rem .9rem;
    border-radius: 13px;
    opacity: .72;
    font-size: .81rem;
    line-height: 1.5;
    border: 1px solid color-mix(in srgb, currentColor 13%, transparent);
}

/* Let native Streamlit components retain proper theme contrast */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
}

/* Responsive layout */
@media (max-width: 1200px) {
    .kpi-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
}
@media (max-width: 780px) {
    .block-container {
        padding-left: .75rem;
        padding-right: .75rem;
    }
    .kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .signal-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width: 480px) {
    .kpi-grid, .signal-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
<div class="hero">
    <div class="hero-title">Restaurant Revenue Executive Cockpit</div>
    <div class="hero-subtitle">
        Predict monthly revenue, benchmark outlet performance, test growth scenarios
        and translate model output into managerial actions.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PREDICTION
# =========================================================
prediction = predict(values)
annualised = prediction * 12
pos = revenue_position(prediction)
delta_vs_median = prediction - pos["median"]
delta_pct = delta_vs_median / pos["median"] * 100 if pos["median"] else 0

badge_cls = "badge-green" if delta_vs_median >= 0 else "badge-red"
arrow = "↑" if delta_vs_median >= 0 else "↓"

st.markdown(
    f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Predicted Monthly Revenue</div>
        <div class="kpi-value">{rupee_short(prediction)}</div>
        <div class="kpi-full">{rupee_full(prediction)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Annualised Revenue</div>
        <div class="kpi-value">{rupee_short(annualised)}</div>
        <div class="kpi-full">{rupee_full(annualised)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Vs Dataset Median</div>
        <div class="kpi-value">{rupee_short(delta_vs_median)}</div>
        <div class="kpi-full">{rupee_full(delta_vs_median)}</div>
        <span class="badge {badge_cls}">{arrow} {abs(delta_pct):.1f}%</span>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Revenue Position</div>
        <div class="kpi-value">{html.escape(pos["band"])}</div>
        <div class="kpi-full">{pos["percentile"]:.0f}th percentile</div>
        <span class="badge badge-blue">Benchmark position</span>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Reference Model R²</div>
        <div class="kpi-value">{TEST_R2:.3f}</div>
        <div class="kpi-full">≈ {TEST_R2*100:.1f}% test variance explained</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# MANAGER INTERPRETATION
# =========================================================
st.markdown('<div class="section-heading">What the prediction means for a manager</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="manager-box">{manager_interpretation(prediction, pos)}</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive View", "Growth Scenarios", "Business Health", "Model Context"]
)

with tab1:
    left, right = st.columns([1.02, 1])

    with left:
        st.markdown('<div class="section-heading">Revenue Benchmark</div>', unsafe_allow_html=True)

        benchmark = pd.DataFrame({
            "Benchmark": [
                "Lower Quartile (25th percentile)",
                "Median Outlet",
                "Upper Quartile (75th percentile)",
                "Your Prediction",
            ],
            "Monthly Revenue": [
                pos["q25"], pos["median"], pos["q75"], prediction
            ],
        })
        benchmark["Monthly Revenue"] = benchmark["Monthly Revenue"].map(rupee_full)

        st.dataframe(benchmark, use_container_width=True, hide_index=True)
        st.progress(max(0.0, min(1.0, pos["percentile"] / 100)))
        st.caption(
            f"Predicted outlet position: approximately **{pos['percentile']:.0f}th percentile** "
            "of the training dataset."
        )

    with right:
        st.markdown('<div class="section-heading">Operating Snapshot</div>', unsafe_allow_html=True)

        monthly_customers = values["Average_Daily_Customers"] * 30
        customers_per_seat = values["Average_Daily_Customers"] / values["Seating_Capacity"]
        promo_ratio = values["Promotion_Spend"] / prediction * 100 if prediction else 0

        st.markdown(
            f"""
<div class="signal-grid">
    <div class="signal">
        <div class="signal-label">Estimated Monthly Customers</div>
        <div class="signal-value">{monthly_customers:,.0f}</div>
        <div class="signal-sub">Daily customers × 30</div>
    </div>
    <div class="signal">
        <div class="signal-label">Customers per Seat / Day</div>
        <div class="signal-value">{customers_per_seat:.2f}</div>
        <div class="signal-sub">Capacity-utilisation signal</div>
    </div>
    <div class="signal">
        <div class="signal-label">Average Bill Value</div>
        <div class="signal-value">{rupee_short(values["Average_Bill_Value"])}</div>
        <div class="signal-sub">{rupee_full(values["Average_Bill_Value"])}</div>
    </div>
    <div class="signal">
        <div class="signal-label">Delivery Mix</div>
        <div class="signal-value">{values["Delivery_Order_Percentage"]:.1f}%</div>
        <div class="signal-sub">Share of delivery orders</div>
    </div>
    <div class="signal">
        <div class="signal-label">Promotion / Revenue</div>
        <div class="signal-value">{promo_ratio:.1f}%</div>
        <div class="signal-sub">Marketing intensity</div>
    </div>
    <div class="signal">
        <div class="signal-label">Customer Rating</div>
        <div class="signal-value">{values["Customer_Rating"]:.1f} / 5</div>
        <div class="signal-sub">Current input rating</div>
    </div>
    <div class="signal">
        <div class="signal-label">Seating Capacity</div>
        <div class="signal-value">{values["Seating_Capacity"]:,.0f}</div>
        <div class="signal-sub">Current outlet capacity</div>
    </div>
    <div class="signal">
        <div class="signal-label">Promotion Spend</div>
        <div class="signal-value">{rupee_short(values["Promotion_Spend"])}</div>
        <div class="signal-sub">{rupee_full(values["Promotion_Spend"])}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    scenarios = scenario_analysis(values)
    recs = priority_recommendations(values, prediction, pos, scenarios)

    st.markdown('<div class="section-heading">Recommended Management Priorities</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    for i, (title, body) in enumerate(recs):
        card = f"""
<div class="rec-card">
    <div class="rec-title">{i+1}. {html.escape(title)}</div>
    <div class="rec-body">{html.escape(body)}</div>
</div>
"""
        with (r1 if i % 2 == 0 else r2):
            st.markdown(card, unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-heading">Model-Based Growth Scenario Ranking</div>', unsafe_allow_html=True)

    scenarios = scenario_analysis(values)
    display = scenarios.copy()
    display["Predicted Revenue"] = display["Predicted Revenue"].map(rupee_full)
    display["Predicted Uplift"] = display["Predicted Uplift"].map(rupee_full)
    display["Uplift %"] = display["Uplift %"].map(lambda x: f"{x:.1f}%")

    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_df = scenarios[["Scenario", "Predicted Uplift"]].set_index("Scenario")
    st.caption("Predicted monthly revenue uplift by single-lever scenario")
    st.bar_chart(chart_df, use_container_width=True)

    top = scenarios.iloc[0]
    st.markdown(
        f"""
<div class="executive-box">
    <b>Highest model-implied lever:</b> {html.escape(top["Scenario"])}<br><br>
    If all other inputs remain constant, the model predicts an uplift of
    <b>{rupee_full(top["Predicted Uplift"])}</b> per month
    ({top["Uplift %"]:.1f}% vs the current prediction).
</div>
""",
        unsafe_allow_html=True,
    )


with tab3:
    st.markdown('<div class="section-heading">Business Health Signals vs Dataset Median</div>', unsafe_allow_html=True)

    medians = data[NUMERICAL_FEATURES].median()

    def pct_gap(value, median_value):
        return ((value - median_value) / median_value * 100) if median_value else 0

    st.markdown(
        f"""
<div class="signal-grid">
    <div class="signal">
        <div class="signal-label">Daily Customers</div>
        <div class="signal-value">{values["Average_Daily_Customers"]:,.0f}</div>
        <div class="signal-sub">{pct_gap(values["Average_Daily_Customers"], medians["Average_Daily_Customers"]):+.1f}% vs median</div>
    </div>
    <div class="signal">
        <div class="signal-label">Average Bill Value</div>
        <div class="signal-value">{rupee_short(values["Average_Bill_Value"])}</div>
        <div class="signal-sub">{pct_gap(values["Average_Bill_Value"], medians["Average_Bill_Value"]):+.1f}% vs median</div>
    </div>
    <div class="signal">
        <div class="signal-label">Delivery Share</div>
        <div class="signal-value">{values["Delivery_Order_Percentage"]:.1f}%</div>
        <div class="signal-sub">{values["Delivery_Order_Percentage"] - medians["Delivery_Order_Percentage"]:+.1f} pp vs median</div>
    </div>
    <div class="signal">
        <div class="signal-label">Seating Capacity</div>
        <div class="signal-value">{values["Seating_Capacity"]:,.0f}</div>
        <div class="signal-sub">{pct_gap(values["Seating_Capacity"], medians["Seating_Capacity"]):+.1f}% vs median</div>
    </div>
    <div class="signal">
        <div class="signal-label">Promotion Spend</div>
        <div class="signal-value">{rupee_short(values["Promotion_Spend"])}</div>
        <div class="signal-sub">{pct_gap(values["Promotion_Spend"], medians["Promotion_Spend"]):+.1f}% vs median</div>
    </div>
    <div class="signal">
        <div class="signal-label">Customer Rating</div>
        <div class="signal-value">{values["Customer_Rating"]:.1f}</div>
        <div class="signal-sub">{values["Customer_Rating"] - medians["Customer_Rating"]:+.1f} points vs median</div>
    </div>
    <div class="signal">
        <div class="signal-label">Cuisine</div>
        <div class="signal-value" style="font-size:1.02rem;">{html.escape(values["Cuisine_Type"])}</div>
        <div class="signal-sub">Current cuisine input</div>
    </div>
    <div class="signal">
        <div class="signal-label">Location</div>
        <div class="signal-value" style="font-size:1.02rem;">{html.escape(values["Location_Type"])}</div>
        <div class="signal-sub">Current location type</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


with tab4:
    st.markdown('<div class="section-heading">Model & Prediction Context</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Records", f"{len(data):,}")
    c2.metric("Reference Test R²", f"{TEST_R2:.3f}")
    c3.metric("Reference Test MAE", rupee_short(TEST_MAE))
    c4.metric("Reference Test RMSE", rupee_short(TEST_RMSE))

    lower = max(0, prediction - TEST_RMSE)
    upper = prediction + TEST_RMSE

    st.info(
        f"**Practical error context:** reference test RMSE is {rupee_full(TEST_RMSE)}. "
        f"A simple ±1 RMSE illustration around the current prediction is approximately "
        f"**{rupee_full(lower)} to {rupee_full(upper)}**. "
        "This is not a formal statistical confidence interval."
    )

    st.write(
        "**Final model inputs:** Average Daily Customers, Average Bill Value, Delivery Order %, "
        "Seating Capacity, Promotion Spend, Customer Rating, Cuisine Type and Location Type."
    )
    st.write(
        "**Important:** The regression predicts **revenue**, not profit. Managers should also consider "
        "contribution margin, food cost, labour cost, customer acquisition cost, capacity constraints "
        "and local market information."
    )


# =========================================================
# EXECUTIVE TAKEAWAY
# =========================================================
scenarios = scenario_analysis(values)
top = scenarios.iloc[0]

st.markdown('<div class="section-heading">Executive Takeaway</div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="executive-box">
    <b>Prediction:</b> {rupee_full(prediction)} per month ({rupee_short(prediction)}).<br><br>
    <b>Benchmark:</b> {html.escape(pos["band"])} — approximately {pos["percentile"]:.0f}th percentile.<br><br>
    <b>Top model-implied growth lever:</b> {html.escape(top["Scenario"])} —
    estimated monthly uplift of <b>{rupee_full(top["Predicted Uplift"])}</b>.<br><br>
    <b>Recommended managerial use:</b> treat this prediction as a planning baseline,
    test the highest-priority lever on a controlled scale, compare actual results with
    the model estimate, and only then decide whether to scale the initiative.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="footer-note">
    <b>Decision-support note:</b> regression-based scenarios represent statistical associations
    and predictive what-if estimates. They do not prove that changing an input will causally
    produce the displayed revenue change.
</div>
""",
    unsafe_allow_html=True,
)
