"""House Price Prediction and Analysis - Streamlit app.

Pages: Overview, Data Explorer, Analysis, Model Training, Predict Price.
Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
from model_utils import (
    BASELINE_NAME,
    CAT_FEATURES,
    FEATURES,
    NUM_FEATURES,
    TARGET,
    clean_data,
    missing_columns,
    ordered_values,
    predict_price,
    train_and_evaluate,
)

DATA_FILE = Path(__file__).parent / "House_Price_Prediction_Dataset_Analysis.csv"
PAGES = ["Overview", "Data Explorer", "Analysis", "Model Training", "Predict Price"]

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

CSS = """<style>
.hero {background:#16324F; padding:1.5rem 1.8rem; border-radius:14px; margin-bottom:1.2rem;}
.hero h1 {color:#FFFFFF !important; margin:0 0 .3rem 0; padding:0; font-size:2rem;}
.hero p {color:#D5E2EC !important; margin:0; font-size:1.02rem;}
[data-testid="stMetric"] {border:1px solid rgba(128,148,168,.40); border-radius:12px; padding:.7rem 1rem;}
.result-card {border-left:6px solid #2A9D8F; background:rgba(42,157,143,.12); border-radius:10px; padding:1.1rem 1.4rem; margin:.4rem 0 1rem 0;}
.result-card .label {font-size:.95rem; opacity:.8;}
.result-card .value {font-size:2.5rem; font-weight:700; line-height:1.2;}
.result-card .range {font-size:.95rem; opacity:.85; margin-top:.2rem;}
[data-testid="stFormSubmitButton"] button {background:#2A9D8F; color:#FFFFFF; border:none;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def money(value: float) -> str:
    return f"${value:,.0f}"


def show(fig) -> None:
    st.pyplot(fig)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_data():
    """Load the CSV that sits next to app.py (or let the user upload one)."""
    if DATA_FILE.exists():
        raw = load_csv(str(DATA_FILE))
    else:
        st.warning(f"Dataset file '{DATA_FILE.name}' was not found next to app.py.")
        uploaded = st.file_uploader("Upload the house price CSV", type="csv", key="upload_csv")
        if uploaded is None:
            st.stop()
        raw = pd.read_csv(uploaded)
    raw = raw.rename(columns=lambda c: str(c).strip())
    missing = missing_columns(raw)
    if missing:
        st.error(f"The dataset is missing these columns: {', '.join(missing)}")
        st.stop()
    return clean_data(raw)


@st.cache_resource(show_spinner=False)
def _train_cached(_df: pd.DataFrame, data_key: str, test_size: float, seed: int) -> dict:
    return train_and_evaluate(_df, test_size=test_size, seed=seed)


def get_bundle(df: pd.DataFrame, test_size: float, seed: int) -> dict:
    data_key = f"{len(df)}-{int(df[TARGET].sum())}-{int(df['Area'].sum())}-{int(df['YearBuilt'].sum())}"
    return _train_cached(df, data_key, test_size, seed)


def price_column_config():
    return {"Price": st.column_config.NumberColumn("Price", format="$%d")}


def hero(title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(df: pd.DataFrame, quality: dict) -> None:
    hero(
        "House Price Prediction and Analysis",
        f"Explore {len(df):,} house listings, compare machine-learning models "
        "and estimate the price of a house.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Houses", f"{len(df):,}")
    c2.metric("Average price", money(df[TARGET].mean()))
    c3.metric("Median price", money(df[TARGET].median()))
    c4.metric("Price range", f"${df[TARGET].min() / 1000:,.0f}k to ${df[TARGET].max() / 1000:,.0f}k")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Input columns", len(FEATURES))
    c6.metric("Missing values", f"{quality['missing_cells']:,}")
    c7.metric("Duplicate rows removed", f"{quality['duplicates']:,}")
    c8.metric("Locations", df["Location"].nunique())

    st.subheader("What each column means")
    descriptions = {
        "Id": "Listing number (only an identifier, not used by the model)",
        "Area": "Living area of the house",
        "Bedrooms": "Number of bedrooms",
        "Bathrooms": "Number of bathrooms",
        "Floors": "Number of floors",
        "YearBuilt": "Year the house was built",
        "Location": "Type of neighbourhood",
        "Condition": "Condition of the house",
        "Garage": "Whether the house has a garage",
        "Price": "Selling price (the value we want to predict)",
    }
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            values = f"{df[col].min():,.0f} to {df[col].max():,.0f}"
        else:
            values = ", ".join(str(v) for v in ordered_values(df, col))
        rows.append({"Column": col, "Meaning": descriptions.get(col, ""), "Values": values})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Data preview")
    st.dataframe(df.head(20), hide_index=True, column_config=price_column_config())

    st.subheader("How to use this app")
    st.markdown(
        "- **Data Explorer**: filter the houses and download what you filtered.\n"
        "- **Analysis**: charts and key findings about prices.\n"
        "- **Model Training**: the data is split into training and test sets, several "
        "models are trained and compared on unseen houses.\n"
        "- **Predict Price**: enter a house's details and get an estimated price."
    )


def page_explorer(df: pd.DataFrame) -> None:
    hero("Data Explorer", "Filter the listings and see how the numbers change.")

    c1, c2, c3 = st.columns(3)
    loc_values = ordered_values(df, "Location")
    cond_values = ordered_values(df, "Condition")
    locs = c1.multiselect("Location", loc_values, default=loc_values, key="ex_loc")
    conds = c2.multiselect("Condition", cond_values, default=cond_values, key="ex_cond")
    garage = c3.selectbox("Garage", ["Any", "Yes", "No"], index=0, key="ex_garage")

    p_min = int(math.floor(df[TARGET].min() / 1000) * 1000)
    p_max = int(math.ceil(df[TARGET].max() / 1000) * 1000)
    a_min, a_max = int(df["Area"].min()), int(df["Area"].max())
    c4, c5 = st.columns(2)
    price_range = c4.slider("Price range ($)", p_min, p_max, (p_min, p_max), step=1000, key="ex_price")
    area_range = c5.slider("Area", a_min, a_max, (a_min, a_max), step=1, key="ex_area")

    mask = (
        df["Location"].isin(locs)
        & df["Condition"].isin(conds)
        & df[TARGET].between(price_range[0], price_range[1])
        & df["Area"].between(area_range[0], area_range[1])
    )
    if garage != "Any":
        mask &= df["Garage"] == garage
    filtered = df[mask]

    if filtered.empty:
        st.info("No houses match these filters. Widen a filter to see results.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Matching houses", f"{len(filtered):,}")
    m2.metric("Share of all houses", f"{len(filtered) / len(df):.0%}")
    m3.metric("Average price", money(filtered[TARGET].mean()))
    m4.metric("Average area", f"{filtered['Area'].mean():,.0f}")
    m5.metric("With garage", f"{(filtered['Garage'] == 'Yes').mean():.0%}")

    st.dataframe(filtered, hide_index=True, column_config=price_column_config())
    st.download_button(
        "Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_houses.csv",
        mime="text/csv",
        key="ex_download",
    )


def key_findings(df: pd.DataFrame) -> list[str]:
    """Plain-English findings that are computed from the data itself."""
    out = []
    mean, std = df[TARGET].mean(), df[TARGET].std()
    out.append(
        f"The average price is **{money(mean)}**, but prices vary a lot "
        f"(standard deviation {money(std)})."
    )

    by_loc = df.groupby("Location")[TARGET].mean().sort_values(ascending=False)
    gap = (by_loc.iloc[0] / by_loc.iloc[-1] - 1) * 100
    out.append(
        f"**{by_loc.index[0]}** has the highest average price ({money(by_loc.iloc[0])}) and "
        f"**{by_loc.index[-1]}** the lowest ({money(by_loc.iloc[-1])}), a gap of {gap:.1f}%."
    )

    by_cond = df.groupby("Condition")[TARGET].mean().sort_values(ascending=False)
    out.append(
        f"By condition, **{by_cond.index[0]}** houses average {money(by_cond.iloc[0])} and "
        f"**{by_cond.index[-1]}** houses average {money(by_cond.iloc[-1])}."
    )

    by_garage = df.groupby("Garage")[TARGET].mean()
    if {"Yes", "No"} <= set(by_garage.index):
        out.append(
            f"Houses with a garage average {money(by_garage['Yes'])} versus "
            f"{money(by_garage['No'])} without one."
        )

    corr = df[NUM_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    top = corr.abs().sort_values(ascending=False).index[0]
    r = corr[top]
    strength = "very weak" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.6 else "strong"
    out.append(
        f"The numeric column most related to price is **{top}** (correlation {r:+.3f}), "
        f"which is a **{strength}** relationship."
    )
    return out


def page_analysis(df: pd.DataFrame) -> None:
    hero("Analysis", "Charts and findings that describe the price data.")

    tab_find, tab_price, tab_cat, tab_num, tab_corr = st.tabs(
        ["Key findings", "Price distribution", "Categories", "Numeric columns", "Correlation"]
    )

    with tab_find:
        for line in key_findings(df):
            st.markdown(f"- {line}")
        st.caption("Findings are calculated from the dataset, so they update if the data changes.")
        show(charts.decade_line(df))

    with tab_price:
        left, right = st.columns([3, 2])
        with left:
            show(charts.price_histogram(df))
        with right:
            st.subheader("Price summary")
            summary = df[TARGET].describe().rename(
                {"count": "Houses", "mean": "Mean", "std": "Std. deviation", "min": "Minimum",
                 "25%": "25th percentile", "50%": "Median", "75%": "75th percentile", "max": "Maximum"}
            )
            table = pd.DataFrame({"Value": summary.values}, index=summary.index)
            table["Value"] = [f"{v:,.0f}" for v in table["Value"]]
            st.dataframe(table)

    with tab_cat:
        col = st.radio("Group houses by", CAT_FEATURES, horizontal=True, key="an_cat")
        left, right = st.columns(2)
        with left:
            show(charts.mean_price_bar(df, col))
        with right:
            show(charts.price_boxplot(df, col))
        counts = df[col].value_counts().rename("Houses").reset_index()
        counts.columns = [col, "Houses"]
        st.dataframe(counts, hide_index=True)

    with tab_num:
        col = st.selectbox("Choose a numeric column", NUM_FEATURES, index=0, key="an_num")
        left, right = st.columns(2)
        with left:
            show(charts.scatter_vs_price(df, col))
        with right:
            show(charts.mean_price_by_bucket(df, col))

    with tab_corr:
        show(charts.correlation_heatmap(df, NUM_FEATURES + [TARGET]))
        st.caption(
            "Values near 0 mean two columns are not linearly related. "
            "Values near +1 or -1 mean a strong relationship."
        )


def page_model(df: pd.DataFrame, test_size: float, seed: int) -> None:
    hero("Model Training", "Train several models on part of the data and test them on houses they have never seen.")

    with st.spinner("Training models... this takes a few seconds the first time."):
        bundle = get_bundle(df, test_size, seed)
    results = bundle["results"]
    best = bundle["best_name"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training houses", f"{len(bundle['X_train']):,}")
    c2.metric("Test houses", f"{len(bundle['X_test']):,}")
    c3.metric("Test share", f"{test_size:.0%}")
    c4.metric("Selected model", best)

    with st.expander("How the models are built", expanded=False):
        st.markdown(
            f"1. The **{len(FEATURES)} input columns** ({', '.join(FEATURES)}) are used to predict **{TARGET}**.\n"
            "2. Numbers are scaled and text columns (Location, Condition, Garage) are one-hot encoded.\n"
            f"3. The data is split into **{1 - test_size:.0%} training** and **{test_size:.0%} test** houses "
            f"(random seed {seed}). Change both in the sidebar.\n"
            "4. Each model is scored with 5-fold cross-validation on the training data and then on the test data.\n"
            "5. The model with the best cross-validation R\u00b2 is selected (the baseline is only for comparison).\n"
            "6. **R\u00b2** shows how much of the price variation is explained (1 = all, 0 = nothing). "
            "**MAE** is the average error in dollars."
        )

    st.subheader("Model comparison")
    table = results.copy()
    table.insert(1, "Selected", ["yes" if m == best else "" for m in table["Model"]])
    for col in ["CV R2 (train)", "Train R2", "Test R2"]:
        table[col] = table[col].round(3)
    for col in ["Test MAE", "Test RMSE"]:
        table[col] = table[col].round(0).astype(int)
    st.dataframe(table, hide_index=True)
    show(charts.model_comparison(results))

    best_row = results[results["Model"] == best].iloc[0]
    baseline_row = results[results["Model"] == BASELINE_NAME].iloc[0]
    if best_row["Test R2"] < 0.10:
        st.warning(
            f"**Important:** even the best model explains almost none of the price variation "
            f"(test R\u00b2 = {best_row['Test R2']:.3f}) and is no better than the baseline that always "
            f"predicts the average price (test R\u00b2 = {baseline_row['Test R2']:.3f}). "
            "In this dataset price has almost no relationship with the input columns "
            "(see the Correlation tab in Analysis), so predictions stay close to the average price. "
            "This is a property of the data, not a bug in the app."
        )
    else:
        st.success(
            f"The best model explains about {best_row['Test R2']:.0%} of the price variation on unseen houses "
            f"and is off by about {money(best_row['Test MAE'])} on average."
        )

    st.subheader("Look closer at one model")
    names = [n for n in results["Model"] if n != BASELINE_NAME]
    chosen = st.selectbox("Model", names, index=names.index(best), key="mt_model")
    row = results[results["Model"] == chosen].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test R\u00b2", f"{row['Test R2']:.3f}")
    m2.metric("Train R\u00b2", f"{row['Train R2']:.3f}")
    m3.metric("Test MAE", money(row["Test MAE"]))
    m4.metric("Test RMSE", money(row["Test RMSE"]))

    left, right = st.columns(2)
    y_test = bundle["y_test"]
    pred = bundle["test_preds"][chosen]
    with left:
        show(charts.actual_vs_predicted(y_test, pred, chosen))
    with right:
        show(charts.residual_histogram(y_test, pred))
    show(charts.importance_bar(bundle["importances"][chosen]))


def page_predict(df: pd.DataFrame, test_size: float, seed: int) -> None:
    hero("Predict Price", "Describe a house and get an estimated price from the trained model.")

    with st.spinner("Preparing models..."):
        bundle = get_bundle(df, test_size, seed)
    results = bundle["results"]
    names = [n for n in results["Model"] if n != BASELINE_NAME]
    best = bundle["best_name"]

    def default_index(options, value):
        return options.index(value) if value in options else 0

    loc_opts = ordered_values(df, "Location")
    cond_opts = ordered_values(df, "Condition")
    garage_opts = ordered_values(df, "Garage")
    bed_opts = [int(v) for v in sorted(df["Bedrooms"].unique())]
    bath_opts = [int(v) for v in sorted(df["Bathrooms"].unique())]
    floor_opts = [int(v) for v in sorted(df["Floors"].unique())]

    with st.form("predict_form"):
        a, b, c = st.columns(3)
        area = a.number_input(
            "Area", min_value=int(df["Area"].min()), max_value=int(df["Area"].max()),
            value=int(df["Area"].median()), step=50, key="in_area",
        )
        year = a.number_input(
            "Year built", min_value=int(df["YearBuilt"].min()), max_value=int(df["YearBuilt"].max()),
            value=int(df["YearBuilt"].median()), step=1, key="in_year",
        )
        bedrooms = b.selectbox("Bedrooms", bed_opts, index=default_index(bed_opts, 3), key="in_bed")
        bathrooms = b.selectbox("Bathrooms", bath_opts, index=default_index(bath_opts, 2), key="in_bath")
        floors = b.selectbox("Floors", floor_opts, index=default_index(floor_opts, 2), key="in_floors")
        location = c.selectbox("Location", loc_opts, index=0, key="in_loc")
        condition = c.selectbox("Condition", cond_opts, index=default_index(cond_opts, "Good"), key="in_cond")
        garage = c.selectbox("Garage", garage_opts, index=default_index(garage_opts, "Yes"), key="in_garage")
        model_name = st.selectbox("Model used for the estimate", names, index=names.index(best), key="in_model")
        submitted = st.form_submit_button("Predict price")

    if submitted:
        st.session_state["last_inputs"] = {
            "Area": int(area), "Bedrooms": int(bedrooms), "Bathrooms": int(bathrooms),
            "Floors": int(floors), "YearBuilt": int(year), "Location": location,
            "Condition": condition, "Garage": garage, "model": model_name,
        }

    saved = st.session_state.get("last_inputs")
    if not saved:
        st.info("Fill in the house details above and press **Predict price**.")
        return

    used_model = saved["model"] if saved["model"] in bundle["pipelines"] else best
    inputs = {k: saved[k] for k in FEATURES}
    estimate = predict_price(bundle["pipelines"][used_model], inputs)
    row = results[results["Model"] == used_model].iloc[0]
    mae = float(row["Test MAE"])
    low, high = max(estimate - mae, 0.0), estimate + mae

    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            f'<div class="result-card"><div class="label">Estimated price ({used_model})</div>'
            f'<div class="value">{money(estimate)}</div>'
            f'<div class="range">Typical error about {money(mae)}, so roughly '
            f"{money(low)} to {money(high)}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Your inputs**")
        st.dataframe(
            pd.DataFrame({"Column": FEATURES, "Value": [str(inputs[f]) for f in FEATURES]}),
            hide_index=True,
        )
    with right:
        loc_avg = float(df.loc[df["Location"] == inputs["Location"], TARGET].mean())
        show(charts.prediction_comparison(estimate, inputs["Location"], loc_avg, float(df[TARGET].mean())))

    if row["Test R2"] < 0.10:
        st.warning(
            f"Model quality is low (test R\u00b2 = {row['Test R2']:.3f}). Price in this dataset barely "
            "depends on the inputs, so treat the estimate as close to the average price rather than a precise value."
        )
    else:
        st.caption(f"On unseen houses this model has R\u00b2 = {row['Test R2']:.3f}.")

    st.subheader("Similar houses in the data")
    similar = df[(df["Location"] == inputs["Location"]) & (df["Condition"] == inputs["Condition"])].copy()
    if len(similar) < 5:
        similar = df[df["Location"] == inputs["Location"]].copy()
    similar["gap"] = (similar["Area"] - inputs["Area"]).abs()
    similar = similar.sort_values("gap").head(5).drop(columns="gap")
    st.dataframe(similar, hide_index=True, column_config=price_column_config())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df, quality = get_data()

    st.sidebar.title("House Price App")
    page = st.sidebar.radio("Go to", PAGES, index=0, key="nav_page")

    st.sidebar.divider()
    st.sidebar.markdown("**Train / test settings**")
    test_pct = st.sidebar.slider("Test set size (%)", 10, 40, 20, step=5, key="sb_test_pct")
    seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key="sb_seed"))
    st.sidebar.caption("Used on the Model Training and Predict Price pages.")
    test_size = test_pct / 100

    if page == "Overview":
        page_overview(df, quality)
    elif page == "Data Explorer":
        page_explorer(df)
    elif page == "Analysis":
        page_analysis(df)
    elif page == "Model Training":
        page_model(df, test_size, seed)
    else:
        page_predict(df, test_size, seed)


main()
