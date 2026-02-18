from pathlib import Path
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, make_scorer


# =========================
# Config
# =========================
RANDOM_STATE = 42

# UMAP
DEFAULT_N_NEIGHBORS = 25
DEFAULT_MIN_DIST = 0.05
DEFAULT_METRIC = "euclidean"

# Explainability
N_TOP_FEATURES_GLOBAL = 25
N_TOP_FEATURES_LOCAL = 12
PERM_REPEATS = 8
RF_TREES = 600

# Category explain
N_TOP_FEATURES_CATEGORY = 15


# =========================
# Helpers
# =========================
def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)

def category_feature_importance_rf_ovr(
    pgsi_cat_value: str,
    df_raw: pd.DataFrame,
    X: pd.DataFrame,
    Xs: np.ndarray,
    feature_names: list,
    top_k: int = 15,
    perm_repeats: int = 8,
    rf_trees: int = 600,
    random_state: int = 42,
):
    """
    Per-category (cluster) feature importance:
    - Define cluster = pgsi_cat_value
    - Train RF classifier to separate this cluster vs rest (one-vs-rest)
    - Use permutation importance with Average Precision (robust to imbalance)
    - Provide direction via mean(z|class)-mean(z|rest) for interpretability
    """
    y_all = df_raw["pgsi_cat"].astype(str).str.strip().values
    cls = str(pgsi_cat_value).strip()
    mask = (y_all == cls).astype(int)

    # Need minimum samples on both sides
    if mask.sum() < 10 or (1 - mask).sum() < 10:
        return None

    # Average precision scorer (needs probabilities)
    ap_scorer = make_scorer(average_precision_score, needs_proba=True)

    clf = RandomForestClassifier(
        n_estimators=rf_trees,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    clf.fit(Xs, mask)

    pi = permutation_importance(
        clf,
        Xs,
        mask,
        scoring=ap_scorer,
        n_repeats=perm_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    out = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_ap": pi.importances_mean,
        "perm_importance_std": pi.importances_std,
    }).sort_values("perm_importance_ap", ascending=False).reset_index(drop=True)

    # Direction signal (in scaled space): mean(z|class)-mean(z|rest)
    Zc = Xs[mask == 1]
    Zr = Xs[mask == 0]
    delta = Zc.mean(axis=0) - Zr.mean(axis=0)
    out["mean(z|class) - mean(z|rest)"] = [delta[i] for i in range(len(feature_names))]

    # Combined score (optional): direction magnitude × permutation importance
    out["cluster_driver_score"] = out["perm_importance_ap"].abs() * out["mean(z|class) - mean(z|rest)"].abs()

    out = out.sort_values("cluster_driver_score", ascending=False).head(top_k).reset_index(drop=True)

    # Nicely named columns
    show = out[[
        "feature",
        "mean(z|class) - mean(z|rest)",
        "perm_importance_ap",
        "perm_importance_std",
        "cluster_driver_score",
    ]].rename(columns={
        "perm_importance_ap": "perm importance (AP)",
        "perm_importance_std": "perm std",
    })

    return show


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    # Excel
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    # Por si alguien sube CSV igualmente
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload an Excel (.xlsx/.xls) or CSV file.")

    df.columns = [c.strip().lower() for c in df.columns]
    return df



@st.cache_resource(show_spinner=True)
def build_umap_and_importance(df: pd.DataFrame, n_neighbors: int, min_dist: float, metric: str):
    required = {"token", "pgsi_cat"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # Features
    drop_cols = {"token", "pgsi", "pgsi_cat", "first_session", "last_session"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = safe_numeric_frame(df[feat_cols].select_dtypes(include=[np.number]))

    if X.shape[1] == 0:
        raise ValueError("No numeric features found in the input CSV.")

    # Labels -> numeric for supervised UMAP
    y_str = df["pgsi_cat"].astype(str).str.strip()
    le = LabelEncoder()
    y = le.fit_transform(y_str).astype(np.int32)

    # Preprocess
    prep = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    Xs = prep.fit_transform(X)

    # Supervised UMAP 3D
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=RANDOM_STATE,
    )
    Z = reducer.fit_transform(Xs, y=y)

    if not np.isfinite(Z).all():
        raise ValueError("UMAP produced NaN/Inf. Check input features for extreme values.")

    emb_df = df[["token", "pgsi_cat"]].copy()
    emb_df["token"] = emb_df["token"].astype(str)
    emb_df["pgsi_cat"] = emb_df["pgsi_cat"].astype(str).str.strip()
    emb_df["u1"], emb_df["u2"], emb_df["u3"] = Z[:, 0], Z[:, 1], Z[:, 2]

    # Axis explainability via permutation importance on regressors u1/u2/u3
    feature_names = X.columns.tolist()
    rf_params = dict(n_estimators=RF_TREES, random_state=RANDOM_STATE, n_jobs=-1)

    reg1 = RandomForestRegressor(**rf_params).fit(Xs, emb_df["u1"].values)
    reg2 = RandomForestRegressor(**rf_params).fit(Xs, emb_df["u2"].values)
    reg3 = RandomForestRegressor(**rf_params).fit(Xs, emb_df["u3"].values)

    pi1 = permutation_importance(
        reg1, Xs, emb_df["u1"].values,
        n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
    )
    pi2 = permutation_importance(
        reg2, Xs, emb_df["u2"].values,
        n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
    )
    pi3 = permutation_importance(
        reg3, Xs, emb_df["u3"].values,
        n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
    )

    imp = pd.DataFrame({
        "feature": feature_names,
        "perm_u1": pi1.importances_mean,
        "perm_u2": pi2.importances_mean,
        "perm_u3": pi3.importances_mean,
    })
    imp["perm_sum_abs"] = imp[["perm_u1", "perm_u2", "perm_u3"]].abs().sum(axis=1)
    imp = imp.sort_values("perm_sum_abs", ascending=False).reset_index(drop=True)

    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    return emb_df, X, Xs, Z, imp, feature_names, label_map


def local_explain_by_token(token: str, df_raw: pd.DataFrame, X: pd.DataFrame, Xs: np.ndarray,
                           imp: pd.DataFrame, feature_names: list):
    idxs = df_raw.index[df_raw["token"].astype(str) == str(token)].tolist()
    if not idxs:
        return None
    i = idxs[0]

    top_global = imp.head(N_TOP_FEATURES_GLOBAL).copy()
    cols = [feature_names.index(f) for f in top_global["feature"].tolist()]

    zvals = Xs[i, cols]
    top_global["z_value"] = zvals
    top_global["raw_value"] = X.iloc[i, cols].values

    # Local proxy: z * global importance
    top_global["local_driver_score"] = top_global["z_value"] * top_global["perm_sum_abs"]

    top_local = (
        top_global.sort_values("local_driver_score", ascending=False)
        .head(N_TOP_FEATURES_LOCAL)
        .reset_index(drop=True)
    )

    out = top_local[["feature", "raw_value", "z_value", "perm_sum_abs", "local_driver_score"]].copy()
    out = out.rename(columns={
        "raw_value": "raw",
        "z_value": "z (scaled)",
        "perm_sum_abs": "global importance",
        "local_driver_score": "local driver score",
    })
    return i, out


def category_explain(pgsi_cat_value: str, df_raw: pd.DataFrame, Xs: np.ndarray,
                     imp: pd.DataFrame, feature_names: list):
    y = df_raw["pgsi_cat"].astype(str).str.strip().values
    cls = str(pgsi_cat_value).strip()
    mask = (y == cls)

    if mask.sum() < 5 or (~mask).sum() < 5:
        return None

    top_global = imp.head(N_TOP_FEATURES_GLOBAL).copy()
    cols = [feature_names.index(f) for f in top_global["feature"].tolist()]

    Zc = Xs[mask][:, cols]
    Zr = Xs[~mask][:, cols]
    delta = Zc.mean(axis=0) - Zr.mean(axis=0)

    out = top_global.copy()
    out["mean(z|class) - mean(z|rest)"] = delta
    out["category driver score"] = np.abs(out["mean(z|class) - mean(z|rest)"]) * out["perm_sum_abs"]

    out = (
        out.sort_values("category driver score", ascending=False)
        .head(N_TOP_FEATURES_CATEGORY)
        .reset_index(drop=True)
    )

    show = out[["feature", "mean(z|class) - mean(z|rest)", "perm_sum_abs", "category driver score"]].rename(columns={
        "perm_sum_abs": "global importance",
    })
    return show


def resolve_selected_token(tokens: list, typed: str, fallback: str) -> str:
    """Pick token by: exact match first, else contains match, else fallback."""
    if typed is None:
        return fallback
    t = str(typed).strip()
    if t == "":
        return fallback

    # exact
    for tok in tokens:
        if tok == t:
            return tok

    # contains
    cand = [tok for tok in tokens if t in tok]
    if len(cand) > 0:
        return cand[0]

    return fallback

def plot_category_bar(cat_df: pd.DataFrame, top_k: int = 15):
    """
    Expects cat_df with columns:
      - feature
      - mean(z|class) - mean(z|rest)
      - global importance
      - category driver score
    Produces a signed bar plot (positive/negative direction) using the score.
    """
    if cat_df is None or len(cat_df) == 0:
        return None

    d = cat_df.copy()

    # signed score: direction * magnitude
    d["signed_score"] = np.sign(d["mean(z|class) - mean(z|rest)"]) * d["category driver score"]

    # keep top-k by absolute signed score
    d = d.reindex(d["signed_score"].abs().sort_values(ascending=False).index).head(top_k)

    # nicer label: show direction explicitly
    d["direction"] = np.where(d["signed_score"] >= 0, "higher in class", "higher in rest")

    # sort for horizontal bar
    d = d.sort_values("signed_score", ascending=True)

    fig = px.bar(
        d,
        x="signed_score",
        y="feature",
        orientation="h",
        color="direction",
        hover_data=[
            "mean(z|class) - mean(z|rest)",
            "global importance",
            "category driver score",
        ],
        title="Category feature effects (signed): + pushes toward class, − pushes away",
    )
    fig.update_layout(height=420)
    return fig

# =========================
# UI
# =========================
st.set_page_config(page_title="Supervised UMAP 3D + Token Search + Highlight", layout="wide")
st.title("Supervised UMAP (3D) + Local & Category Explainability")
st.caption("Search/select a token. Selected point is highlighted. Local explanation on the right; category-level drivers below the plot.")

from PIL import Image

with st.sidebar:
    st.image(Image.open("assets/igi.png"), use_container_width=True)
    st.image(Image.open("assets/air_hub.png"), use_container_width=True)
    st.markdown("---")
    # luego ya tus sliders/inputs...

    st.header("Data upload (required)")
    uploaded = st.file_uploader(
        "Upload your player features file",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=False
    )
    st.caption("The app only runs after you upload a file. No local paths are used.")


    st.header("UMAP params")
    n_neighbors = st.slider("n_neighbors", 5, 200, DEFAULT_N_NEIGHBORS, step=1)
    min_dist = st.slider("min_dist", 0.0, 0.99, float(DEFAULT_MIN_DIST), step=0.01)
    metric = st.selectbox("metric", ["euclidean", "manhattan", "cosine"], index=0)

    st.markdown("---")
    st.header("Color")
    color_mode = st.selectbox("Color points by", ["pgsi_cat", "binary (PGSI 8+ vs rest)"])

    st.markdown("---")
    st.header("Token search")
    typed_token = st.text_input("Search token (exact or substring)", value="")
    st.caption("Tip: paste the full token; otherwise paste part of it and it will pick the first match.")

    st.markdown("---")
    show_bar = st.checkbox("Show local driver bar chart", value=True)


# Load + build
# =========================
# Load + build (only after upload)
# =========================
df_raw = load_data_from_upload(uploaded)

if df_raw is None:
    st.info("⬅️ Please upload your Excel file from the sidebar to start.")
    st.stop()

# Optional: sanity check early
required = {"token", "pgsi_cat"}
miss = required - set(df_raw.columns)
if miss:
    st.error(f"Missing required columns in uploaded file: {miss}")
    st.stop()

emb_df, X, Xs, Z, imp, feature_names, label_map = build_umap_and_importance(
    df_raw, n_neighbors, min_dist, metric
)

plot_df = emb_df.copy()
plot_df["token"] = plot_df["token"].astype(str)

if color_mode.startswith("binary"):
    plot_df["pgsi_bin"] = (plot_df["pgsi_cat"].astype(str).str.strip() == "PGSI 8+").map(
        {True: "PGSI 8+", False: "Other"}
    )
    color_col = "pgsi_bin"
else:
    color_col = "pgsi_cat"

tokens = plot_df["token"].tolist()

with st.sidebar:
    fallback_token = st.selectbox("Fallback token selector", options=tokens, index=0)

sel_token = resolve_selected_token(tokens, typed_token, fallback_token)

# Layout
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("UMAP 3D")

    fig = px.scatter_3d(
        plot_df, x="u1", y="u2", z="u3",
        color=color_col,
        hover_data=["token", "pgsi_cat"],
        opacity=0.65,
        title="Supervised UMAP 3D"
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=740)

    sel_row = plot_df[plot_df["token"] == str(sel_token)]
    if len(sel_row) == 1:
        # Halo ring (big, low opacity)
        fig_ring = px.scatter_3d(sel_row, x="u1", y="u2", z="u3", hover_data=["token", "pgsi_cat"])
        fig_ring.update_traces(marker=dict(size=18, symbol="circle", opacity=0.18))
        fig.add_traces(fig_ring.data)

        # Selected point (diamond)
        fig_sel = px.scatter_3d(sel_row, x="u1", y="u2", z="u3", hover_data=["token", "pgsi_cat"])
        fig_sel.update_traces(marker=dict(size=10, symbol="diamond", opacity=1.0))
        fig.add_traces(fig_sel.data)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Cluster explainability (PGSI categories)")

pgsi_values = sorted(df_raw["pgsi_cat"].astype(str).str.strip().unique().tolist())
chosen_cat = st.selectbox("Select pgsi_cat (cluster)", options=pgsi_values, index=0, key="cat_sel")

method = st.radio(
    "Method",
    options=[
        "Geometry (UMAP-axis weighted mean diff)",
        "RF one-vs-rest (permutation importance)",
    ],
    index=1,
    horizontal=False,
)

if method.startswith("Geometry"):
    cat_df = category_explain(chosen_cat, df_raw, Xs, imp, feature_names)
    if cat_df is None:
        st.warning("Not enough samples in this category to compute a stable explanation.")
    else:
        st.dataframe(cat_df, use_container_width=True, height=280)

        st.markdown("#### Category feature effects (bar plot)")
        st.caption(
            "Positive = feature is higher for this PGSI group vs rest (scaled space). "
            "Negative = higher in the rest. Magnitude is weighted by global UMAP-axis importance."
        )
        fig_cat = plot_category_bar(cat_df, top_k=N_TOP_FEATURES_CATEGORY)
        if fig_cat is not None:
            st.plotly_chart(fig_cat, use_container_width=True)

else:
    rf_df = category_feature_importance_rf_ovr(
        chosen_cat,
        df_raw=df_raw,
        X=X,
        Xs=Xs,
        feature_names=feature_names,
        top_k=N_TOP_FEATURES_CATEGORY,
        perm_repeats=PERM_REPEATS,
        rf_trees=RF_TREES,
        random_state=RANDOM_STATE,
    )
    if rf_df is None:
        st.warning("Not enough samples in this category to compute a stable RF explanation.")
    else:
        st.dataframe(rf_df, use_container_width=True, height=320)

        st.markdown("#### Cluster feature effects (signed, RF-weighted)")
        st.caption(
            "Signed direction comes from mean(z|class)-mean(z|rest). "
            "Ranking uses permutation importance (Average Precision) × |direction|."
        )

        # Reuse your bar plot helper by adapting column names
        rf_plot = rf_df.rename(columns={
            "perm importance (AP)": "global importance",  # just for hover naming reuse
            "cluster_driver_score": "category driver score",  # reuse
        })

        # plot_category_bar expects these exact columns:
        rf_plot = rf_plot.rename(columns={
            "mean(z|class) - mean(z|rest)": "mean(z|class) - mean(z|rest)",
            "category driver score": "category driver score",
            "global importance": "global importance",
        })

        fig_rf = plot_category_bar(
            rf_plot[["feature", "mean(z|class) - mean(z|rest)", "global importance", "category driver score"]],
            top_k=N_TOP_FEATURES_CATEGORY
        )
        if fig_rf is not None:
            st.plotly_chart(fig_rf, use_container_width=True)


with right:
    st.subheader("Local explanation")
    st.caption("Local driver score = z(feature) × global importance(feature) across UMAP axes.")

    st.markdown("**Label mapping (pgsi_cat → int for UMAP supervision):**")
    st.write(label_map)

    st.markdown("---")
    st.markdown(f"**Selected token:** `{sel_token}`")

    meta_row = plot_df.loc[plot_df["token"] == str(sel_token)]
    if len(meta_row) == 1:
        meta_row = meta_row.iloc[0]
        st.markdown(f"**PGSI:** {meta_row['pgsi_cat']}")
        st.markdown(f"**UMAP:** ({meta_row['u1']:.3f}, {meta_row['u2']:.3f}, {meta_row['u3']:.3f})")

    res = local_explain_by_token(sel_token, df_raw, X, Xs, imp, feature_names)
    if res is None:
        st.warning(f"Token not found: {sel_token}")
    else:
        _, local_df = res

        st.markdown("### Top drivers")
        st.dataframe(local_df, use_container_width=True, height=420)

        if show_bar:
            bar = local_df.sort_values("local driver score", ascending=True)
            fig_bar = px.bar(
                bar,
                x="local driver score",
                y="feature",
                orientation="h",
                title="Local driver score (higher = stronger influence)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
