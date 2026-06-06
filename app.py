"""Streamlit dashboard for the federated-learning medical-imaging framework.

Two views:
  - Run Experiment : configure and launch an FL run on the synthetic
    multi-center playground, with live per-round metric curves and (for
    AHA-Fed) live client-quality bars.
  - Browse Results : load any saved JSON under ``runs/`` and inspect its
    config, training curves, and final metrics.

Launch:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.datasets import SyntheticMedicalImaging
from src.data.partitioning import make_synthetic_clients
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer, build_aggregator
from src.models.backbones import build_model
from src.utils.seed import set_seed

RUNS_DIR = ROOT / "runs"
METRIC_COLS = ["loss", "accuracy", "auroc", "auprc", "ece"]

st.set_page_config(page_title="FL Medical Imaging", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def history_to_df(history: list[dict]) -> pd.DataFrame:
    """History list -> tidy DataFrame indexed by round (numeric metrics only)."""
    rows = []
    for h in history:
        row = {"round": h.get("round")}
        for k in METRIC_COLS:
            v = h.get(k)
            row[k] = v if isinstance(v, (int, float)) else None
        rows.append(row)
    df = pd.DataFrame(rows).set_index("round")
    return df


def render_curves(container, history: list[dict]) -> None:
    """Draw the four metric curves + final-metric cards into a container."""
    if not history:
        return
    df = history_to_df(history)
    latest = history[-1]

    with container.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Round", latest.get("round"))
        c2.metric("Accuracy", _fmt(latest.get("accuracy")))
        c3.metric("Loss", _fmt(latest.get("loss")))
        c4.metric("AUROC", _fmt(latest.get("auroc")))
        c5.metric("ECE", _fmt(latest.get("ece")))

        left, right = st.columns(2)
        if df["loss"].notna().any():
            left.caption("Loss (lower is better)")
            left.line_chart(df[["loss"]])
        if df["accuracy"].notna().any():
            right.caption("Accuracy (higher is better)")
            right.line_chart(df[["accuracy"]])
        cal_cols = [c for c in ["auroc", "auprc", "ece"] if df[c].notna().any()]
        if cal_cols:
            st.caption("AUROC / AUPRC (higher is better) · ECE (lower is better)")
            st.line_chart(df[cal_cols])


def render_quality(container, history: list[dict]) -> None:
    """For AHA-Fed: bar chart of the latest per-client quality signal."""
    latest = history[-1] if history else {}
    q = latest.get("client_quality")
    if not q:
        container.empty()
        return
    qdf = pd.DataFrame(
        {"quality": q}, index=[f"client {i}" for i in range(len(q))]
    )
    with container.container():
        st.caption(
            "AHA-Fed per-client quality (round "
            f"{latest.get('round')}) — a noisy client is driven low"
        )
        st.bar_chart(qdf)


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def find_result_files() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.rglob("*.json"))


# --------------------------------------------------------------------------- #
# Sidebar — experiment configuration
# --------------------------------------------------------------------------- #
def experiment_sidebar() -> dict:
    st.sidebar.header("Experiment config")

    aggregator = st.sidebar.selectbox(
        "Aggregator",
        ["fedavg", "fedprox", "fedbn", "feddisco", "adaptive"],
        index=4,
        help="`adaptive` is AHA-Fed.",
    )

    agg_kwargs: dict = {}
    proximal_mu = 0.0
    if aggregator == "fedprox":
        proximal_mu = st.sidebar.slider("FedProx μ (proximal)", 0.0, 1.0, 0.01, 0.01)
    elif aggregator == "feddisco":
        agg_kwargs["a"] = st.sidebar.slider("FedDisco a (distance penalty)", 0.0, 8.0, 1.0, 0.5)
    elif aggregator == "adaptive":
        st.sidebar.markdown("**AHA-Fed weights**")
        agg_kwargs["alpha"] = st.sidebar.slider("α  log-volume", 0.0, 4.0, 1.0, 0.5)
        agg_kwargs["beta"] = st.sidebar.slider("β  distance penalty", 0.0, 8.0, 2.0, 0.5)
        agg_kwargs["gamma"] = st.sidebar.slider("γ  quality reward", 0.0, 8.0, 4.0, 0.5)

    st.sidebar.divider()
    num_clients = st.sidebar.slider("Clients", 2, 12, 4)
    num_rounds = st.sidebar.slider("Rounds", 2, 60, 20)
    local_epochs = st.sidebar.slider("Local epochs", 1, 5, 1)
    local_lr = st.sidebar.select_slider(
        "Local LR", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01
    )
    samples_per_client = st.sidebar.slider("Samples / client", 50, 500, 200, 50)
    num_classes = st.sidebar.slider("Classes", 2, 8, 4)
    image_size = st.sidebar.select_slider(
        "Image size", options=[16, 32, 64], value=32
    )
    heterogeneity = st.sidebar.slider("Heterogeneity (Dirichlet skew)", 0.0, 1.0, 0.7, 0.1)

    st.sidebar.divider()
    st.sidebar.markdown("**Label-noise injection**")
    noisy_on = st.sidebar.checkbox("Add a noisy client", value=(aggregator == "adaptive"))
    noisy_client = -1
    noisy_rate = 0.0
    if noisy_on:
        noisy_client = st.sidebar.selectbox("Noisy client #", list(range(num_clients)), index=min(3, num_clients - 1))
        noisy_rate = st.sidebar.slider("Noise rate", 0.0, 1.0, 0.5, 0.1)

    st.sidebar.divider()
    st.sidebar.markdown("**Differential privacy (DP-SGD)**")
    dp_on = st.sidebar.checkbox("Enable DP-SGD", value=False)
    dp_epsilon = dp_delta = dp_max_grad_norm = None
    if dp_on:
        dp_epsilon = st.sidebar.slider("Target ε", 0.1, 10.0, 1.0, 0.1)
        dp_delta = st.sidebar.select_slider("δ", options=[1e-3, 1e-4, 1e-5, 1e-6], value=1e-5)
        dp_max_grad_norm = st.sidebar.slider("Max grad norm (clip)", 0.1, 5.0, 1.0, 0.1)

    st.sidebar.divider()
    arch = st.sidebar.selectbox("Backbone", ["small_cnn", "med_cnn", "densenet121"], index=0)
    seed = st.sidebar.number_input("Seed", value=42, step=1)
    cuda = torch.cuda.is_available()
    device = st.sidebar.radio("Device", (["cuda", "cpu"] if cuda else ["cpu"]), horizontal=True)
    save_json = st.sidebar.checkbox("Save results to runs/ui/", value=False)

    return dict(
        aggregator=aggregator, agg_kwargs=agg_kwargs, proximal_mu=proximal_mu,
        num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs,
        local_lr=local_lr, samples_per_client=samples_per_client, num_classes=num_classes,
        image_size=image_size, heterogeneity=heterogeneity, noisy_client=noisy_client,
        noisy_rate=noisy_rate, dp_epsilon=dp_epsilon, dp_delta=dp_delta,
        dp_max_grad_norm=dp_max_grad_norm, arch=arch, seed=int(seed), device=device,
        save_json=save_json,
    )


# --------------------------------------------------------------------------- #
# Training driver (uses the server's round_callback for live updates)
# --------------------------------------------------------------------------- #
def run_experiment(cfg: dict, curves_box, quality_box, progress) -> list[dict]:
    set_seed(cfg["seed"])

    label_noise = [0.0] * cfg["num_clients"]
    if 0 <= cfg["noisy_client"] < cfg["num_clients"]:
        label_noise[cfg["noisy_client"]] = cfg["noisy_rate"]

    client_datasets = make_synthetic_clients(
        num_clients=cfg["num_clients"],
        samples_per_client=cfg["samples_per_client"],
        num_classes=cfg["num_classes"],
        image_size=cfg["image_size"],
        heterogeneity=cfg["heterogeneity"],
        base_seed=cfg["seed"],
        label_noise=label_noise,
    )
    clients = [
        FederatedClient(i, ds, batch_size=32, device=cfg["device"])
        for i, ds in enumerate(client_datasets)
    ]

    eval_ds = SyntheticMedicalImaging(
        num_samples=400, num_classes=cfg["num_classes"], image_size=cfg["image_size"],
        center_id=0, center_offset=0.0, seed=cfg["seed"] + 9999,
    )
    eval_loader = DataLoader(eval_ds, batch_size=32)

    model = build_model(cfg["arch"], num_classes=cfg["num_classes"], pretrained=False)
    server = FederatedServer(
        global_model=model, clients=clients,
        aggregator=build_aggregator(cfg["aggregator"], **cfg["agg_kwargs"]),
        eval_loader=eval_loader, num_classes=cfg["num_classes"], device=cfg["device"],
    )

    dp_config = None
    if cfg["dp_epsilon"] is not None:
        dp_config = {
            "epsilon": cfg["dp_epsilon"], "delta": cfg["dp_delta"],
            "max_grad_norm": cfg["dp_max_grad_norm"],
        }

    history: list[dict] = []

    def on_round(metrics: dict) -> None:
        history.append(metrics)
        progress.progress(metrics["round"] / cfg["num_rounds"],
                          text=f"Round {metrics['round']}/{cfg['num_rounds']}")
        render_curves(curves_box, history)
        render_quality(quality_box, history)

    server.run(
        num_rounds=cfg["num_rounds"], local_epochs=cfg["local_epochs"],
        local_lr=cfg["local_lr"], proximal_mu=cfg["proximal_mu"],
        eval_every=1, dp_config=dp_config, round_callback=on_round,
    )

    if cfg["save_json"]:
        out = RUNS_DIR / "ui" / f"{cfg['aggregator']}_seed{cfg['seed']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": {k: v for k, v in cfg.items() if k != "agg_kwargs"},
                   "agg_kwargs": cfg["agg_kwargs"], "dp_config": dp_config,
                   "history": history, "final": history[-1] if history else None}
        out.write_text(json.dumps(payload, indent=2, default=str))
        st.toast(f"Saved: {out.relative_to(ROOT)}")

    return history


# --------------------------------------------------------------------------- #
# Page: Run Experiment
# --------------------------------------------------------------------------- #
def page_run() -> None:
    st.subheader("Run a federated experiment")
    st.caption(
        "Synthetic multi-center medical-imaging playground. Configure in the "
        "sidebar, then **Run**. Curves update live each round."
    )
    cfg = experiment_sidebar()

    summary = (
        f"**{cfg['aggregator']}**"
        + (f" {cfg['agg_kwargs']}" if cfg["agg_kwargs"] else "")
        + f" · {cfg['num_clients']} clients · {cfg['num_rounds']} rounds · "
        f"het={cfg['heterogeneity']}"
        + (f" · noise: client#{cfg['noisy_client']}@{cfg['noisy_rate']}" if cfg["noisy_client"] >= 0 else "")
        + (f" · DP ε={cfg['dp_epsilon']}" if cfg["dp_epsilon"] else "")
        + f" · {cfg['device']}"
    )
    st.info(summary)

    run = st.button("Run experiment", type="primary", use_container_width=True)

    progress = st.empty()
    curves_box = st.empty()
    quality_box = st.empty()

    if run:
        if cfg["dp_epsilon"] is not None and cfg["arch"] == "small_cnn":
            st.warning(
                "DP-SGD works best with the `med_cnn` backbone (GroupNorm + "
                "non-inplace ops). `small_cnn` may error under Opacus."
            )
        try:
            with st.spinner("Training…"):
                history = run_experiment(cfg, curves_box, quality_box, progress)
            st.session_state["run_history"] = history
            st.session_state["run_cfg"] = cfg
            progress.empty()
            st.success("Done.")
        except Exception as e:  # surface training errors in the UI, don't crash
            st.error(f"Run failed: {type(e).__name__}: {e}")
            st.exception(e)
    elif st.session_state.get("run_history"):
        # Re-render last run after an unrelated rerun (e.g. sidebar tweak).
        render_curves(curves_box, st.session_state["run_history"])
        render_quality(quality_box, st.session_state["run_history"])


# --------------------------------------------------------------------------- #
# Page: Browse Saved Results
# --------------------------------------------------------------------------- #
def page_browse() -> None:
    st.subheader("Browse saved results")
    files = find_result_files()
    if not files:
        st.warning("No JSON results found under `runs/`.")
        return

    rel = [str(p.relative_to(ROOT)) for p in files]
    choice = st.selectbox(f"Result file ({len(files)} found)", rel)
    path = ROOT / choice
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        st.error(f"Could not parse {choice}: {e}")
        return

    # Sweep summary file?
    if "summary" in data and "runs" in data:
        st.markdown("**Multi-seed sweep summary**")
        summ = data["summary"]
        cols = st.columns(len(summ) if isinstance(summ, dict) else 1)
        for col, (k, v) in zip(cols, summ.items()):
            if isinstance(v, dict) and "mean" in v:
                col.metric(k, f"{v['mean']:.4f}", f"± {v.get('std', 0):.4f}")
        st.dataframe(pd.DataFrame(data["runs"]), use_container_width=True)
        with st.expander("Raw args"):
            st.json(data.get("args", {}))
        return

    # Standard run file.
    final = data.get("final") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", _fmt(final.get("accuracy")))
    c2.metric("Loss", _fmt(final.get("loss")))
    c3.metric("AUROC", _fmt(final.get("auroc")))
    c4.metric("AUPRC", _fmt(final.get("auprc")))
    c5.metric("ECE", _fmt(final.get("ece")))

    history = data.get("history") or []
    if history:
        df = history_to_df(history)
        left, right = st.columns(2)
        if df["loss"].notna().any():
            left.caption("Loss (lower is better)"); left.line_chart(df[["loss"]])
        if df["accuracy"].notna().any():
            right.caption("Accuracy (higher is better)"); right.line_chart(df[["accuracy"]])
        cal = [c for c in ["auroc", "auprc", "ece"] if df[c].notna().any()]
        if cal:
            st.caption("AUROC / AUPRC (higher is better) · ECE (lower is better)"); st.line_chart(df[cal])
        if history[-1].get("client_quality"):
            render_quality(st.container(), history)

    with st.expander("Run config"):
        st.json(data.get("config", {}))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Federated Learning for Medical Imaging")
st.caption("Heterogeneity-aware & privacy-preserving FL · AHA-Fed, FedAvg, FedProx, FedBN, FedDisco")

tab_run, tab_browse = st.tabs(["Run Experiment", "Browse Saved Results"])
with tab_run:
    page_run()
with tab_browse:
    page_browse()
