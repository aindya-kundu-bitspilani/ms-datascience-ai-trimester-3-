"""
generate_diagram.py
--------------------
Generates diagrams/architecture.png -- the end-to-end system diagram
required by the assignment (data sources -> pipelines -> features ->
training -> model registry -> serving -> monitoring -> retraining).

Uses Graphviz with an explicit 3-row grid layout (rank=same per row)
for a clean, professional, roughly-16:9 diagram rather than one long
horizontal strip.

Run:
    python diagrams/generate_diagram.py
"""

import graphviz

COLORS = {
    "data": "#DCE8FC",
    "pipeline": "#FDEBC8",
    "model": "#DDF2DE",
    "serve": "#F6D8D8",
    "monitor": "#EAE0F8",
}
FONT = "Helvetica"


def add_node(g, name, label, fill):
    g.node(
        name, label,
        shape="box", style="filled,rounded", fillcolor=fill,
        fontname=FONT, fontsize="12", color="#4A4A4A", penwidth="1.3",
        margin="0.22,0.16", width="2.1", height="0.9",
    )


def main():
    g = graphviz.Digraph(
        "architecture",
        format="png",
        graph_attr={
            "rankdir": "TB",
            "splines": "spline",
            "nodesep": "0.7",
            "ranksep": "1.0",
            "fontname": FONT,
            "label": "Mini Production ML System \u2014 Churn Risk Scoring: Architecture Overview",
            "labelloc": "t",
            "fontsize": "22",
            "pad": "0.5",
            "bgcolor": "white",
            "dpi": "160",
        },
        node_attr={"fontname": FONT},
        edge_attr={"fontname": FONT, "fontsize": "10", "color": "#4A4A4A", "arrowsize": "0.8"},
    )

    # ---------- Row 1: Data & Features ----------
    add_node(g, "source", "Open-Source Dataset\n(Kaggle: Telco Customer\nChurn, 7,043 rows)", COLORS["data"])
    add_node(g, "ingest", "Ingestion Script\nsrc/data_ingestion.py\nvalidate \u00b7 append \u00b7 log", COLORS["pipeline"])
    add_node(g, "table", "Processed Training Table\ndata/processed/\ntraining_data.csv", COLORS["data"])
    add_node(g, "features", "Shared Feature Module\nsrc/features.py\n(identical code: train + serve)", COLORS["pipeline"])
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("source"); s.node("ingest"); s.node("table"); s.node("features")
    g.edge("source", "ingest")
    g.edge("ingest", "table")
    g.edge("table", "features")

    # ---------- Row 2: Training & Registry & Serving ----------
    add_node(g, "train", "Training Pipeline\nsrc/train.py\nbaseline vs. candidate", COLORS["model"])
    add_node(g, "eval", "Offline Evaluation +\nPromotion Guardrail\nartifacts/eval/*.json", COLORS["model"])
    add_node(g, "registry", "Model Registry (simple)\nmodels/production_model.pkl\n+ model_version.json", COLORS["model"])
    add_node(g, "api", "Serving API (FastAPI)\napi/main.py \u2192 POST /predict\nonline request-response", COLORS["serve"])
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("train"); s.node("eval"); s.node("registry"); s.node("api")
    g.edge("features", "train")
    g.edge("train", "eval")
    g.edge("eval", "registry")
    g.edge("registry", "api")

    # ---------- Row 3: Client & Monitoring & Retraining ----------
    add_node(g, "trigger", "Retrain Trigger + Execution\nsrc/retrain_trigger.py (decide)\nsrc/retrain.py (retrain \u00b7 promote \u00b7 archive)", COLORS["monitor"])
    add_node(g, "monitor", "Monitoring\nsrc/monitoring.py\ndata quality + drift checks", COLORS["monitor"])
    g.node("spacer", label="", style="invis", shape="point", width="0")
    add_node(g, "client", "Client / App\nsends JSON request,\nreceives churn_probability", COLORS["serve"])
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("trigger"); s.node("monitor"); s.node("spacer"); s.node("client")

    g.edge("monitor", "trigger")
    g.edge("api", "client")
    g.edge("table", "monitor", style="dashed", xlabel="recent batch  ")
    g.edge("client", "monitor", style="dashed", xlabel="requests  ", constraint="false")
    g.edge("trigger", "train", style="dashed", color="#B33A3A", fontcolor="#B33A3A",
           xlabel="triggers retraining  ", constraint="false")

    # Invisible ordering edges to keep row order stable
    g.edge("source", "trigger", style="invis")
    g.edge("table", "spacer", style="invis")

    g.render("diagrams/architecture", cleanup=True)
    print("Saved diagrams/architecture.png")


if __name__ == "__main__":
    main()
