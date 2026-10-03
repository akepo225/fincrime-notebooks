"""Synthetic week for Finding Mule Networks in Transaction Graphs.

Article-scoped data, written for this article. Seeded. No real payment, account,
or device. Degree is fixed by the edge list. The Louvain seed is fixed.
A second run on the same library versions writes the same figure bytes and the
same printed degree, betweenness, and community assignment.

The partition and the betweenness order are outputs. This script does not
assert them.
"""

import os
from collections import defaultdict
from pathlib import Path

os.environ["SOURCE_DATE_EPOCH"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

SEED = 42
WINDOW_HOURS = 6
FIGDIR = Path(__file__).resolve().parent / "figures"

PAPER = "#f6f3ec"
INK = "#1c1917"
MUTED = "#57534e"
RULE = "#d6d3d1"
FAINT = "#e7e5e4"
NAVY = "#1e3a5f"
NAVY_FILL = "#e7eef5"
RUST = "#8f3d32"
RUST_FILL = "#f8ebe3"
GREEN = "#3f6212"
GREEN_FILL = "#e7f0e4"
DEVICE_FILL = "#f3efe4"

# id, short label, role, x, y, major
# Coordinates are fixed. Do not switch this to a spring layout.
ACCOUNTS = [
    ("wage_payer", "Wage payer", "wage", 16, 78, True),
    ("employee_3", "E3", "staff", 8, 70, False),
    ("employee_4", "E4", "staff", 20, 70, False),
    ("employee_rent", "Pays rent", "rent_employee", 36, 54, True),
    ("employee_5", "E5", "staff", 8, 54, False),
    ("employee_6", "E6", "staff", 20, 54, False),
    ("employee_7", "E7", "staff", 8, 38, False),
    ("employee_8", "E8", "staff", 20, 38, False),
    ("employee_mule", "Employee-mule", "employee_mule", 52, 62, True),
    ("rent_payee", "Rent", "rent_payee", 36, 26, False),
    ("victim_1", "V1", "victim", 52, 78, False),
    ("victim_2", "V2", "victim", 68, 78, False),
    ("victim_3", "V3", "victim", 84, 78, False),
    ("mule_1", "Pure mule", "mule", 66, 62, True),
    ("mule_2", "Pure mule", "mule", 82, 62, True),
    ("exit", "Exit", "exit", 68, 30, True),
    ("customer_1", "C1", "customer", 98, 82, False),
    ("customer_2", "C2", "customer", 110, 82, False),
    ("customer_3", "C3", "customer", 122, 82, False),
    ("customer_4", "C4", "customer", 98, 76, False),
    ("customer_5", "C5", "customer", 110, 76, False),
    ("customer_6", "C6", "customer", 122, 76, False),
    ("merchant", "Merchant", "merchant", 110, 60, True),
    ("household_1", "Household", "household", 98, 38, True),
    ("household_2", "Household", "household", 122, 38, True),
    ("payee_1", "P1", "payee", 98, 20, False),
    ("payee_2", "P2", "payee", 122, 20, False),
]

# Devices are nodes only under the shared-device rule.
# The rent-paying employee's phone sits in the payroll band so it does not
# read as a second tie on the employee-mule.
DEVICES = [
    ("device_mules", "Shared phone", 68, 46),
    ("device_household", "Shared phone", 110, 38),
    ("device_rent", "Own phone", 36, 40),
]

# src, dst, hour from the start of the synthetic week. One payment per pair.
PAYMENTS = [
    ("wage_payer", "employee_mule", 9),
    ("wage_payer", "employee_rent", 9),
    ("wage_payer", "employee_3", 9),
    ("wage_payer", "employee_4", 9),
    ("wage_payer", "employee_5", 9),
    ("wage_payer", "employee_6", 9),
    ("wage_payer", "employee_7", 9),
    ("wage_payer", "employee_8", 9),
    ("employee_rent", "rent_payee", 14),
    ("victim_1", "employee_mule", 30),
    ("employee_mule", "exit", 33),
    ("victim_2", "mule_1", 40),
    ("mule_1", "exit", 42),
    ("victim_3", "mule_2", 50),
    ("mule_2", "exit", 54),
    ("customer_1", "merchant", 20),
    ("customer_2", "merchant", 22),
    ("customer_3", "merchant", 25),
    ("customer_4", "merchant", 27),
    ("customer_5", "merchant", 60),
    ("customer_6", "merchant", 62),
    ("household_1", "payee_1", 26),
    ("household_2", "payee_2", 28),
]

DEVICE_USE = [
    ("employee_mule", "device_mules"),
    ("mule_1", "device_mules"),
    ("mule_2", "device_mules"),
    ("household_1", "device_household"),
    ("household_2", "device_household"),
    ("employee_rent", "device_rent"),
]

EDGE_RAD = {}
POS = {row[0]: (row[3], row[4]) for row in ACCOUNTS}
POS.update({row[0]: (row[2], row[3]) for row in DEVICES})
LABEL = {row[0]: row[1] for row in ACCOUNTS}
LABEL.update({row[0]: row[1] for row in DEVICES})
ROLE = {row[0]: row[2] for row in ACCOUNTS}
MAJOR = {row[0]: row[5] for row in ACCOUNTS}
LABEL_ABOVE = {
    "wage_payer",
    "victim_1",
    "victim_2",
    "victim_3",
    "customer_1",
    "customer_2",
    "customer_3",
    "customer_4",
    "customer_5",
    "customer_6",
}


def pass_through_edges(payments, window):
    """Keep both legs when an outbound follows an inbound inside the window."""
    inbound = defaultdict(list)
    outbound = defaultdict(list)
    for src, dst, hour in payments:
        inbound[dst].append((hour, src))
        outbound[src].append((hour, dst))

    edges = set()
    for account, arrivals in inbound.items():
        for hour_in, payer in arrivals:
            for hour_out, payee in outbound.get(account, []):
                gap = hour_out - hour_in
                if 0 < gap <= window:
                    edges.add((payer, account))
                    edges.add((account, payee))
    return edges


def accounts_that_pass_through(edges):
    incoming = {dst for _, dst in edges}
    outgoing = {src for src, _ in edges}
    return incoming & outgoing


def style_for(node_id):
    role = ROLE.get(node_id, "device")
    if role == "employee_mule":
        return RUST_FILL, NAVY, 1.8
    if role in {"mule"}:
        return RUST_FILL, RUST, 1.3
    if role == "exit":
        return "#e7e5e4", INK, 1.4
    if role == "merchant":
        return GREEN_FILL, GREEN, 1.3
    if role in {"wage", "rent_employee", "household"}:
        return NAVY_FILL, NAVY, 1.3
    if role == "device":
        return DEVICE_FILL, INK, 1.2
    return FAINT, MUTED, 0.8


def draw_edge(ax, src, dst, directed):
    x1, y1 = POS[src]
    x2, y2 = POS[dst]
    rad = EDGE_RAD.get((src, dst), 0.0)
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>" if directed else "-",
        mutation_scale=8,
        linewidth=0.9,
        color=INK,
        shrinkA=9,
        shrinkB=9,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)


def draw_node(ax, node_id, active):
    x, y = POS[node_id]
    face, edge, width = style_for(node_id)
    major = MAJOR.get(node_id, node_id.startswith("device"))
    radius = 2.6 if major else 1.7
    alpha = 1.0 if active else 0.28
    if node_id.startswith("device"):
        box = FancyBboxPatch(
            (x - 3.4, y - 2.2),
            6.8,
            4.4,
            boxstyle="round,pad=0.15,rounding_size=0.4",
            facecolor=face,
            edgecolor=edge,
            linewidth=width,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(box)
    else:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=face,
                edgecolor=edge,
                linewidth=width,
                alpha=alpha,
                zorder=2,
            )
        )
    if active or major:
        above = node_id in LABEL_ABOVE
        ax.text(
            x,
            y + radius + 1.2 if above else y - radius - 0.8,
            LABEL[node_id],
            ha="center",
            va="bottom" if above else "top",
            fontsize=7.0 if major else 6.0,
            color=INK if active else MUTED,
            alpha=1.0 if active else 0.55,
            zorder=3,
        )


def draw_regions(ax):
    bands = [
        (1, 8, 44, 86, "Payroll"),
        (48, 8, 40, 86, "Mule payments"),
        (90, 8, 36, 86, "Merchant and household"),
    ]
    for x, y, w, h, name in bands:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="#fbfaf6",
                edgecolor=RULE,
                linewidth=0.6,
                zorder=0,
            )
        )
        ax.text(
            x + 1.5,
            y + h - 2.0,
            name,
            fontsize=8,
            color=MUTED,
            ha="left",
            va="top",
        )


def draw_panel(ax, edges, title, directed, include_devices):
    ax.set_xlim(0, 128)
    ax.set_ylim(4, 100)
    ax.axis("off")
    ax.set_facecolor(PAPER)
    draw_regions(ax)
    ax.text(1.5, 97.5, title, fontsize=11, color=INK, ha="left", va="top")

    active = {node for edge in edges for node in edge}
    if include_devices:
        for device in (row[0] for row in DEVICES):
            active.add(device)

    for src, dst in edges:
        draw_edge(ax, src, dst, directed)

    for node_id in POS:
        is_device = node_id.startswith("device")
        if is_device and not include_devices:
            continue
        draw_node(ax, node_id, node_id in active)


def save_figure(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(
        path,
        dpi=150,
        facecolor=PAPER,
        edgecolor="none",
        metadata={"Software": "fincrime-notebooks", "Creation Time": None},
    )
    plt.close(fig)
    return path


def figure_three_drawings(ledger_edges, through_edges, device_edges):
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 15.6))
    fig.patch.set_facecolor(PAPER)
    fig.suptitle(
        "Three drawings of one week",
        fontsize=15,
        color=INK,
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.955,
        "Synthetic accounts. Same week, three link rules. No algorithm output.",
        fontsize=9,
        color=MUTED,
        ha="left",
    )
    draw_panel(axes[0], ledger_edges, "Ledger rule: any payment in the extract", True, False)
    draw_panel(
        axes[1],
        through_edges,
        "Pass-through rule: money out after money in, inside 6 hours",
        True,
        False,
    )
    draw_panel(
        axes[2],
        device_edges,
        "Shared-device rule: accounts used from the same device",
        False,
        True,
    )
    fig.text(
        0.06,
        0.012,
        "Navy: wage payer, the employee who pays rent, household.  "
        "Rust: pure mule.  Navy ring: employee who is also a mule.  "
        "Green: merchant.  Gray: other accounts.  Square: device.\n"
        "Arrows follow the payment. Device lines do not. "
        "E3 to E8 are the other employees. V1 to V3 are victim payments. "
        "C1 to C6 are customers. P1 and P2 are ordinary payees.",
        fontsize=7.4,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.055, hspace=0.08)
    return save_figure(fig, "fig1_three_drawings.png")


def figure_degree(degree):
    rows = [
        ("Wage payer", degree["wage_payer"], NAVY),
        ("Merchant", degree["merchant"], GREEN),
        ("Exit", degree["exit"], INK),
        ("Employee-mule", degree["employee_mule"], RUST),
        ("Pure mule", degree["mule_1"], RUST),
        ("Pure mule", degree["mule_2"], RUST),
        ("Employee who pays rent", degree["employee_rent"], NAVY),
        ("Household", degree["household_1"], NAVY),
        ("Household", degree["household_2"], NAVY),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    y = list(range(len(rows) - 1, -1, -1))
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]
    ax.barh(y, values, color=colors, height=0.62, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9, color=INK)
    ax.set_xlabel("Unique counterparties", fontsize=9, color=INK)
    ax.set_xlim(0, 10)
    ax.set_title("Ledger degree by role", loc="left", fontsize=14, color=INK, pad=18)
    ax.text(
        0.0,
        1.01,
        "Synthetic design. The tie at 2 is the point. Not a finding about real networks.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    for yi, value in zip(y, values):
        ax.text(value + 0.15, yi, str(value), va="center", ha="left", fontsize=8.5, color=INK)
    ax.axvline(2, color=RULE, linewidth=0.8, zorder=1)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    return save_figure(fig, "fig2_ledger_degree.png")


def community_lines(communities):
    ordered = sorted(tuple(sorted(community)) for community in communities)
    lines = []
    home = None
    for index, community in enumerate(ordered, start=1):
        line = f"community {index}: {', '.join(community)}"
        lines.append(line)
        if "employee_mule" in community:
            home = line
    return lines, home


def main():
    account_ids = [row[0] for row in ACCOUNTS]
    if len(account_ids) != len(set(account_ids)):
        raise SystemExit("duplicate account id")
    if len(account_ids) != 27:
        raise SystemExit(f"expected 27 accounts, found {len(account_ids)}")

    ledger = nx.Graph()
    ledger.add_nodes_from(account_ids)
    for src, dst, _hour in PAYMENTS:
        if ledger.has_edge(src, dst):
            raise SystemExit(f"pair paid more than once: {src} {dst}")
        ledger.add_edge(src, dst)

    degree = dict(ledger.degree())
    expected = {
        "wage_payer": 8,
        "merchant": 6,
        "exit": 3,
        "employee_mule": 3,
        "mule_1": 2,
        "mule_2": 2,
        "employee_rent": 2,
        "household_1": 1,
        "household_2": 1,
    }
    for node_id, count in expected.items():
        if degree[node_id] != count:
            raise SystemExit(f"degree {node_id} is {degree[node_id]}, expected {count}")

    through = pass_through_edges(PAYMENTS, WINDOW_HOURS)
    fired = accounts_that_pass_through(through)
    if fired != {"employee_mule", "mule_1", "mule_2", "employee_rent"}:
        raise SystemExit(f"pass-through fired on {sorted(fired)}")

    device_graph = nx.Graph()
    for account, device in DEVICE_USE:
        device_graph.add_edge(account, device)
    shared = sorted(
        device
        for device in {row[0] for row in DEVICES}
        if device_graph.degree(device) >= 2
    )
    if shared != ["device_household", "device_mules"]:
        raise SystemExit(f"shared devices: {shared}")

    digraph = nx.DiGraph()
    digraph.add_edges_from(sorted(through))
    betweenness = nx.betweenness_centrality(digraph, normalized=True, endpoints=False)
    ranked = sorted(betweenness.items(), key=lambda item: (-item[1], item[0]))
    top_score = ranked[0][1]
    highest = [name for name, score in ranked if score == top_score]

    communities = nx.community.louvain_communities(
        ledger, weight="weight", resolution=1, seed=SEED
    )
    lines, home = community_lines(communities)
    if home is None:
        raise SystemExit("employee_mule missing from the partition")

    print("synthetic week")
    print(f"window_hours {WINDOW_HOURS}")
    print(f"louvain_seed {SEED}")
    print(f"ledger_accounts {ledger.number_of_nodes()}")
    print(f"ledger_edges {ledger.number_of_edges()}")
    print("degree is unique counterparties on the undirected ledger graph")
    for node_id in sorted(degree, key=lambda name: (-degree[name], name)):
        print(f"degree {node_id} {degree[node_id]}")
    print("pass-through edges, directed")
    for src, dst in sorted(through):
        print(f"pass_through {src} {dst}")
    print("pass-through fires on")
    for name in sorted(fired):
        print(f"fires {name}")
    print(
        "betweenness directed on the pass-through graph, normalized, endpoints excluded"
    )
    print(f"betweenness_nodes {digraph.number_of_nodes()}")
    for name, score in ranked:
        print(f"betweenness {name} {score:.6f}")
    print("betweenness_highest " + ",".join(highest))
    print("louvain partition of the undirected ledger graph")
    print(f"communities {len(lines)}")
    for line in lines:
        print(line)
    print(f"employee_mule_community {home}")

    directed_ledger = sorted((src, dst) for src, dst, _hour in PAYMENTS)
    if ledger.number_of_edges() != len(directed_ledger):
        raise SystemExit("directed and undirected edge counts diverged")
    figure_three_drawings(directed_ledger, sorted(through), sorted(DEVICE_USE))
    figure_degree(degree)


if __name__ == "__main__":
    main()
