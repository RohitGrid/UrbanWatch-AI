<img src="assets/logo.jpg" alt="UrbanWatch AI" width="100%" />

### AI-driven parking intelligence that turns raw violation data into targeted enforcement decisions

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Folium](https://img.shields.io/badge/Folium-Geospatial-77B829?style=flat-square&logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Built%20for-%5BHackathon%20Name%5D-blueviolet?style=flat-square)](#)

</div>

---
<p align="center">
  <a href="https://skill-map-nine.vercel.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Visit%20Now-brightgreen?style=for-the-badge" />
  </a>
</p>

## Problem Statement

<table>
<tr>

<td width="140" align="center" valign="middle">

<img src="assets/vis.png" width="120">

</td>

<td valign="top">

### Theme: Poor Visibility on Parking-Induced Congestion

Illegal and spillover on-street parking near commercial strips, metro stations, and event venues regularly chokes carriageways and intersections — but enforcement today is patrol-based and reactive. There's no map connecting where violations happen to how much congestion they actually cause, which makes it hard to know where to send limited officers and tow trucks first.

**Our question:** how can AI-driven parking intelligence detect illegal-parking hotspots and quantify their traffic-flow impact, so enforcement can be targeted instead of random?

</td>

</tr>
</table>

## What We Built

**UrbanWatch AI** is a single-page enforcement dashboard backed by three independent data-science components, all trained on real Bengaluru Traffic Police parking-violation records (`police.csv`, ~298K rows). Together they answer three different questions:

| Question | Component | Output |
|---|---|---|
| *Where, geographically, is parking congestion worst?* | **Hotspot Map Engine** | Interactive heatmap (`parking_hotspot_map.html`) |
| *Which named junctions need enforcement first, and why?* | **Junction CSI Scoring Engine** | Top-20 ranked junctions with a 0–100 Congestion Severity Index |
| *When will a given zone be high-risk, and how severe is a specific violation report?* | **Hotspot Prediction Model** | Day/hour risk grid (`hotspot_predictions.json`) consumed live by the dashboard |

All three feed into `app.html`, a single-file dashboard with 9 views: Command Overview, Hotspot Analysis, Heatmap, Congestion Intelligence (CSI), Enforcement, Hotspot Predictor, Violation Severity Analyser, Zone Report, and a What-If simulator for testing enforcement scenarios.

## Screenshots

<table>
<tr>
<td width="50%">

**Hotspot Analysis — Top 20 CSI-ranked junctions**
<img src="assets/screenshot-hotspot-analysis.png" alt="Hotspot Analysis" width="100%"/>

</td>
<td width="50%">

**Hotspot Predictor — day/hour risk forecast**
<img src="assets/screenshot-hotspot-predictor.png" alt="Hotspot Predictor" width="100%"/>

</td>
</tr>
<tr>
<td width="50%">

**Congestion Severity Index breakdown**
<img src="assets/screenshot-csi-breakdown.png" alt="CSI Breakdown" width="100%"/>

</td>
<td width="50%">

**Violation Severity Analyser**
<img src="assets/screenshot-violation-analyser.png" alt="Violation Analyser" width="100%"/>

</td>
</tr>
<tr>
<td width="50%">

**Enforcement recommendations**
<img src="assets/screenshot-enforcement.png" alt="Enforcement" width="100%"/>

</td>
<td width="50%">

**Command overview dashboard**
<img src="assets/screenshot-dashboard.png" alt="Dashboard" width="100%"/>

</td>
</tr>
</table>

<p align="center">
  <img src="https://img.shields.io/badge/Violation count-298K-3B82F6?style=for-the-badge">
  <img src="https://img.shields.io/badge/Junctions-20-22C55E?style=for-the-badge">
  <img src="https://img.shields.io/badge/Critical_Zones-4-F59E0B?style=for-the-badge">
  <img src="https://img.shields.io/badge/Random_Forest-300_Trees-EF4444?style=for-the-badge">
</p>

## Architecture

```mermaid
flowchart TD
    A[("police.csv<br/>~298,450 parking-violation records")]

    A --> B["Hotspot Map Engine<br/>notebooks/hotspot_map_eda.ipynb"]
    A --> C["Junction CSI Scoring Engine<br/>scripts/junction_factors.py"]
    A --> D["Hotspot Prediction Model<br/>notebooks/hotspot_predictor_training.ipynb"]

    B -->|"Grid-cell clustering<br/>+ folium HeatMap"| E["parking_hotspot_map.html"]
    C -->|"4-signal weighted CSI<br/>per named junction"| F["junction_csi_clean.csv<br/>(Top 20 ranked junctions)"]
    D -->|"RandomForestRegressor<br/>300 trees, cyclical time features"| G["hotspot_predictions.json<br/>20 zones × 7 days × 24 hours"]

    E --> H["UrbanWatch AI Dashboard<br/>app.html"]
    F --> H
    G --> H

    H --> I["Heatmap tab"]
    H --> J["Hotspot Analysis ·<br/>Enforcement · CSI tabs"]
    H --> K["Hotspot Predictor ·<br/>Violation Analyser tabs"]
```

> **Design note:** the three components select their "top junctions" independently (grid-cell bins vs. cleaned junction names vs. raw violation-count ranking), so their outputs are deliberately *complementary views* of the same problem rather than one merged ranking. The CSI engine's named-junction ranking feeds the Hotspot Analysis/Enforcement/CSI tabs; the RandomForest's own top-20-by-volume selection feeds the Predictor/Analyser tabs.

## The three models

### 1. Hotspot Map Engine — `notebooks/hotspot_map_eda.ipynb`
<img src="assets/heatmap.png" class="engine-image">
Exploratory analysis plus an unsupervised geo-clustering pass:

- Bins every GPS-tagged, approved violation into a **0.001°×0.001° grid cell** (≈100m) and keeps cells with ≥10 violations.
- Scores each cell with a 3-signal **Congestion Impact Score**:

  `score = 0.5 × violation density + 0.3 × junction proximity + 0.2 × peak-hour share`

- Renders an interactive **Folium heatmap** (dark-matter basemap) with a `HeatMap` density layer, ranked `CircleMarker`s for the top hotspot cells, and a `MarkerCluster` of the busiest named junctions.
- Outputs: `eda_overview.png` (4-panel EDA), `hotspot_clusters.csv`, **`parking_hotspot_map.html`** (the file the dashboard's Heatmap tab embeds in an iframe).

### 2. Junction CSI Scoring Engine — `scripts/junction_factors.py`
<img src="assets/CSI Engine.png" class="engine-image">
Turns raw violations into a single, explainable **Congestion Severity Index (0–100)** per named junction:

| Signal | Weight | What it measures |
|---|---|---|
| Violation Density | 40% | Total approved violations at the junction |
| Repeat Violation Frequency | 25% | Violations per unique vehicle (`violations ÷ unique vehicles`) |
| Peak-Hour Violation Rate | 20% | Share of violations in 8–11 AM / 5–9 PM windows |
| Road/Parking-Type Weight | 15% | Severity-weighted mix of violation types at that junction |

Each raw signal is **robust min-max scaled** (5th–95th percentile clipping, so a handful of outlier junctions can't distort the 0–100 scale) before the weighted sum is taken. Parking-type severity weights:

| Violation type | Weight |
|---|---|
| Parking in a Main Road | 10 |
| Parking Near Road Crossing | 10 |
| Double Parking | 9 |
| Parking on Footpath | 8 |
| Wrong Parking | 6 |
| No Parking | 5 |

Output: `junction_csi_clean.csv` — the ranked top-20 list (with all 4 component scores + final CSI) that powers the **Hotspot Analysis**, **Enforcement**, and **Congestion Intelligence** tabs.

### 3. Hotspot Prediction Model — `notebooks/hotspot_predictor_training.ipynb`
<img src="assets/rf_confidence_scatter.png" class="scatter-image">
A supervised model that answers *"how many violations should I expect at zone X, on day D, at hour H?"*

- Selects the **top 20 zones by raw violation volume**, then builds a fully-enumerated training table: 20 zones × 7 days × 24 hours = 3,360 rows, with cyclical `sin`/`cos` encodings for hour-of-day and day-of-week so the model understands that 23:00 and 00:00 are adjacent.
- Trains a **`RandomForestRegressor`** (300 trees, max depth 12, min 2 samples/leaf) on **123,734 records** to predict violation count per zone/day/hour cell.
- Re-predicts every cell using each tree individually to derive a **confidence score** from the ensemble's prediction variance (low variance → high confidence, scaled to 65–97%), and labels each cell **Low / Medium / High** based on its intensity relative to that zone's peak hour.
- Serializes everything into **`hotspot_predictions.json`** — a fully-enumerated O(1) lookup table, so the live dashboard never re-runs the model; it just looks up `grid[zone][day][hour]`.

This single JSON file powers **both** the Hotspot Predictor (pick a day/hour, see ranked risk zones) and the Violation Severity Analyser (enter a violation report, get the nearest zone's risk + a severity nudge based on violation type, vehicle type, and junction proximity).

## Dataset

- **Source:** Bengaluru Traffic Police parking-violation records (`police.csv`) — `[insert dataset link here]`
- **Scale:** ~298,450 total records; 123,734 of those fall within the 20 highest-volume zones used for model training
- **Key fields used:** `created_datetime`, `latitude`/`longitude`, `location`, `junction_name`, `violation_type`, `vehicle_type`, `vehicle_number`, `validation_status`, `validation_timestamp`

`police.csv` is not committed to this repo (large, and may be access-restricted by the hackathon organizers) — see [`data/README.md`](data/README.md) for where to place it.

## Project structure

```
UrbanWatch-ai/
├── README.md
├── LICENSE
├── .gitignore
├── app.html                        # the dashboard — open this in a browser to run it
├── hotspot_predictions.json        # Model 3 output — MUST stay next to app.html (fetch('./hotspot_predictions.json'))
├── parking_hotspot_map.html        # Model 1 output — MUST stay next to app.html (iframe target) — add after running the notebook
├── notebooks/
│   ├── hotspot_map_eda.ipynb           # Model 1: EDA + geo-clustering + Folium heatmap
│   └── hotspot_predictor_training.ipynb # Model 3: RandomForestRegressor training
├── scripts/
│   └── junction_factors.py             # Model 2: junction CSI scoring engine
├── data/
│   ├── README.md                       # where to get police.csv
│   └── police.csv                      # (gitignored — add your own copy)
└── assets/
    ├── banner.svg / banner.png
    └── screenshot-*.png
```

> ⚠️ **`hotspot_predictions.json`** and **`parking_hotspot_map.html`** must live in the **same folder as `app.html`** — the dashboard loads both with relative `fetch()`/`iframe` paths.

## Getting started

### Run the dashboard (what judges will actually look at)

```bash
git clone https://github.com/<your-org>/UrbanWatch-ai.git
cd UrbanWatch-ai
python -m http.server 8000
# open http://localhost:8000/app.html
```

A local server (rather than opening the file directly) avoids browser CORS restrictions on the `fetch('./hotspot_predictions.json')` call.

### Reproduce the models from scratch

```bash
pip install pandas numpy scikit-learn folium matplotlib seaborn jupyter

# 1. Place police.csv in the repo root (see data/README.md)

# 2. Junction CSI scoring (Model 2)
python scripts/junction_factors.py
# → junction_csi_clean.csv

# 3. Hotspot map (Model 1)
jupyter nbconvert --to notebook --execute notebooks/hotspot_map_eda.ipynb
# → parking_hotspot_map.html  (copy to repo root)

# 4. Hotspot prediction model (Model 3)
jupyter nbconvert --to notebook --execute notebooks/hotspot_predictor_training.ipynb
# → hotspot_predictions.json  (copy to repo root)
```

## Tech stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (`RandomForestRegressor`), `DBSCAN`-ready clustering |
| Geospatial | folium, Leaflet (via folium), `HeatMap` / `MarkerCluster` plugins |
| Visualization (EDA) | matplotlib, seaborn |
| Frontend | vanilla HTML/CSS/JS, Chart.js (radar/bar/line/scatter charts) |
| Data interchange | static JSON lookup table |

## Roadmap

- [ ] Unify the three top-20 junction selections behind one canonical zone list
- [ ] Replace the static `hotspot_predictions.json` lookup with a live inference endpoint
- [ ] Add officer GPS check-in to close the loop between recommendation and actual response time
- [ ] Auto-refresh CSI scores on a rolling window instead of a static snapshot
- [ ] Mobile-friendly officer view for in-field hotspot lookup

## Team

| Name | Role | GitHub |
|---|---|---|
| _[Your Name]_ | _[e.g. Team Lead / ML Engineer]_ | [@handle](https://github.com/) |
| _[Your Name]_ | _[e.g. Data Engineer]_ | [@handle](https://github.com/) |
| _[Your Name]_ | _[e.g. Frontend Developer]_ | [@handle](https://github.com/) |
| _[Your Name]_ | _[e.g. Geospatial/ML]_ | [@handle](https://github.com/) |

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Built for **[Hackathon Name]** — Theme: *Poor Visibility on Parking-Induced Congestion*
- Dataset: Bengaluru Traffic Police parking-violation records
