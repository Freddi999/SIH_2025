# SUMO + QMIX (PyMARL2) Setup

This project integrates the Connaught Place SUMO traffic network with the PyMARL2 QMIX framework.

## 📂 Files Overview
- `maps/connaught_place.net.xml` → The network file (your uploaded Connaught Place map).
- `maps/connaught_place.rou.xml` → Generated routes (cars, buses, bikes, trucks with flows).
- `maps/connaught_place.sumocfg` → SUMO config file linking `.net.xml` and `.rou.xml`.
- `src/envs/SUMOEnv.py` → Custom SUMO environment for PyMARL2.
- `src/config/envs/sumo.yaml` → Environment config file for PyMARL2.

## ⚙️ Installation
1. Install SUMO and ensure `sumo` and `sumo-gui` are available in your PATH.
2. Create a Python virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt
   pip install traci sumolib
   ```

## ▶️ Running the Simulation (Baseline Test)
Before training, test if SUMO runs correctly:
```bash
sumo-gui -c maps/connaught_place.sumocfg
```
This should open SUMO GUI and vehicles should flow.

## 🚀 Training QMIX with PyMARL2
From the project root run:
```bash
python src/main.py --config=qmix --env-config=sumo
```

## 📊 Logs & Models
- CSV logs will be saved under `results/` (episode rewards, waiting times, queues).
- Models are checkpointed automatically by PyMARL2.

## 🔧 Notes
- Episode length = 3600s (1 hour of simulated traffic).
- Decision interval = 5 seconds per QMIX step.
- Reward = Negative total waiting time (global).

## ✅ Next Steps
- Adjust traffic demand in `connaught_place.rou.xml` for experiments.
- Add weather or demand factors later for regressor integration.
