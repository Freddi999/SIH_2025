# SIH Project

## Overview
This project implements a SUMO (Simulation of Urban MObility) environment for simulating traffic scenarios. It provides a structured approach to define environments, configurations, and run simulations.

## Project Structure
```
SIH
├── src
│   ├── envs
│   │   ├── __init__.py
│   │   └── sumo_env.py
│   ├── config
│   │   ├── __init__.py
│   │   └── sumo_qmix.yaml
│   ├── main.py
│   └── run.py
├── maps
│   ├── connaught_place.net.xml
│   ├── connaught_place.rou.xml
│   └── connaught_place.sumocfg
├── requirements.txt
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage
1. Configure the SUMO environment settings in `src/config/sumo_qmix.yaml`.
2. Define the road network and routes in the `maps` directory.
3. Run the simulation using the `run.py` script.

```bash
python src/run.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.