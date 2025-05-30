# frozen-ground-fem

[![License](https://img.shields.io/github/license/annapekinasova/frozen-ground-fem)](LICENSE)

## Overview

**frozen-ground-fem** is a Python package for advanced, multiphysics simulation of frozen ground processes using the Finite Element Method (FEM). Designed for research and engineering applications in geotechnics and geosciences, it enables high-fidelity modelling of heat transfer, water migration, phase change, and large-strain consolidation phenomena in freezing and thawing soils.

This branch (`thesis`) reflects active research code supporting thesis work, including detailed implementations for thermal, consolidation, and fully coupled thermo-hydro-mechanical (THM) processes in 1D soil columns. The package is robust, modular, and extensible, making it suitable for both academic research and practical engineering studies involving permafrost, seasonal frost, and related scenarios of ground freezing and thawing.

> **Note:** For the most stable release, see the `main` branch. This branch may include features under development.

---

## Table of Contents

- [Purpose and Scope](#purpose-and-scope)
- [Motivation and Significance](#motivation-and-significance)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example Scripts](#example-scripts)
- [Source Code Details](#source-code-details)
  - [thermal.py](#thermalpy)
  - [consolidation.py](#consolidationpy)
  - [coupled.py](#coupledpy)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

---

## Purpose and Scope

The **frozen-ground-fem** package is developed to:

- Accurately simulate the complex interactions in frozen or thawing ground, including temperature evolution, water flow, phase change (ice/water), and soil deformation.
- Support advanced research in geotechnical and environmental engineering, permafrost science, and climate studies.
- Provide a transparent, extensible, and well-documented framework for implementing and testing new models or methods for frozen ground multi-physics.
 
---

## Motivation and Significance

Frozen ground dynamics play a crucial role in many engineering and environmental applications, especially in cold regions. Thawing permafrost, ground subsidence, and freeze-thaw cycles can significantly impact infrastructure, ecosystems, and climate feedbacks. This software aims to provide an open-source, extensible platform for simulating and analyzing such processes, supporting both research and practical applications.

- Understand the behaviour of soils under freezing and thawing conditions
- Model the coupled thermal and mechanical processes in permafrost and seasonally frozen ground
- Aid in engineering design and risk assessments in cold regions

You can access the repository at:  
[https://github.com/annapekinasova/frozen-ground-fem.git](https://github.com/annapekinasova/frozen-ground-fem.git)

---

## Features

- **Thermal Analysis:** Heat transfer in soils, including phase change (freezing/thawing), latent heat effects, and temperature-dependent material properties.
- **Large Strain Consolidation:** Simulation of soil consolidation under loading, accounting for large deformations, evolving void ratio, and changes in hydraulic conductivity.
- **Fully Coupled Thermo-Hydro-Mechanical (THM) Modeling:** Simultaneous solution of heat, water, and deformation processes in a unified FEM framework.
- **Flexible Mesh and Boundary Condition Handling:** Easily define custom meshes, initial/boundary conditions, and time-dependent loading.
- **Extensive Example Scripts:** Ready-to-run examples for benchmarking, laboratory validation, and convergence studies.
- **Modular, Extensible Design:** Easily extend or customize models, materials, and solvers.
- **Reproducibility:** Example data and scripts for all major features.

---

## Installation

### Prerequisites

- Python 3.10+
- Recommended: Use a virtual environment

### Quick Install

```bash
git clone --branch thesis https://github.com/annapekinasova/frozen-ground-fem.git
cd frozen-ground-fem
pip install -r requirements.txt
```

#### Development Install

```bash
pip install -r requirements-dev.txt
pip install -e .
# or, for full test environments:
tox
```

---

## Usage

### Running an Example

Example scripts are provided in the [`examples/`](examples) directory. For instance, to run a coupled consolidation and thermal simulation:

```bash
python examples/coupled_example.py
```

Scripts are extensively commented and refer to associated input files (CSV/BAT) for parameters and validation data.

### As a Library

You may also use the package as a Python library:

```python
from frozen_ground_fem.thermal import ThermalAnalysis1D
# see docstrings for full API usage
```

---

## Project Structure

```
frozen-ground-fem/
├── .github/             # GitHub workflows and issue templates
├── examples/            # Example simulation scripts (see below)
├── src/
│   └── frozen_ground_fem/
│       ├── thermal.py           # Thermal FEM implementation
│       ├── consolidation.py     # Large strain consolidation FEM
│       ├── coupled.py           # Coupled thermal-consolidation FEM
│       └── ...                  # Supporting modules (geometry, materials, etc.)
├── tests/               # Unit and integration tests
├── requirements.txt     # Runtime dependencies
├── requirements-dev.txt # Developer dependencies
├── pyproject.toml       # Build system configuration
├── tox.ini              # Tox configuration for automated testing
├── LICENSE
└── README.md
```

> For a complete list of files, see the [thesis branch directory](https://github.com/annapekinasova/frozen-ground-fem/tree/thesis).

---

## Example Scripts

### Notable Examples

- [`thermal_freeze_thaw_benchmark.py`](examples/thermal_freeze_thaw_benchmark.py): Simulates freezing/thawing front in a soil column, benchmarks against analytical/laboratory results.
- [`consolidation_benchmark.py`](examples/consolidation_benchmark.py): Validates large strain consolidation solver with standard benchmarks.
- [`consolidation_static.py`](examples/consolidation_static.py): Static consolidation example, useful for debugging and validation.
- [`coupled_example.py`](examples/coupled_example.py): Minimal example of the fully coupled THM solver.
- [`coupled_freezing_front_lab_benchmark.py`](examples/coupled_freezing_front_lab_benchmark.py): Laboratory-based benchmark for coupled thermal-hydraulic modeling of a freezing front.
- [`coupled_thaw_consolidation_lab_benchmark.py`](examples/coupled_thaw_consolidation_lab_benchmark.py): Validation of the coupled model with thawing/consolidation laboratory data.
- [`stiffness_mass_matrices.py`](examples/stiffness_mass_matrices.py): Visualize stiffness/mass matrix assembly.
- [`cubic_shape_plot.py`](examples/cubic_shape_plot.py): Plotting of high-order FEM shape functions.

> **Note:** Only a portion of example scripts are listed here due to API limits. [View the complete examples directory](https://github.com/annapekinasova/frozen-ground-fem/tree/thesis/examples) for more.

Each example script is self-contained and includes detailed comments, references to input files (such as `.csv` for parameters or validation data), and instructions for reproducing published results.

---

## Source Code Details

### `thermal.py`

**Purpose:**  
Implements the finite element method for 1D transient heat transfer in freezing/thawing soils, including phase change.

**Key Classes:**
- `ThermalElement1D`: Computes element-level heat flow (conductivity) and heat storage matrices, handles phase change via latent heat and degree of saturation models.
- `ThermalBoundary1D`: Encapsulates Dirichlet (temperature), Neumann (heat flux), and gradient boundary conditions. Supports time-dependent (function) boundary values.
- `ThermalAnalysis1D`: High-level manager for simulation: mesh and boundary definition, time stepping, global matrix assembly, iterative solution, and updating all states (temperature, heat flux, phase change).

**Notable Features:**
- Crank-Nicolson implicit time integration (with adjustable implicitness).
- Latent heat handled via enthalpy or degree-of-saturation formulations.
- Modular: can be extended to higher dimensions or more complex couplings.

### `consolidation.py`

**Purpose:**  
Implements 1D large strain consolidation of soils, accounting for evolving void ratio, permeability, and effective stress.

**Key Classes:**
- `ConsolidationElement1D`: Computes element stiffness and mass matrices for large strain consolidation. Integrates non-linear soil mechanical behavior and hydraulic conductivity.
- `ConsolidationBoundary1D`: Handles boundary conditions for void ratio and water flux, including time-dependent values.
- `ConsolidationAnalysis1D`: Sets up consolidation problems, manages the mesh, global matrix assembly, time stepping, iterative correction, and computes settlement/deformation.

**Notable Features:**
- Large strain (nonlinear) consolidation formulation.
- Full tracking of void ratio, effective stress, and hydraulic properties at all integration points.
- Supports laboratory and field benchmarks.

### `coupled.py`

**Purpose:**  
Implements fully coupled 1D thermo-hydro-mechanical (THM) elements by inheriting features from both `ThermalElement1D` and `ConsolidationElement1D`.

**Key Classes:**
- `CoupledElement1D`: Hybrid class enabling simultaneous solution of heat transfer, water migration, and large-strain deformation. Leverages both parent implementations for element-level assembly.

**Notable Features:**
- Enables true THM simulations for freezing/thawing soils.
- Easily extendable for more complex coupling (e.g., 2D/3D, advanced material models).

---

## Testing

Run the test suite with:

```bash
pytest tests/
```
or for full environments:
```bash
tox
```

Tests cover key modules and regression checks for physical correctness against benchmarks.

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repo and create a feature branch.
2. Add or update tests as appropriate.
3. Submit a pull request with a clear description of your changes.

For questions or feature requests, use GitHub issues or discussions.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- Bear, J. "Dynamics of Fluids in Porous Media", Dover Publications.
- Zienkiewicz, O.C., Taylor, R.L. "The Finite Element Method", Elsevier.
- [Thesis documentation and related publications will be added here.]

---

## Contact

For more information, please contact the repository owner or open an issue: anna.pekinasova@ucalgary.ca

---

> **Note:** Only a subset of files and examples are listed here due to API limits. For a full and up-to-date directory, visit the [thesis branch](https://github.com/annapekinasova/frozen-ground-fem/tree/thesis) on GitHub.
