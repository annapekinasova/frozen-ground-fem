# frozen-ground-fem

[![License](https://img.shields.io/github/license/annapekinasova/frozen-ground-fem)](LICENSE)

## Overview

**frozen-ground-fem** is a Python package for advanced, multiphysics simulation of frozen ground processes using 1D Finite Element Method (FEM) framework. Designed for research and engineering applications in geotechnics and geosciences, it enables high-fidelity modelling of heat transfer, water migration, phase change, and large-strain consolidation phenomena in freezing and thawing soils. The code is a component of an active research project supporting thesis work in geotechnical and permafrost engineering, including detailed implementations for thermal, consolidation, and fully coupled thermo-hydro-mechanical (THM) processes in 1D soil columns. The package is robust, modular, and extensible, making it suitable for both academic research and practical engineering studies involving permafrost, seasonal frost, and related scenarios of ground freezing and thawing.

> **Note:** For the most stable release, see the `main` branch. Other branches (e.g. `dev`, `thesis`, `feature/...`) may include features under development.

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

### Install latest release using `pip`

```bash
pip install frozen-ground-fem
```

### Install from source on a branch

```bash
git clone --branch thesis https://github.com/annapekinasova/frozen-ground-fem.git
cd frozen-ground-fem
pip install .
```

### Development install

```bash
git clone https://github.com/annapekinasova/frozen-ground-fem.git
cd frozen-ground-fem
git switch dev
pip install -e .
pip install -r requirements-dev.txt
```

---

## Usage

### Running an example

Example scripts are provided in the [`examples/`](examples) directory. For instance, to run a coupled consolidation and thermal simulation:

```bash
python examples/coupled_thaw_consolidation_lab_benchmark.py
```

Example scripts are extensively commented and refer to associated input files (CSV/BAT) for parameters and validation data. Note that since this is research code, the examples may be in various states of functionality.

### As a Python package

You may also use the package to develop your own scripts in Python:

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
|       ├── __init__.py          # package initialization, loads some functions/classes into the global namespace
|       ├── materials.py         # Useful constants, Material class for constitutive models
|       ├── geometry.py          # Base classes for geometry and mesh generation
│       ├── thermal.py           # Thermal FEM implementation
│       ├── consolidation.py     # Large strain consolidation FEM
│       └── coupled.py           # Coupled thermo-hydro-mechanical FEM
├── tests/               # Unit and integration tests
├── requirements.txt     # Runtime dependencies
├── requirements-dev.txt # Developer dependencies for testing and code formatting
├── pyproject.toml       # Build system configuration
├── tox.ini              # Tox configuration for automated testing
├── LICENSE
└── README.md
```

---

## Example Scripts

### Notable Examples

- [`thermal_freeze_thaw_benchmark.py`](examples/thermal_freeze_thaw_benchmark.py): Simulates freezing/thawing front in a soil column, benchmarks against analytical/laboratory results.
- [`thermal_trumpet_curves.py`](examples/thermal_trumpet_curves.py): Simulates multi-year cyclic seasonal temperature boundary condition to initialize a stable ground temperature profile over one full year.
- [`consolidation_static_benchmark.py`](examples/consolidation_static_benchmark.py): Validates large strain consolidation solver with standard benchmarks.
- [`coupled_freezing_front_lab_benchmark.py`](examples/coupled_freezing_front_lab_benchmark.py): Laboratory-based benchmark for coupled thermal-hydraulic modelling of a freezing front.
- [`coupled_thaw_consolidation_lab_benchmark.py`](examples/coupled_thaw_consolidation_lab_benchmark.py): Validation of the coupled model with thawing/consolidation laboratory data.

> **Note:** Only a portion of the example scripts are listed here. [View the complete examples directory](https://github.com/annapekinasova/frozen-ground-fem/examples) for more.

Each example script is self-contained and includes detailed comments, references to input files (such as `.csv` for parameters or validation data), and instructions for reproducing published results.

---

## Source Code Details

### `materials.py`

**Purpose:**  
Contains important constants, information on assumed units, and the `Material` class implementing constitutive models for thermal properties, void ratio - effective stress relationship, and void ratio - hydraulic conductivity relationships.

**Key Classes:**
- `Material`: Contains parameters of various constitutive models and methods for updating material properties during a simulation.

**Notable Features:**
- Modular: can be extended to modify the details of constitutive models used in thermal, consolidation, and thermo-hydro-mechanical modelling.

### `geometry.py`

**Purpose:**  
Contains base classes for finite element geometry, including nodes, integration points, elements, and mesh generation. Many classes in this module are abstract base classes that are extended by other modules with specific physics.

**Key Classes:**
- `Node1D`: Basic geometric object in the finite element method, which is primarily responsible for storing primary variable values during a simulation.
- `IntegrationPoint1D`: Basic geometric object in the finite element method, which carries information about interpolated primary variables, secondary solution variables, and material properties.
- `Boundary1D`: Abstract base class for storing boundary condition information.
- `Element1D`: Abstract base class for organizing a single finite element, consisting of nodes and integration points.
- `Mesh1D`: Abstract base class for performing a physics simulation, providing methods for generating a mesh of elements (each containing nodes and integration points) and updating the system as part of a time-stepping scheme.

**Notable Features:**
- Provides abstract base classes for extending different types of physics while reusing common geometric information.
- Classes use Python `@property` decorator extensively, to distinguish between private attributes and public getters and setters, controlling information hiding and validation.

### `thermal.py`

**Purpose:**  
Implements the finite element method for 1D transient conductive-advective heat transfer in freezing/thawing soils, including phase change.

**Key Classes:**
- `ThermalElement1D`: Computes element-level heat flow (conductivity) and heat storage matrices, handles phase change via latent heat and degree of saturation models.
- `ThermalBoundary1D`: Encapsulates Dirichlet (temperature), Neumann (heat flux), and temperature gradient boundary conditions. Supports time-dependent (function) boundary values.
- `ThermalAnalysis1D`: High-level manager for simulation: mesh and boundary definition, time stepping, global matrix assembly, iterative solution, and updating all states (temperature, heat flux, phase change).

**Notable Features:**
- Implicit time integration with adjustable implicit factor (default of 0.5 is Crank-Nicolson) and options for constant or adaptive time stepping using the `ThermalAnalysis1D.solve_to()` method.
- Latent heat handled via enthalpy or degree-of-saturation formulations.
- Modular: can be extended to higher dimensions or more complex couplings.

### `consolidation.py`

**Purpose:**  
Implements 1D large strain consolidation of soils, accounting for evolving void ratio, permeability, and effective stress.

**Key Classes:**
- `ConsolidationElement1D`: Computes element stiffness and mass matrices for large strain consolidation. Integrates non-linear soil mechanical behaviour and hydraulic conductivity.
- `ConsolidationBoundary1D`: Handles boundary conditions for void ratio and water flux, including time-dependent values.
- `ConsolidationAnalysis1D`: Sets up consolidation problems, manages the mesh, global matrix assembly, time stepping, iterative correction, and computes settlement/deformation.

**Notable Features:**
- Large strain (nonlinear) consolidation formulation.
- Full tracking of void ratio, effective stress, pore pressure, total stress, and hydraulic properties at all integration points.
- Supports laboratory and field-scale simulations.

### `coupled.py`

**Purpose:**  
Implements coupled 1D thermo-hydro-mechanical (THM) elements by inheriting features from classes in both the `thermal.py` and `consolidation.py` modules, while adding extended functionality relevant to coupled modelling.

**Key Classes:**
- `CoupledElement1D`: Hybrid class enabling simultaneous solution of heat transfer, water migration, and large-strain deformation. Leverages both parent implementations for element-level assembly.
- `CoupledAnalysis1D`: Hybrid class for solving coupled thermal and consolidation problems, manages the mesh, global matrix assembly, time stepping, iterative correction, and computes temperature profile and settlement/deformation.

**Notable Features:**
- Enables true THM simulations for freezing/thawing soils.
- Easily extendable for more complex coupling (e.g., 2D/3D, advanced material models).

---

## Testing

Run the test suite with:

```bash
pytest tests/
```
or for full environments and type checking:
```bash
tox
```

Tests cover key modules and regression checks for physical correctness against benchmarks. Note that as this is a research code, commits with not all tests passing are common. Collaborators are welcome to submit issues with unit tests, and fixing non-passing unit tests is a good way for new collaborators to contribute.

---
## Why a 1D Formulation?

### Justification

- **Computational Efficiency:**  
  1D models are computationally light, enabling rapid simulations and parameter studies on standard hardware. This is especially crucial for long-term permafrost and climate-driven scenarios, where many runs may be needed.

- **Fundamental Process Understanding:**  
  Many key processes—such as vertical freeze/thaw propagation, water migration, and consolidation—are dominated by vertical gradients. 1D modelling provides valuable insight into these mechanisms and is the natural starting point for model development.

- **Alignment with Data Availability:**  
  Field and laboratory investigations often yield vertical profile data (e.g., borehole temperature logs, soil columns), making 1D models ideal for validation, calibration, and sensitivity analysis.

- **Rapid Prototyping:**  
  1D FEM is an excellent platform for prototyping new physics, testing novel constitutive relations, and benchmarking against analytical solutions before scaling up to higher dimensions.

### Limitations

- **Idealization:**  
  The 1D formulation is restricted to idealized problems (e.g., vertical soil columns, boreholes) where lateral processes are negligible.

- **Lateral Effects:**  
  Many real-world scenarios—such as embankments, slopes, or infrastructure foundations—exhibit strong 2D/3D effects (lateral heat/fluid flow, edge effects) that cannot be captured in 1D.

---

## Vision for 2D/3D Extensions

While 1D serves as an efficient and robust foundation, we recognize the need for higher-dimensional models in engineering and climate applications:

- **Modular and Extensible Design:**  
  The codebase is structured to facilitate future extension to 2D and 3D, with reusable solvers, material models, and boundary condition handlers.

- **Algorithmic Pathway:**  
  Finite element assembly, time-stepping, and constitutive algorithms are dimension-agnostic and can be generalized to higher-dimensional domains.

- **Community-Driven Development:**  
  Input and collaboration from the scientific and engineering community are welcome to guide feature development, benchmarks, and code optimization.

- **Future Roadmap:**  
  - **Short Term:** Expand 1D physics, improve user interface, and enhance I/O.
  - **Medium Term:** Prototype 2D models (e.g., embankment cross-sections, lateral heat flow).
  - **Long Term:** Develop robust 2D/3D capabilities for infrastructure and complex terrain, leveraging high-performance computing and parallelization as needed.

---

## Get Involved

- **Use Cases:** Have a scenario in mind that would benefit from 2D/3D? [Open an issue](https://github.com/annapekinasova/frozen-ground-fem/issues) or join the discussion!
- **Contributions:** Contributions are welcome, from documentation and testing to new features or code optimization.
- **Contact:** For questions, feature requests, or collaboration opportunities, please reach out via GitHub issues or pull requests.

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

- Andersland, O.B. and Ladanyi, B. 2004. "Frozen Ground Engineering, 2nd. ed.", Wiley.
- Bear, J. 1988. "Dynamics of Fluids in Porous Media", Dover Publications.
- Dumais S, Konrad J-M. One-dimensional large-strain thaw consolidation using nonlinear effective stress–void ratio–hydraulic conductivity relationships. Can Geotech J 2018;55(3):414–26. https://doi.org/10.1139/cgj-2017-0221.
- Fox, P. J. and Berles, J. D. (1997). CS2: A piecewise-linear model for large strain consolidation. International Journal for Numerical and Analytical Methods in Geomechanics, 21(7), 453–475, https://doi.org/10.1002/(SICI)1096-9853(199707)21:7%3C453::AID-NAG887%3E3.0.CO;2-B.
- Fox, P. J. and Pu, H. (2015). Benchmark problems for large strain consolidation. Journal of Geotechnical and Geoenvironmental Engineering, 144(3), https://doi.org/10.1061/(ASCE)GT.1943-5606.0001826.
- Pekinasova A, Hayley JL, Karchewski B. Preliminary results of thermo-hydro-mechanical modelling of large-strain deformation of ground in Canadian cold regions. Proc 76th Can Geotech Conf; 2023. Saskatoon, SK, Canada.
- Pekinasova A, Hayley JL, Karchewski B. Benchmark validation of a new numerical tool for settlement prediction of freezing and thawing ground in cold regions. Proc 77th Can Geotech Conf; 2024. Montréal, QC, Canada.
- Yu F, Guo P, Lai Y, Stolle D. Frost heave and thaw consolidation modelling. Part 1: A water flux function for frost heaving. Can Geotech J 2020;57(10):1581–94. https://doi.org/10.1139/cgj-2019-0218.
- Yu F, Guo P, Lai Y, Stolle D. Frost heave and thaw consolidation modelling. Part 2: One-dimensional thermohydromechanical (THM) framework. Can Geotech J 2020;57(10):1595–610. https://doi.org/10.1139/cgj-2019-0306.
- Zienkiewicz, O.C., Taylor, R.L., and Zhu, J.Z. 2013. "The Finite Element Method: Its Basis and Fundamentals", Elsevier.
- [Thesis documentation and related publications will be added here.]

---

## Contact

For more information, please contact the repository owner or open an issue: [anna.pekinasova@ucalgary.ca].

---

> **Note:** Only a subset of files and examples are listed here. For a full and up-to-date directory, visit the [thesis branch](https://github.com/annapekinasova/frozen-ground-fem/tree/thesis) on GitHub.
