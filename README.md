# SchNet–Gaussian 16 Interface

A lightweight interface that enables **SchNet Neural Network Potentials (NNPs)** to be used as an external energy and force calculator within **Gaussian 16**.

This project bridges **Gaussian 16** and **SchNetPack** through Gaussian's **External** interface, allowing Gaussian to perform geometry optimizations and frequency calculations using a trained SchNet model instead of quantum chemical energy and force evaluations. :contentReference[oaicite:0]{index=0}

---

## Features

- Interface between Gaussian 16 and SchNetPack
- Geometry optimization using SchNet NNP
- Frequency calculation support
- Compatible with pretrained SchNet models
- Simple integration into existing Gaussian workflows
- Easily configurable through a single Python script

---

## Workflow

```text
Gaussian 16
      │
      ▼
External Interface
(# external)
      │
      ▼
gausch.sh
      │
      ▼
gausch.py
      │
      ▼
SchNet Neural Network Potential
      │
      ▼
Predicted Energy & Forces
      │
      ▼
Returned to Gaussian 16
```

---

## Requirements

- Gaussian 16
- Python 3
- PyTorch
- SchNetPack
- ASE (Atomic Simulation Environment)

The Python environment should contain all required packages and be activated in `gausch.sh`. :contentReference[oaicite:1]{index=1}

---

## Installation

Clone this repository and place

```
gausch.py
gausch.sh
```

in your working directory (or another directory accessible by Gaussian).

---

## Usage

### 1. Prepare a trained SchNet model

Place your trained SchNet model (e.g. `best_model`) in the directory specified by `nnp_path`.

By default, the interface searches the current working directory.

---

### 2. Configure Gaussian

Include the following keyword in your Gaussian input file:

```text
# external='bash gausch.sh' opt=nomicro
```

`opt=nomicro` is required when performing geometry optimizations through the Gaussian External interface.

Additional Gaussian keywords (e.g. `freq`, `nosymm`, `charge`, and `multiplicity`) may be used as usual.

---

### 3. Configure the Python environment

Modify `gausch.sh` so that it activates the Python environment containing PyTorch, ASE, and SchNetPack before executing the interface.

Example:

```bash
source ~/script/python_env.sh
```

---

### 4. Configure the interface

Open `gausch.py` and modify the parameters in `preprocess()` according to your environment.

Typical parameters include:

- `nnp_path`
- `model`
- `device`
- `mpath`
- SchNet architecture parameters (if rebuilding a model)

---

### 5. Run Gaussian

```bash
g16 input.gjf output.log
```

The interface automatically predicts energies and forces using the trained SchNet model and returns them to Gaussian.

---

## Notes

- Supports loading a pretrained `best_model`.
- Supports rebuilding a SchNet model from checkpoints.
- Frequency calculations are supported through ASE's phonon module.
- CPU execution is enabled by default. GPU execution can be enabled by changing the device setting.

---

## Citation

If this interface contributes to your research, please cite the corresponding publication.

P. K. Tsou, H. T. Huynh, H. T. Phan, J. L. Kuo, Phys. Chem. Chem. Phys., 2023, 25, 3332. A self-adapting firstprinciples
exploration on the dissociation mechanism in sodiated aldohexose pyranoses assisted with neural
network potentials.

H. T. Phan, P. K. Tsou, P. J. Hsu and J. L. Kuo, Phys. Chem. Chem. Phys., 2023, 25, 5817. A first-principles
exploration of the conformational space of sodiated pyranose assisted by neural network potentials.
