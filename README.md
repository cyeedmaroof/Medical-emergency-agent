# Medical Emergency Agent

This repository provides tools and interfaces for a Medical Emergency Agent, leveraging a property graph knowledge store (`kgstore`) and interactive Jupyter notebooks. The project enables users to interact with medical emergency data and visualize or analyze information through notebooks and a user interface (UI) that integrates with the property graph store.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Property Graph Store: kgstore](#property-graph-store-kgstore)
- [UI Configuration](#ui-configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

- Two main Jupyter notebooks for direct exploration and execution.
- Property graph knowledge store (`kgstore`) for flexible data management.
- UI integration to change the data source location to `kgstore`.

## Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/) (for dependency management)
- Jupyter Notebook

## Installation

### 1. Install Poetry

Poetry is a tool for dependency management and packaging in Python. To install Poetry, run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or, for pipx users:

```bash
pipx install poetry
```

Make sure to add Poetry to your PATH according to the install instructions output.

### 2. Clone the Repository

```bash
git clone https://github.com/cyeedmaroof/Medical-emergency-agent.git
cd Medical-emergency-agent
```

### 3. Install Dependencies

```bash
poetry install
```

This will create a virtual environment and install all necessary dependencies.

### 4. Activate the Virtual Environment

You can use Poetry's shell to activate the environment:

```bash
poetry shell
```

Or, run commands directly with:

```bash
poetry run <command>
```

## Usage

### Running the Jupyter Notebooks

There are two main `.ipynb` files in the repository. You can run them directly by pressing `run all cell`

Then, open your browser and navigate to the `localhost:[port number]` to start interacting with the Medical Emergency Agent.


## UI Configuration

In the UI, you can change the data source location to `kgstore` to access and work with all available data. 
