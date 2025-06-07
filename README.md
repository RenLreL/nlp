# Estimating Political Bias of Texts on the Basis of a Large News Dataset

A simple first description

## Description

More in-depth

- a point
- another one

## Data

Our data stems from two different datasets:

- a dataset of 268k American News Articles
- the AllSides Rankings for English language news outlets

To make the data useful for our classification task, we had to match the names of the media outlets to the countless different domains of the news dataset.
Therefore we used the Levenshtein distance metric to compare domain names and variations of the original outlet names.
We also limited the fraction of a single news outlet to a maximum of a third of articles for smaller classes and to a maximum of five percent for the more frequent classes of our dataset.
The names and name variations used for matching URLs and media outlet names were also filtered out of the dataset.
Thereby we want to avoid the model focusing on accidental media name mentions instead of our classification goal: detecting political bias

## Getting Started

### Installation

1. Install uv if not already installed. Choose one of the methods below based on your system.

- macOS: `sh <(curl -fsSL <https://astral.sh/uv/install.sh>)`
- Windows: `powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"`
- pipx: `pipx install uv`

Verfy installtion with: `uv --help`

Note: The commands to install uv have not been tested as uv was already installed when making this project. If you run into trouble with the commands above, search the internet for how to install uv or ask an AI of your choice.

1. Move into the folder "frontend": `cd frontend`

The command assumes you are in the folder "nlp"

1. Run command: `uv sync`

This command should make you a .venv based on the pyproject.toml.

1. Turn off all virtual enviroments that may be turned on by default.

Run command: `deactivate` or `conda deactivate`

This might not be strictly necessary but is documented as a step for the sake of consistency.

1. Run the API and the streamlit app.

You will need a split terminal to run both at the same time. Make sure you are in the directory "frontend" in both terminals. Run the command `uv run python src/api.py` in one terminal first and the command `uv run streamlit run src/streamlitapp.py`in the other second. After running the second command, the streamlit app should open automatically in your default browser.

## Usage

## Help

## Authors

Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;

Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;

Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;

Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;

Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>

## Acknowledgement
