# Estimating Political Bias of Texts on the Basis of a Large News Dataset

This project aims to classify text into political leanings (left, left-center, center, right-center, right). It has three sections that are visible for the user:

- a section where the user can have their own text evaluated and classified
- a section where different media are evaluated based on class average
- a section where different media are evaluated based on article classification

## Description

### Structure

The project consists of four large parts:

- the model
- the frontend / user interface
- the backend where the user input is processed and an API to connect frontend to backend
- the classification of the scraped articles and their evaluation in two different ways (class average and article classification)

There are also a few tests provided.

### Pipeline

When the user enters their text, it is send to the API which sends it to the `text_classification.py` script where the text is preprocessed and then given the model to evaluate. The output of the model is returned to the API which hands it to the frontend for display.

To display the charts of the different media, the scraped data is also preprocessed and given the script to evaluate each article. The results are then further aggregated based on the average probability of each class per medium and the number of articles for each class. The results are saved as csv-files so they don't need to be re-calculated for every refresh of the page. These files are then read by the frontend to visualize them as stacked bar charts. The frontend can not trigger a re-calculation of these values, only read the files. If a re-calculation of the scraped articles is needed (e. g. when the model is swapped out), the respective scripts must be run again.

## Data

Our data stems from two different datasets:

- a dataset of 268k American News Articles
- the AllSides Rankings for English language news outlets

To make the data useful for our classification task, we had to match the names of the media outlets to the countless different domains of the news dataset.
Therefore we used the Levenshtein distance metric to compare domain names and variations of the original outlet names.
We also limited the fraction of a single news outlet to a maximum of a third of articles for smaller classes and to a maximum of five percent for the more frequent classes of our dataset.
The names and name variations used for matching URLs and media outlet names were also filtered out of the dataset.
Thereby we want to avoid the model focusing on accidental media name mentions instead of our classification goal: detecting political bias

## The Model

- what are we using (BERT, Tensorflow, Keras ...)
- what steps of preprocessing do we do
- what is the output

## Getting Started

1. Install uv if not already installed. Choose one of the methods below based on your system.

- **Standalone**
- macOS: `sh <(curl -fsSL <https://astral.sh/uv/install.sh>)`
- Windows: `powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"`

- **Other**
- pip: `pip install uv`
- pipx: `pipx install uv`
- homebrew:`brew install uv`

  Verfy installtion with `uv --help`

  Note: The commands to install uv have not been tested as we had uv already installed when making this project. Here is the documentation for installing uv <https://docs.astral.sh/uv/getting-started/installation/>

1. Make sure you are in the folder "nlp" which is the root of the project.

1. Run command `uv sync`. This command should make you a .venv based on the pyproject.toml.

1. Run the API and the streamlit app. You will need a split terminal to run both at the same time. Make sure you are in the directory "nlp" in both terminals. Run the command `uv run python src/backend/api.py` in one terminal first and the command `uv run streamlit run src/frontend/streamlitapp.py`in the other second. After running the second command, the streamlit app should open automatically in your default browser.

To run the tests, run the command `uv run pytest tests/tests.py`

## Usage

### Classify your own text

Once the site is running, paste a political article into the textbox and press the green button "Analysiere". On the right side of the textbox, there should then appear a bar chart with the probabilities for each class (left, left-center, center, right-center, right) and a line of text underneath that says which class had the highest probability. To delete your input text, i. e. the contents of the textbox, press the red button "Lösche Input".

### Classification of different media

When you scroll down the page, there is a stacked bar chart displaying the political leaning of different media. You can hover over each segment to see the information in a more concise way. The numbers were calculated by first getting the model's estimate on how likely each class is for each article, and then averaging the values per class for each medium. If you click the grey button "->" on the right of the chart, you should see another stacked bar chart. You can hover over the segments here as well. This time, it evaluates the classes based on how many articles of that class appeared in each medium. In this chart, you can click on the segments (it might take multiple clicks) to see which article went into the clicked class. The titles of the articles are listed on the right of the chart and when clicked, it will open a new tab to the article. To return to the previous stacked bar chart, click the grey button "<-" again.

## Help

## Authors

Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;

Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;

Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;

Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;

Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>

## Acknowledgement
