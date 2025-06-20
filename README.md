# Estimating Political Bias of Texts on the Basis of a Large News Dataset

A simple first description

## Description

More in-depth

a
- point
- another one
# Data

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


## Usage


## Help

## Authors

Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>

## Acknowledgement

