"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-05-26: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""


import pandas as pd

class RatingsData:


    def __init__(self, ratings_data_path):
        self.data_path = ratings_data_path
        self.dataset = pd.read_csv(ratings_data_path)
        self.media_outlet_names = self.get_media_outlet_names()


    def get_media_outlet_names(self):
        media_outlet_list = self.dataset["name"].unique()
        
        media_outlet_names = pd.DataFrame(
            {
                "name": media_outlet_list
            }
        )

        print(media_outlet_names)

        return media_outlet_names
    