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


class OutletNameVariator:


    def __init__(self, ratings):
        self.ratings = ratings
        self.outlet_name_variations = self.get_outlet_name_variations()


    def get_outlet_name_variations(self):
        outlet_name_insensitives = self.get_outlet_name_insensitives()
        outlet_name_initials = self.get_outlet_name_initials(partial=False)
        outlet_name_partial = self.get_outlet_name_initials(partial=True)
        variation_datasets = [outlet_name_insensitives, outlet_name_initials, outlet_name_partial]
        outlet_name_variations = pd.concat(
            variation_datasets,
            axis=0,
            ignore_index=True
        )

        return outlet_name_variations
    

    def get_outlet_name_insensitives(self):
        media_outlet_names = self.ratings.media_outlet_names.copy()
        outlet_name_insensitives = OutletNameVariator.get_insensitives(media_outlet_names)

        return outlet_name_insensitives
    

    def get_outlet_name_initials(self, partial=False):
        outlet_name_partial = self.ratings.media_outlet_names.copy()
        initials_row = lambda row: OutletNameVariator.name_initials(row, partial)
        outlet_name_partial["name_modification"] = outlet_name_partial["name"].apply(initials_row)
        outlet_name_partial = OutletNameVariator.get_insensitives(
            outlet_name_partial,
            "name_modification",
            "name_modification"
        )
        outlet_name_partial = outlet_name_partial[outlet_name_partial["name_modification"].str.len() > 2]

        return outlet_name_partial
    

    @staticmethod
    def name_initials(outlet_name, partial=False):
        name_components = outlet_name.split(" ")
        initials = ""
        for word in name_components:
            initials += word[0]

        if partial:
            initials += name_components[-1][1:]
            
        return initials
    

    @staticmethod
    def get_insensitives(dataframe, sensitive_column_name="name", modified_column_name="name_modification"):
        dataframe[modified_column_name] = dataframe[sensitive_column_name].str.replace(
            r"[()-+!.\s]",
            "",
            regex=True
        )
        dataframe[modified_column_name] = dataframe[modified_column_name].str.lower()

        return dataframe