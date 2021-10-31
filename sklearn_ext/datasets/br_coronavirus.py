import pandas as pd


def load_br_covid():
    return pd.read_csv(
        "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    )
