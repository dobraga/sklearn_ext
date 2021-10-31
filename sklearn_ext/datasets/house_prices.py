import pandas as pd
from os.path import dirname, join

module_path = dirname(__file__)


def load_house_prices():
    """https://www.kaggle.com/c/house-prices-advanced-regression-techniques"""

    train = pd.read_csv(join(module_path, "data", "house_prices/train.csv"))
    test = pd.read_csv(join(module_path, "data", "house_prices/test.csv"))

    return train, test
