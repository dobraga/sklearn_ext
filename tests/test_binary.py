from sklearn_ext.preprocessing import TransformBinary
from sklearn_ext.datasets import load_house_prices


def test_binary():
    df, _ = load_house_prices()

    binary = TransformBinary(threshold_min=0.75, threshold_max=0.9, fillna="None")
    binary.fit(df)

    assert binary.feature_names_in_.tolist() == [
        "MSZoning",
        "LandContour",
        "Condition1",
        "BldgType",
        "RoofStyle",
        "ExterCond",
        "BsmtCond",
        "BsmtFinType2",
        "GarageQual",
        "Fence",
        "SaleType",
        "SaleCondition",
    ]

    assert binary.get_feature_names_out().tolist() == [
        "bin_MSZoning_RL",
        "bin_LandContour_Lvl",
        "bin_Condition1_Norm",
        "bin_BldgType_1Fam",
        "bin_RoofStyle_Gable",
        "bin_ExterCond_TA",
        "bin_BsmtCond_TA",
        "bin_BsmtFinType2_Unf",
        "bin_GarageQual_TA",
        "bin_Fence_None",
        "bin_SaleType_WD",
        "bin_SaleCondition_Normal",
    ]
