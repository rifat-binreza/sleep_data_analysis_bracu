import pandas as pd

from app import _clean_dataset, _numeric_stats


def test_clean_dataset_engineers_expected_features():
    raw = pd.DataFrame(
        {
            "Person ID": [1],
            "Age": [35],
            "BMI Category": ["Normal Weight"],
            "Blood Pressure": ["120/80"],
            "Sleep Disorder": [None],
            "Quality of Sleep": [8],
            "Sleep Duration": [8.0],
            "Stress Level": [4],
            "Physical Activity Level": [59],
        }
    )

    result = _clean_dataset(raw)

    assert "Person ID" not in result
    assert "Blood Pressure" not in result
    assert result.loc[0, "BMI Category"] == "Normal"
    assert result.loc[0, "Sleep Disorder"] == "None"
    assert result.loc[0, "Systolic_BP"] == 120
    assert result.loc[0, "Diastolic_BP"] == 80
    assert result.loc[0, "Age_Group"] == "Adult"
    assert result.loc[0, "Sleep_Efficiency"] == 1.0


def test_numeric_stats_supports_constant_series():
    minimum, maximum, median, step = _numeric_stats(pd.Series([7, 7]))

    assert (minimum, maximum, median, step) == (6.0, 8.0, 7.0, 1.0)
