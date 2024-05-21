import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
import pandas as pd


def run_exuberance_index_regression_event_study_data(data):
    """This function runs the regression of macro fundamentals on the sentiment
    index."""

    # Define the regression formula
    formula_no_fe = "McDonald_Sentiment_Index ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth  + Q('10y_Maturity_Bond_Yield_US') + VIX_Daily_Close_Quarterly_Mean + Moody_Rating_PD"

    formula_fe = formula_no_fe + " + C(Country)"

    # Run the regression
    model_no_fe = smf.ols(formula_no_fe, data=data).fit()

    # Run the regression with FE

    model_fe = smf.ols(formula_fe, data=data).fit()

    # Create a Stargazer object
    summary = Stargazer([model_no_fe, model_fe])

    summary.covariate_order(
        [
            "Q('Public_Debt_as_%_of_GDP')",
            "GDP_in_Current_Prices_Growth",
            "Q('10y_Maturity_Bond_Yield_US')",
            "VIX_Daily_Close_Quarterly_Mean",
            "Moody_Rating_PD",
        ]
    )

    # Create a dictionary mapping old names to new names
    rename_dict = {
        "Q('Public_Debt_as_%_of_GDP')": "Public Debt as \% of GDP",
        "GDP_in_Current_Prices_Growth": "GDP Growth in Current Prices",
        "Q('10y_Maturity_Bond_Yield_US')": "10 Year Maturity Bond Yield US",
        "VIX_Daily_Close_Quarterly_Mean": "VIX Daily Close",
        "Moody_Rating_PD": "Moody Sovereign Rating",
        "McDonald_Sentiment_Index": "McDonald and Loughran Sentiment Index",
    }

    # Rename the covariates
    summary.rename_covariates(rename_dict)

    # Add Custom Lines
    summary.add_line("Country FE", [" ", "$\\checkmark$"])

    summary.show_degrees_of_freedom(False)

    summary.title("Regression of Sentiment Index on Economic Fundamentals")

    summary.add_custom_notes(
        [
            "The dependent variable is the sentiment index at the",
            "last day of the quarter obtained using, the dictionary from \citet{loughran_when_2011}.",
        ]
    )

    return model_no_fe, summary
