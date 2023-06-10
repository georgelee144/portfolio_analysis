import pandas as pd
import numpy as np
from scipy import stats


def annualized_returns(returns_array, periods_in_year):

    return ((returns_array + 1).prod() ** (periods_in_year / len(returns_array))) - 1


def semivariance(returns_array, cutoff, ddof=0):

    return returns_array[returns_array < cutoff].var(ddof=0)


def semideviation(returns_array, cutoff, ddof=0):

    return semivariance(returns_array, cutoff, ddof) ** 0.5


def sharpe_ratio(returns, volatility, risk_free_rate=0):

    return (returns - risk_free_rate) / volatility


def wealth_index(returns_array, starting_wealth=100):

    return starting_wealth * (returns_array + 1).cum_prod()


def drawdown(wealth_index):

    previous_peaks = wealth_index.cummax()
    drawdowns = wealth_index / previous_peaks - 1

    return pd.DataFrame(
        {
            "wealth_index": wealth_index,
            "previous Peak": previous_peaks,
            "drawdown": drawdowns,
        }
    )


def var_historic(returns_array, level=5):

    return_at_level = np.percentile(returns_array, level)

    if return_at_level > 0:
        raise ValueError("No negative returns in returns_array")
    else:
        return return_at_level


def condtiontal_var(returns_array, cutoff):

    worse_returns = returns_array <= cutoff

    return returns_array[worse_returns].mean()


def var_gaussian(r, level=5, modified=False, ddof=0):

    # compute the Z score assuming it was Gaussian
    z = stats.norm.ppf(level / 100)

    # Cornish-Fisher modification, tries to fit any curve to normal disturbution
    # https://faculty.washington.edu/ezivot/econ589/ssrn-id1997178.pdf
    if modified:
        s = stats.skewness(r)
        k = stats.kurtosis(r)

        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )

    return r.mean() + z * r.std(ddof=ddof)


def portfolio_return(weights, returns):

    return weights.T @ returns


def portfolio_volatility(weights, covariance_matrix):

    return (weights.T @ covariance_matrix @ weights) ** 0.5

