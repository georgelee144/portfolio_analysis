from scipy import optimize
import pandas as pd
import numpy as np
from portfolio_analysis import portfolio_return, portfolio_volatility


def minimize_volatility(target_return, expected_returns, covariance_matrix):

    inital_weights = [1 / len(expected_returns) for _ in expected_returns]

    # sets bounds of each assest, 0 prevents shorting and 1 prevents overleveraging
    bounds = ((0.0, 1.0),) * len(expected_returns)
    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    return_is_target = {
        "type": "eq",
        "args": (expected_returns,),
        "fun": lambda weights, expected_returns: target_return
        - portfolio_return(weights, expected_returns),
    }
    weights = optimize.minimize(
        portfolio_volatility,
        inital_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1, return_is_target),
        bounds=bounds,
    )

    return weights.x


def optimal_weights(expected_returns, covariance_matrix, points):

    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), points)
    weights = [
        minimize_volatility(target_return, expected_returns, covariance_matrix)
        for target_return in target_returns
    ]

    return weights


def efficent_frontier_curve(expected_returns, covariance_matrix, points=50):

    weights = optimal_weights(expected_returns, covariance_matrix, points)
    rets = [portfolio_return(w, expected_returns) for w in weights]
    vols = [portfolio_volatility(w, covariance_matrix) for w in weights]

    efficent_frontier = pd.DataFrame({"Returns": rets, "Volatility": vols})

    return efficent_frontier
