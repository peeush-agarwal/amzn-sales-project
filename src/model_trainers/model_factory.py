class ModelFactory:
    """Build estimator instances by name.

    Supported names: 'random_forest', 'gbr', 'ridge'
    """

    @staticmethod
    def create(name: str, **params):
        name = name.lower()
        if name == "linear_regression":
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**params)
        if name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**params)
        if name in ("gbr", "gradient_boosting", "gradientboosting"):
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(**params)
        if name in ("ridge", "linear_ridge"):
            from sklearn.linear_model import Ridge

            return Ridge(**params)
        raise ValueError(f"Unknown model name: {name}")
