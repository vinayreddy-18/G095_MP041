from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def test_model_training():
    X = [[2, 5], [4, 10], [6, 15], [8, 20]]
    y = [50, 80, 110, 140]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict([[5, 12]])

    assert len(predictions) == 1
    assert predictions[0] > 0


def test_model_multiple_predictions():
    X = [[2, 5], [4, 10], [6, 15], [8, 20]]
    y = [50, 80, 110, 140]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    predictions = model.predict([[3, 7], [5, 12], [7, 18]])

    assert len(predictions) == 3


def test_model_mae():
    actual = [100, 150, 200]
    predicted = [110, 140, 190]

    mae = mean_absolute_error(actual, predicted)

    assert mae == 10