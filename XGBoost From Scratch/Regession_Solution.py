from XGBoostRegressor import XGBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from plotly import graph_objs

# Generating sample regression data
base_data_origin, predicted_data_origin = make_regression(
    n_samples=1000, n_features=10, noise=100, random_state=42)
base_data_origin_train, base_data_origin_test, predicted_data_origin_train, predicted_data_origin_test = train_test_split(
    base_data_origin, predicted_data_origin, test_size=0.2, random_state=42)

# build xgboost for regressor
xgb_regressor = XGBoostRegressor()
xgb_regressor.fit(base_data_origin_train, predicted_data_origin_train)

# make prediction
predictions = xgb_regressor.predict(base_data_origin_test)

# demonstrate on figure
figure = graph_objs.Figure()
figure.add_trace(graph_objs.Scatter(x=predicted_data_origin_test, y=predictions, mode='markers',
                                    marker=dict(color='blue'), name='Actual vs Predicted'))

figure.add_trace(graph_objs.Scatter(x=predicted_data_origin_test, y=predicted_data_origin_test, mode='lines',
                                    line=dict(color='red', dash='dash'), name='Perfect Prediction'))

figure.update_layout(title='Actual vs Predicted Values (Regression)',
                     xaxis_title='Actual',
                     yaxis_title='Predicted')

figure.show()
