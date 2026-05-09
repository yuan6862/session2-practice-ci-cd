import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv('training_data.csv')
model = LinearRegression().fit(data[['x']], data['y'])


with open('linear_model.txt', 'w') as f:
    f.write(f"Slope: {model.coef_[0]}, Intercept: {model.intercept_}")

print("训练完成！linear_model.txt 已生成。")
