import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = {
    "area": [1200, 1500, 1800, 1000, 2000, 1600, 1400, 2200, 1100, 1700,
             1300, 1900, 1550, 2100, 1250, 1750, 1450, 2300, 1350, 1850],
    "bedrooms": [2, 3, 3, 2, 4, 3, 2, 4, 2, 3,
                 2, 4, 3, 4, 2, 3, 3, 5, 2, 4],
    "age": [10, 5, 8, 15, 3, 7, 12, 2, 20, 6,
            14, 4, 9, 1, 18, 5, 11, 2, 16, 7],
    "location_score": [7, 8, 9, 6, 9, 8, 7, 10, 6, 8,
                       7, 9, 8, 10, 6, 8, 7, 10, 6, 9],
    "price": [4500000, 6000000, 7200000, 3500000, 8500000,
              6500000, 5000000, 9500000, 3200000, 6800000,
              4700000, 8000000, 6200000, 9000000, 3800000,
              7000000, 5200000, 10000000, 4000000, 7600000]
}

df=pd.DataFrame(data)
x=df.drop("price",axis=1)
y=df["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"MAE : {mae:.2f} MSE : {mse:.2f} R2 : {r2:.2f}")
plt.figure(figsize=(5,8))
plt.scatter(range(len(y_test)),y_test,color="blue",label="Actual",alpha=0.6)
plt.scatter(range(len(y_pred)),y_pred,color="red",label="Predicted",alpha=0.6)
plt.title("Actual vs Predicted")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()