import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

# 读取txt文件
def read_data(filename):
    data = np.loadtxt(filename, delimiter=' ')
    true_values = data[:, 0]
    predicted_values = data[:, 1]
    return true_values, predicted_values

# 回归分析并绘制散点图
def plot_regression(true_values, predicted_values):
    # 散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predicted_values, color='blue', label='Predicted vs True')

    # 拟合直线
    true_values = true_values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(true_values, predicted_values)
    predicted_line = regressor.predict(true_values)

    # 计算标准差
    residuals = predicted_values - predicted_line
    std_dev = np.std(residuals)

    # 绘制回归直线
    plt.plot(true_values, predicted_line, color='red', label='Regression Line')

    # 添加标准差带
    plt.fill_between(
        true_values.flatten(),
        predicted_line - std_dev,
        predicted_line + std_dev,
        color='pink',
        alpha=0.3,
        label=f'Regression ±1 std (std={std_dev:.2f})'
    )

    # 设置图表信息
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Plot with True vs Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 读取文件路径
    filename = input("/media/lhj/Momery/PD_predictDL/Data/UPDRS2_1.txt")
    
    # 读取数据
    try:
        true_values, predicted_values = read_data(filename)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

    # 绘制回归图
    plot_regression(true_values, predicted_values)
