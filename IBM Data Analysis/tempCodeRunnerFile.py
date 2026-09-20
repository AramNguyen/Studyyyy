# Linear Regression Prediction Example

#Chương trình dưới đây mô phỏng lại quá trình train model, kiểm tra coefficient và dự đoán giá trị mới tương tự như trong slide bài giảng.


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Tạo dữ liệu giả lập (mock data) có xu hướng tương đồng
# Highway-mpg tăng thì giá (price) giảm
data = {
    'highway-mpg': [22, 25, 28, 31, 35],
    'prices': [20000, 18000, 15000, 13000, 10000]
}
df = pd.DataFrame(data)

# 2. Tách biến - Lưu ý X phải là mảng 2 chiều (dùng 2 dấu ngoặc vuông)
X = df[['highway-mpg']] 
y = df['prices']

# 3. Khởi tạo và huấn luyện mô hình
lm = LinearRegression()
lm.fit(X, y)

# 4. Trích xuất thông số của mô hình
coefficient = lm.coef_[0]
intercept = lm.intercept_

print(f"Hệ số góc (lm.coef_): {coefficient:.8f}")
print(f"Tung độ gốc (Intercept): {intercept:.2f}")
print(f"Phương trình: Price = {intercept:.2f} + ({coefficient:.2f}) * highway-mpg")
print("-" * 40)

# 5. Dự đoán giá xe với 30 highway-mpg
# Cách 1: Dùng chuẩn bài bản như bài giảng (numpy reshape)
x_new_standard = np.array(30.0).reshape(-1, 1)
predicted_price_1 = lm.predict(x_new_standard)

# Cách 2: Góc "tà đạo" (nhanh gọn lẹ bằng mảng 2D Python list)
x_new_quick = [[30.0]]
predicted_price_2 = lm.predict(x_new_quick)

print(f"Kết quả dự đoán (Cách 1 - Reshape): ${predicted_price_1[0]:.2f}")
print(f"Kết quả dự đoán (Cách 2 - 2D List): ${predicted_price_2[0]:.2f}")