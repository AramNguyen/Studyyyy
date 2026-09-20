import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ==========================================
# 6. TRỰC QUAN HÓA DỮ LIỆU (VISUALIZATION)
# ==========================================

# Khởi tạo kích thước khung hình
plt.figure(figsize=(8, 5))

# A. Vẽ các điểm dữ liệu thực tế (chấm xanh)
plt.scatter(df['highway-mpg'], df['prices'], color='blue', label='Thực tế (Actual Data)')

# B. Vẽ đường xu hướng dự đoán của mô hình (đường xanh lá)
y_pred_line = lm.predict(X)
plt.plot(df['highway-mpg'], y_pred_line, color='green', linewidth=2, label='Đường hồi quy (Regression Line)')

# C. Đánh dấu điểm dự đoán mới tại 30 highway-mpg (ngôi sao đỏ)
plt.scatter([30.0], predicted_price_2, color='red', marker='*', s=200, zorder=5, 
            label=f'Dự đoán tại 30 mpg: ${predicted_price_2[0]:.2f}')

# D. Cấu hình tiêu đề, chú thích và lưới cho biểu đồ
plt.title('Dự đoán giá xe dựa trên mức tiêu thụ nhiên liệu (Highway-mpg)')
plt.xlabel('Mức tiêu thụ nhiên liệu (Highway-mpg)')
plt.ylabel('Giá xe (Prices - $)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Hiển thị biểu đồ
plt.show()