'''import pandas as pd
url = "/mnt/e/Toán - Tin TCTA/Data Import/winequality-red-selected-columns.csv"
df = pd.read_csv(url, header = None)
print(df)

dh = df.head(5)
dt = df.tail(5)

print(dh)
print(dt)

path = "/mnt/e/Toán - Tin TCTA/Data Export/test.csv"
df.to_csv(path)'''

'''import pandas as pd
import matplotlib.pyplot as plt
url = '/mnt/e/Toán - Tin TCTA/Data Import/winequality-red-selected-columns.csv'
df = pd.read_csv(url, header = 0)
print(df)

df.describe()

df.dtypes

df.describe(include = "all")

df.info()

x = df["fixed acidity"]
y = df["pH"]
plt.scatter (x,y)

plt.title("The relationship between fixed acidity and pH")
plt.xlabel("fixed acidity")
plt.ylabel("pH")

plt.savefig("My_scatter_plot.png")'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. TẠO DỮ LIỆU GIẢ LẬP (Mock Data)
data = {
    'drive-wheels': ['rwd', 'rwd', 'fwd', 'fwd', '4wd', '4wd', 'rwd', 'fwd', '4wd'],
    'body-style': ['convertible', 'hardtop', 'hatchback', 'sedan', 'hatchback', 'sedan', 'sedan', 'convertible', 'hardtop'],
    'price': [35000, 34000, 12000, 14000, 10000, 15000, 22000, 18000, 16000]
}
df = pd.DataFrame(data)
print("--- DỮ LIỆU GỐC ---")
print(df, "\n")

# 2. GOM NHÓM DỮ LIỆU BẰNG GROUPBY
df_group = df[['drive-wheels', 'body-style', 'price']]
df_group_avg = df_group.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
print("--- DỮ LIỆU SAU KHI GROUPBY & TÍNH TRUNG BÌNH ---")
print(df_group_avg, "\n")

# 3. TẠO BẢNG CHÉO BẰNG PIVOT
grouped_pivot = df_group_avg.pivot(index='drive-wheels', columns='body-style', values='price')
# Lấp đầy các giá trị NaN bằng 0 để tránh lỗi khi vẽ biểu đồ
grouped_pivot = grouped_pivot.fillna(0) 
print("--- DỮ LIỆU SAU KHI PIVOT (Lưới chữ nhật) ---")
print(grouped_pivot, "\n")

# 4. TRỰC QUAN HÓA BẰNG HEAT MAP
plt.figure(figsize=(8, 5))
plt.pcolor(grouped_pivot, cmap='RdBu') # Sử dụng hệ màu Đỏ - Xanh dương
plt.colorbar(label='Average Price ($)')

# Tùy chỉnh các trục để hiển thị nhãn chữ
plt.xticks(np.arange(0.5, len(grouped_pivot.columns), 1), grouped_pivot.columns)
plt.yticks(np.arange(0.5, len(grouped_pivot.index), 1), grouped_pivot.index)

plt.title('Heat Map: Average Car Price by Drive Wheels and Body Style')
plt.xlabel('Body Style')
plt.ylabel('Drive Wheels')
plt.show()