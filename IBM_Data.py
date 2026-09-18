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

import pandas as pd
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

plt.savefig("My_scatter_plot.png")