import pandas as pd

a = {"a": 1, "b": 2}
b = {"a": 3, "b": 4}
data = pd.DataFrame([a, b])
print(data)
print(data.median)