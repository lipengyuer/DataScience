import pandas as pd

a = [[1,2],[2,3],[4,5],[5,6]]
df = pd.DataFrame(a)
print(df)
index = df[df[0]>2].index.tolist()
df2 = df.iloc[index]
print(df2)