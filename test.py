import pandas as pd

df = pd.DataFrame([{'a':1, 'b':2, 'c':3}, {'a':2, "b":4, 'c':6}, {'a':3, 'b':6, 'c':9}])
df.set_index('a', inplace=True)
print(df)