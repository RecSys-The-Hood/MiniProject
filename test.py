import numpy as np
import pandas as pd

x = np.array([5,4,np.nan,2,np.nan,3])
y = np.array([0,3,3,np.nan,2,4])

x_df = pd.DataFrame(x, columns = ['a'])
y_df = pd.DataFrame(y, columns = ['a'])

x_indices = x_df[x_df['a'].notnull()].index
y_indices = y_df[y_df['a'].notnull()].index

indices = x_indices.intersection(y_indices)

x_reqd = x_df.iloc[indices]
y_reqd = y_df.iloc[indices]

print(sum(y_reqd['a']))

