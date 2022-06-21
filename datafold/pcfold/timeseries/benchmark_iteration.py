from time import time

import numpy as np
import pandas as pd

from datafold import TSCDataFrame

data = np.random.default_rng(1).uniform(size=(1000000, 100))
df = TSCDataFrame.from_array(data)
df.is_validate = False

# start = time()
# for i in df.iterrows():
#     pass
# print(f"iterrows took time = {time() - start:.5f} seconds")

# start = time()
# for i in range(data.shape[0]-1):
#     df.iloc[i:i+1, :]
# print(f"iloc took time = {time() - start:.5f} seconds")


# df.is_validate = False
# start = time()
# for i in range(data.shape[0]-1):
#     a = df.iloc[i:i+1, :]
# print(f"iloc:range took time = {time() - start:.5f} seconds")

start = time()
df.is_validate = False
for i in np.split(df, df.shape[0] // 2):
    i
print(f"iloc:range took time = {time() - start:.5f} seconds")

# start = time()
# idx = pd.MultiIndex.from_arrays([[0, 0], [0, 1]])
# for i in range(data.shape[0]-2):
#     TSCDataFrame(data[i:i+2], validate=False, index=idx)
#     # pd.DataFrame(data[i:i+2], index=idx)
# print(f"DataFrame cast range took time = {time() - start:.5f} seconds")
