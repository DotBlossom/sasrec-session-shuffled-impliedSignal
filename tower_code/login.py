
import pandas as pd
items_tmp = pd.read_parquet("D:/trainDataset/localprops/features_item.parquet")
print("진짜 컬럼들:", items_tmp.columns.tolist())