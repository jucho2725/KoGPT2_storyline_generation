import pandas as pd

df = pd.read_csv("../data/korean_naver_2.csv")

print(len(df))

df2 = df[:5]
df2.to_csv("data/korean_naver_ex.csv")