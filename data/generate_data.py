import numpy as np
import pandas as pd

def generate(n = 200, seed = 42):
  np.random.seed(seed)
  
  # target: AD = 0 & FTD = 1
  target = np.random.binomial(1,0.4,n)

  # demo & cognition
  age = np.random.normal(70,7,n)
  mmse = np.where(target == 0, np.random.normal(20,4,n), np.random.normal(22,4,n))
  moca = np.where(target == 0, np.random.normal(18,4,n), np.random.normal(19,4,n))
  cdr_sb = np.random.uniform(1.0,10.0,n)

  # fluid biomarkers
  ptau217 = np.where(target == 0, np.random.normal(2.5,0.5,n), np.random.normal(1.0,0.3,n))
  nfl = np.where(target == 1, np.random.normal(45,10,n), np.random.normal(20,5,n))

  # mri parameters
  hipo_vol = np.where(target == 0, np.random.normal(2800,300,n), np.random.normal(3400,300,n))
  frontal_vol = np.where(target == 1, np.random.normal(22000,2000,n), np.random.normal(28000,2000,n))

  df = pd.DataFrame({
    'ID':[f'{i}' for i in range(n)],
    'dx_ftd': target.astype(int),
    'age': np.round(age,1),
    'mmse': np.clip(np.round(mmse), 0,30),
    'moca': np.clip(np.round(moca), 0,30),
    'cdrsb': np.round(cdr_sb,1),
    'ptau217': np.round(ptau217,3),
    'nfl': np.round(nfl,2),
    'hipoV': np.round(hipo_vol,1),
    'frontalV': np.round(frontal_vol,1)
  })

  # inject missingness of 5%
  mask = np.random.rand(*df.shape) < 0.05
  df = df.mask(mask & (df.columns != 'dx-ftd') & ( df.columns != 'ID'))

  return df

if __name__ == "__main__":
  df = generate()
  df.to_parquet("data/mock9.parquet", engine="pyarrow", index=False)
  print("mock data created successfully")
