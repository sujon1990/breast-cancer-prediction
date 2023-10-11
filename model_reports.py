import os
import pandas as pd

pd.set_option('display.width', 120)
pd.set_option('display.max_columns',8)

files = os.listdir('reports')


for file in files:
    print(file.replace('.csv', ''))
    print('************** Top 10 **************')
    report = pd.read_csv(f'reports/{file}')
    print(report.sort_values(by=['score'], ascending=False).head(10))
    print('===============================================================================================================')
