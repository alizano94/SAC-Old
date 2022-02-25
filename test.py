import pandas as pd
import numpy as np

#Create a DataFrame

df1 = {
    'State':['Arizona AZ','Georgia GG','Newyork NY','Indiana IN','Florida FL'],
   'Score':[62,47,55,74,31]}

df1 = pd.DataFrame(df1,columns=['State','Score'])
print(df1)

print(df1.iloc[1])
df1.loc[len(df1.index)] = df1.iloc[1]
print(df1)
