import pandas as pd
import numpy as np

df = pd.read_csv("prices_2020.csv", names=[
    'Date', 'UUID', 'Diesel', 'E5', 'E10'])
ids = ['51df69e3-03e1-0f50-e100-80009459e038', '51d4b432-a095-1aa0-e100-80009459e03a',
       'bf9d3a1f-c8a0-4ec3-93ed-018900da43c1', 'fd606c1d-2f43-47df-8934-d1ea98871767', 'afae9263-388f-4fdf-bac3-d9b8ec670937', '3685377f-1884-4466-a314-276b5d4a8545', 'bb262a44-fcd8-4558-bf11-e118f6d96a5d',
       'cd8ba6a6-bf87-1ed6-a3fe-b698130417bc']
df['Brand'] = 'placeholder'
for id in enumerate(ids):
    if id == '51df69e3-03e1-0f50-e100-80009459e038':
        df['Brand'] = 'Jet'
    elif id == '51d4b432-a095-1aa0-e100-80009459e03a':
        df['Brand'] = 'Jet'
    elif id == 'bf9d3a1f-c8a0-4ec3-93ed-018900da43c1':
        df['Brand'] = 'Eni'
    elif id == 'fd606c1d-2f43-47df-8934-d1ea98871767':
        df['Brand'] = 'Total'
    elif id == 'afae9263-388f-4fdf-bac3-d9b8ec670937':
        df['Brand'] = 'Total'
    elif id == '3685377f-1884-4466-a314-276b5d4a8545':
        df['Brand'] = 'Total'
    elif id == 'bb262a44-fcd8-4558-bf11-e118f6d96a5d':
        df['Brand'] = 'Esso'
    elif id == 'cd8ba6a6-bf87-1ed6-a3fe-b698130417bc':
        df['Brand'] = 'Jet'

print(df.head())
