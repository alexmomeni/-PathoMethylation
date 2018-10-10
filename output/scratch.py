import pandas as pd

df = pd.read_csv('/Users/alexmomeni/Documents/Alex/Stanford/code/histo_methylation/output/LGGGBM/SG/Random Forest/metrics.csv',
                 header = 0, index_col=0)
df = df.sort_values(by=['AUC'],ascending=False)
df = df.loc[:,'AUC']
#df  = df.iloc[:100]

genes = list(df.index)
l = []
for gene in genes:
    try:
      x = gene.split('-')
      l.append(x[0])
    except:
        x = gene
        l.append(x)
df.index = l

df.to_csv('example.csv')