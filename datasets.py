import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelBinarizer


class Dataset:
    
    def __init__(self, config):

        self.config=config
        self.DMvalues = self.get_DMvalues()
        if self.config.clustering == None:
            self.binarize()
        else:
            self.cluster_binarize()
        self.DMvalues = self.DMvalues.iloc[:,0:self.config.n_tasks]
    
    def get_DMvalues(self):
        
        if self.config.source == "KIRC":
            DMvalues = pd.read_table("inputs/DMvaluesKIRC.txt",header=0,index_col=0).T            
        else:
            DMvalues = pd.read_table("inputs/DMvaluesLGGGBM.txt",header=0,index_col=0).T
            array = np.load('inputs/case_labels.npy')
            labels_GBMLGG = pd.DataFrame(data=array[1:, 1:], index=array[1:, 0], columns=['Type'])
            if self.config.source == "LGGGBM":
                labels =  labels_GBMLGG
            if self.config.source == "LGG":
                labels = labels_GBMLGG[labels_GBMLGG.Type == "LGG"]
            if self.config.source == "GBM":
                labels =  labels_GBMLGG[labels_GBMLGG.Type == "GBM"]
            DMvalues = self.intersect_rows(DMvalues, labels)[0]
        
        return DMvalues
                
    def binarize(self):
    
        self.DMvalues[self.DMvalues>0] = 1
        self.DMvalues[self.DMvalues<0] = -1
        self.DMvalues = self.DMvalues.astype(int)
        binary_genes = np.array([(len(list(np.unique(self.DMvalues[column]))) == 2) for column in self.DMvalues.columns])
        self.DMvalues = self.DMvalues.iloc[:,binary_genes]
        self.remove_unbalanced_genes()  
        self.DMvalues = self.DMvalues.apply(lambda x:LabelBinarizer().fit_transform(x)[:,0], axis=0)
                    
    def cluster_binarize(self):

        if self.config.clustering == "kmeans":
            model = KMeans(n_clusters=self.config.n_clusters, random_state=0, n_init=50)
        if self.config.clustering == "hierarchical":
            model = AgglomerativeClustering(n_clusters=self.config.n_clusters, affinity='euclidean', linkage='ward')
        labels = model.fit_predict(self.DMvalues.T)
        
        self.clusters = []
        for label in np.unique(labels):
            cluster = list(np.array(list(self.DMvalues.columns))[labels == label])
            cluster = [gene.split('---')[0] for gene in cluster]
            self.clusters.append(cluster)

        self.DMvalues = pd.DataFrame(data=np.asarray([np.mean(self.DMvalues.iloc[:,labels==label], axis=1).values
                                                      for label in np.unique(labels)]).T,
                                     index=self.DMvalues.index,
                                     columns=["cluster_%d"%label for label in np.unique(labels)])
        self.temp = self.DMvalues.copy()
        
        gmm = GaussianMixture(n_components=2)
        for i in range(self.config.n_clusters):
            data = list(self.DMvalues.iloc[:,i])
            data.sort()
            data = np.array(data).reshape((-1,1))
            gmm.fit(data)
            pred = gmm.predict(data)
            threshold = data[np.argmax(pred!=pred[0])][0]
            self.DMvalues.iloc[:,i][self.DMvalues.iloc[:,i] >= threshold] = 1
            self.DMvalues.iloc[:,i][self.DMvalues.iloc[:,i] < threshold] = 0
        self.DMvalues = self.DMvalues.apply(lambda x:LabelBinarizer().fit_transform(x)[:,0], axis=0)
        
    def intersect_rows(self, df1, df2):
        common = df1.index & df2.index
        df1 = df1.filter(common, axis=0)
        df2 = df2.filter(common, axis=0)
        return df1, df2

    def remove_unbalanced_genes(self, threshold=0.90):

        for column in self.DMvalues.columns:
            ratio_hyper = np.mean(self.DMvalues.loc[:,column] == 1)
            ratio_null = np.mean(self.DMvalues.loc[:,column] == 0)
            ratio_hypo = np.mean(self.DMvalues.loc[:,column] == -1)
            if ratio_hyper>threshold or ratio_null>threshold or ratio_hypo>threshold:
                del self.DMvalues[column]    


class DatasetML(Dataset):
    
    def __init__(self, config):
        Dataset.__init__(self, config)
        self.get_morpho_context()

    def get_morpho_context(self):
        if self.config.source == "LGGGBM":
            morphoLGG = self.get_morpho("LGG")
            morphoGBM = self.get_morpho("GBM")
            morpho = pd.concat([morphoLGG, morphoGBM], axis=0, join='inner')
        elif self.config.source == "KIRC":
            morpho = self.get_morpho("KIRC")
            morpho.index = [index.replace('.', '-') for index in morpho.index.values]
        else:
            morpho = self.get_morpho(self.config.source)
        context = self.get_context()
        morpho, self.DMvalues = self.intersect_rows(morpho, self.DMvalues)
        morpho, context = self.intersect_rows(morpho, context)
        self.morpho_context = pd.concat([morpho, context], axis=1, join = "inner")
        self.morpho_context = self.morpho_context.apply(self.normalize, axis=0)
        self.morpho_context = self.morpho_context.dropna(axis=1)        
        
    def get_morpho(self, source):

        morpho = pd.read_table("inputs/morpho%s.txt"% source, header = 0)
        morpho["Feature_stat"] = morpho["Feature"]+'_'+morpho["Statistics"]
        morpho.set_index(keys="Feature_stat", drop=True, append=False, inplace=True, verify_integrity=True)
        morpho.drop(labels=["Feature", "Statistics"], axis=1, level=None, inplace=True, errors='raise')
        morpho = morpho.T

        return morpho

    def get_context(self):
        
        if self.config.source == "KIRC":
            context = pd.read_table("inputs/contextKIRC.txt", header = 0, index_col = 0)
            
        else:
            context = pd.read_table("inputs/contextLGGGBM.txt", header = 0, index_col = 0)
        
        context.drop("Unnamed: 9", axis=1, level=None, inplace=True, errors='raise')
        context.sort_values("bin_1", ascending=False, inplace=True)

        return context

    def intersect_rows(self, df1, df2):

        common = df1.index & df2.index
        df1 = df1.filter(common, axis=0)
        df2 = df2.filter(common, axis=0)

        return df1, df2

    def normalize(self, x):
        
        return (x-np.mean(x))/np.std(x)