import numpy as np
import pandas as pd
import os
import shutil
from config import Config
from models import ModelML


#print('seed',np.random.get_state()[1][0])
#np.random.seed(2188022949)
config = Config(folder="HC_10_Clusters",source="LGGGBM", clustering= "hierarchical", n_tasks= 1147)
    
variable_name = "classifier"
values = ["lr", "rf", "svm", "nb", "nn"]
models = ["Logistic Regression", "Random Forest", "Support Vector Machines", "Naive Bayes", "Neural Network"]

metrics_summary = []
for value, experiment in zip(values, models):
    print("\n", experiment)
    setattr(config, variable_name, value)
    config.models = experiment
    model = ModelML(config)
    y_scores, y_preds = model.train_predict()
    model.get_metrics(y_scores, y_preds)
    model.plot_ROCs(y_scores)
    model.plot_PRs(y_scores)
    metrics_summary.append(np.mean(model.metrics, axis = 0).values)
metrics_summary = pd.DataFrame(data=np.array(metrics_summary), index=models, columns=model.metrics.columns)
metrics_summary.to_csv("output/%s/%s/metrics_summary" % (config.source,config.folder))
config.fig.savefig("output/%s/%s/curves"%(config.source,config.folder))
print(metrics_summary)