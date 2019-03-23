import matplotlib.pyplot as plt
import pandas
from helpers import clusters


colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']

all_folders = ['clustering', 'ica', 'pca', 'random_forest', 'randomized_projections']
all_folders_name = ['Original', 'ICA', 'PCA', 'Random Forest', 'Randomized Projections']

nn_items = ['', 'param_ica__n_components', 'param_pca__n_components', 'param_filter__n', 'param_rp__n_components'] 

nn_items_cluster = ['param_gmm__n_components', 'param_km__n_clusters'] 


nn_cols = ['mean_fit_time', 'mean_test_score', 'mean_train_score', 'param_NN__alpha', 'param_NN__hidden_layer_sizes', 'params']

def plot_learning_curve(iterations, train_scores, test_scores, title):
#     _, _, test_scores_base = base_curve

    plt.figure()
    plt.title(title)
    plt.ylim((.3, 1.01))
    
    # if datasetNum == 1:
    #     plt.ylim((.55, 1.01))

    plt.xlabel("# Iterations")
    plt.ylabel("Score")
    
    plt.grid()
    
    plt.plot(iterations, train_scores, 'o-', color="r",
             label="Training score")
#     plt.plot(train_sizes, test_scores_base_mean, 'o-', color="b",
#              label="Test Score without CV")
    plt.plot(iterations, test_scores, 'o-', color="g",
             label="Test Score")

    plt.legend(loc="best")
    return plt

def plot_many_curves(iterations, valLabels, valIndex, title, yLabel):
    plt.figure()
    plt.title(title)

    plt.xlabel("# Iterations")
    plt.ylabel(yLabel)
    
    plt.grid()

    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']
    for i in range(len(valLabels)):
        val = valLabels[i][valIndex]
        label = valLabels[i][-1]
        color = colors[i]

        plt.plot(iterations, val, 'o-', color=color,
             label=label)

    plt.legend(loc="best")
    return plt

def plot_timing_curve(iterations, timeDuration, title):
#     _, _, test_scores_base = base_curve

    plt.figure()
    plt.title(title)
    # plt.ylim((.3, 1.01))
    
    # if datasetNum == 1:
    #     plt.ylim((.55, 1.01))

    plt.xlabel("# Iterations")
    plt.ylabel("Training Time (s)")
    
    plt.grid()
    
    plt.plot(iterations, timeDuration, 'o-', color="r",
             label="Duration")

    plt.legend(loc="best")
    return plt

def get_df(path, headerVal=0):
    return pandas.read_csv('../results/' + path, header=headerVal)

def print_latex_row(vals, sig=3):
    sigString = '{:0.' + str(sig) + 'f}'
    vals = ' & '.join(list(map(lambda x: sigString.format(x), vals)))
    print(vals + ' \\\\ \hline')

def save_plot(plot, title):
    print (title.replace(' ', '_').replace('.', 'pt').lower())
    plot.savefig('../analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '.jpg')

def plot_item(df, valName, title, yLabel, lineLabel, xLabel = 'Number of Clusters', latexVal = 3):
    clusters = df[df.columns[0]]
    vals = df[valName]
    print_latex_row(vals, latexVal)

    plt.figure()
    plt.title(title) 

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    plt.grid()
    
    plt.plot(clusters, vals, 'o-', color="r",
             label=lineLabel)

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_sse(folder, valName, title):
    df = get_df(folder + '/SSE.csv')
    plot_item(df, valName, title, 'Inner Cluster Sum of Squared Errors', "Sum of Squared Errors")

def plot_clustering_accuracy(folderFile, title): 
    ami = get_df(folderFile + ' adjMI.csv')
    acc = get_df(folderFile + ' acc.csv')

    clusters = ami.columns[1:].values

    gmm_ami = ami.loc[0].values[1:]
    gmm_acc = acc.loc[0].values[1:]

    k_means_ami = ami.loc[1].values[1:]
    k_means_acc = acc.loc[1].values[1:]

    print_latex_row(k_means_ami)
    print_latex_row(k_means_acc)
    print_latex_row(gmm_ami)
    print_latex_row(gmm_acc)

    newTitle = title + ' Scoring'
    plt.figure()
    plt.title(newTitle) 
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(clusters, k_means_ami, 'o-', color="g", label="k-means Adjusted Mutual Info Score")
    plt.plot(clusters, k_means_acc, 'o-', color="b", label="k-means Cluster Accuracy")

    plt.plot(clusters, gmm_ami, 'o-', color="r", label="Expectation Maximization Adjusted Mutual Info Score")
    plt.plot(clusters, gmm_acc, 'o-', color="black", label="Expectation Maximization Cluster Accuracy")

    plt.legend(loc="best")

    save_plot(plt, newTitle)
    plt.close()

def plot_log_liklihood(folder, valName, title):
    df = get_df(folder + '/logliklihood.csv')
    plot_item(df, valName, title, 'Log Likelihood', 'Log Likelihood', 'Number of Components', 2)

def plot_eigen(folderFile, title):
    df = get_df(folderFile + ' scree.csv', None)

    plt.figure()
    plt.title(title) 

    plt.xlabel('# Components')
    plt.ylabel('Eigenvalues')
    
    plt.grid()
    
    plt.plot(df[0].values, df[1].values, 'o-', color="g",
             label='Eigenvalue')
    plt.plot(df[0].values, df[2].values, 'o-', color="b",
             label='Variance')
    plt.plot(df[0].values, df[3].values, 'o-', color="black",
             label='Cumulative Variance')

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_kurt(folderFile, title):
    df = get_df(folderFile + ' scree.csv', None)

    plt.figure()
    plt.title(title) 

    plt.xlabel('# Components')
    plt.ylabel('Kurtosis')
    
    plt.grid()
    
    plt.plot(df[0].values, df[1].values, 'o-', color="g",
             label='Kurtosis')

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_reconstruction(folderFile, title):
    df = get_df(folderFile + ' scree2.csv')

    plt.figure()
    plt.title(title) 

    plt.xlabel('# Components')
    plt.ylabel('Reconstruction Error')
    
    plt.grid()

    for i in range(len(colors)):
        color = colors[i]
        
        plt.plot(df[df.columns[0]].values, df[df.columns[i + 1]].values, 'o-', color=color,
                 label='Error for Trial ' + str(i + 1))

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()


def plot_pairwise_distance_corr(folderFile, title):
    df = get_df(folderFile + ' scree1.csv')

    plt.figure()
    plt.title(title) 

    plt.xlabel('# Components')
    plt.ylabel('Parise Distance Correlation')
    
    plt.grid()

    for i in range(len(colors)):
        color = colors[i]
        
        plt.plot(df[df.columns[0]].values, df[df.columns[i + 1]].values, 'o-', color=color,
                 label='Correlation for Trial ' + str(i + 1))

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_feature_importances(folderFile, title):
    df = get_df(folderFile + ' scree.csv', None)

    plt.figure()
    plt.title(title) 

    plt.xlabel('Feature #')
    plt.ylabel('Importance')
    
    plt.grid()
    
    plt.plot(df[0].values, df[1].values, 'o-', color="g",
             label='Importance')

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_all_sse(valName, title):
    title = title + ' SSE Comparison after DR'
    plt.figure()
    plt.title(title) 

    plt.xlabel('# of Clusters')
    plt.ylabel('Inner Cluster Sum of Squared Errors')
    
    plt.grid()

    index = 0
    for folder in all_folders:
        line = '*-'
        if index == 0:
            line = 'o-'
        df = get_df(folder + '/SSE.csv')
        color = colors[index]
        name = all_folders_name[index]  
        vals = df[valName]
        plt.plot(clusters, vals, line, color=color,
             label=name + ' SSE')

        index += 1

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_all_log_liklihood(valName, title):
    title = title + ' Log Liklihood Comparison after DR'
    plt.figure()
    plt.title(title) 

    plt.xlabel('# of Clusters')
    plt.ylabel('Log Liklihood')
    
    plt.grid()

    index = 0
    for folder in all_folders:
        line = '*-'
        if index == 0:
            line = 'o-'
        df = get_df(folder + '/logliklihood.csv')
        color = colors[index]
        name = all_folders_name[index]  
        vals = df[valName]
        plt.plot(clusters, vals, line, color=color,
             label=name + ' Log Liklihood')

        index += 1

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_all_scoring(fileName, title, kMeans=True):

    title = title + ' Scoring'
    plt.figure()
    plt.title(title) 

    plt.xlabel('# of Clusters')
    plt.ylabel('Score')
    
    plt.grid()

    index = 0
    for folder in all_folders:
        line = '*-'
        if index == 0:
            line = 'o-'

        color = colors[index]
        # color1 = colors[-1 * index - 1]
        name = all_folders_name[index] 

        ami = get_df(folder + '/' + fileName + ' adjMI.csv')
        acc = get_df(folder + '/' + fileName + ' acc.csv')

        vals1 = ami.loc[0].values[1:]
        vals2 = acc.loc[0].values[1:]

        if kMeans: 
            vals1 = ami.loc[1].values[1:]
            vals2 = acc.loc[1].values[1:]

        plt.plot(clusters, vals1, line, color=color,
             label=name + ' AMI')
        plt.plot(clusters, vals2, line, color=color,
             label=name + ' Accuracy')

        index += 1

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()


def plot_nn(fileBase, extra, xLabel, title):
    title = title + ' Scoring'
    plt.figure()
    plt.title(title) 

    plt.xlabel(xLabel)
    plt.ylabel('Score')
    
    plt.grid()

    index = 0
    for folder in all_folders:
        fileName = fileBase + extra

        if index == 0:
            folder = 'base'
            fileName = fileBase
            index += 1
            continue

        line = '*-'
        if index == 0:
            line = 'o-'

        color = colors[index]

        name = all_folders_name[index] 

        df = get_df(folder + '/' + fileName + '.csv').sort_values('rank_test_score')
        best = df.head(1)
        # print(best)

        best_data = best[nn_cols]

        best_data = map(lambda x: str(x), list(best_data.values[0, :]))
        # print(best_data)
        print(name + ' & ' + ' & '.join(best_data) + ' \\\\ \\hline')

        xVar = nn_items[index]

        nnAlpha = list(best['param_NN__alpha'].values)[0]
        nnLayers = list(best['param_NN__hidden_layer_sizes'].values)[0]

        dfLayers = df['param_NN__hidden_layer_sizes'] == nnLayers
        dfAlpha = df['param_NN__alpha'] == nnAlpha 

        best_series = df[dfLayers & dfAlpha].sort_values(xVar)
        # print(best_series)

        xVals = list(set(df[xVar].values))

        vals = best_series['mean_test_score'].values

        # print(xVals)
        # print(vals)

        plt.plot(xVals, vals, line, color=color,
             label=name + ' NN')

        index += 1

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_nn_cluster(fileBase, extra, xLabel, title):
    print('')
    title = title + ' Scoring'
    plt.figure()
    plt.title(title) 

    plt.xlabel(xLabel)
    plt.ylabel('Score')
    
    plt.grid() 

    df = get_df('clustering/Housing cluster GMM.csv').sort_values('rank_test_score')
    best = df.head(1)
    # print(best)

    best_data = best[nn_cols]

    best_data = map(lambda x: str(x), list(best_data.values[0, :]))
    # print(best_data)
    print('Expectation Maximization' + ' & ' + ' & '.join(best_data) + ' \\\\ \\hline')

    xVar = nn_items_cluster[0]

    nnAlpha = list(best['param_NN__alpha'].values)[0]
    nnLayers = list(best['param_NN__hidden_layer_sizes'].values)[0]

    dfLayers = df['param_NN__hidden_layer_sizes'] == nnLayers
    dfAlpha = df['param_NN__alpha'] == nnAlpha 

    best_series = df[dfLayers & dfAlpha].sort_values(xVar)
    # print(best_series)

    xVals = list(set(df[xVar].values))
    xVals.sort() 

    vals = best_series['mean_test_score'].values

    # print(xVals)
    # print(vals)

    plt.plot(xVals, vals, 'o-', color='g',
         label='Expectation Maximization NN') 



    df = get_df('clustering/Housing cluster Kmeans.csv').sort_values('rank_test_score')
    best = df.head(1)
    # print(best)

    best_data = best[nn_cols]

    best_data = map(lambda x: str(x), list(best_data.values[0, :]))
    # print(best_data)
    print('k-means' + ' & ' + ' & '.join(best_data) + ' \\\\ \\hline')

    xVar = nn_items_cluster[1]

    nnAlpha = list(best['param_NN__alpha'].values)[0]
    nnLayers = list(best['param_NN__hidden_layer_sizes'].values)[0]

    dfLayers = df['param_NN__hidden_layer_sizes'] == nnLayers
    dfAlpha = df['param_NN__alpha'] == nnAlpha 

    best_series = df[dfLayers & dfAlpha].sort_values(xVar)
    # print(best_series)

    xVals = list(set(df[xVar].values))
    xVals.sort()

    vals = best_series['mean_test_score'].values

    # print(xVals)
    # print(vals)

    plt.plot(xVals, vals, 'o-', color='b',
         label='k-Means NN') 

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()


def plot_clustering():
    plot_sse('clustering', 'perm SSE (left)', 'Default Credit Card Payment SSE')
    plot_sse('clustering', 'housing SSE (left)', 'Wine Quality  SSE')

    plot_clustering_accuracy('clustering/Housing', 'Wine Quality  Clustering')
    plot_clustering_accuracy('clustering/Perm', 'Default Credit Card Payment Clustering')

    plot_log_liklihood('clustering', 'housing log-likelihood', 'Wine Quality Logliklihood')
    plot_log_liklihood('clustering', 'perm log-likelihood', 'Default Credit Card Payment Logliklihood')

def plot_pca():
    plot_eigen('pca/housing', 'Wine Quality  PCA Eigenvalues')
    plot_eigen('pca/perm', 'Default Credit Card Payment PCA Eigenvalues')

    plot_sse('pca', 'perm SSE (left)', 'PCA Default Credit Card Payment SSE')
    plot_sse('pca', 'housing SSE (left)', 'PCA Housing SSE')

    plot_clustering_accuracy('pca/Housing', 'PCA Wine Quality  Clustering')
    plot_clustering_accuracy('pca/Perm', 'PCA Default Credit Card Payment Clustering')

    plot_log_liklihood('pca', 'housing log-likelihood', 'PCA Wine Quality  Logliklihood')
    plot_log_liklihood('pca', 'perm log-likelihood', 'PCA Default Credit Card Payment Logliklihood')

def plot_ica():
    plot_kurt('ica/housing', 'Wine Quality  ICA Kurtosis')
    plot_kurt('ica/perm', 'Default Credit Card Payment ICA Kurtosis')

    plot_sse('ica', 'perm SSE (left)', 'ICA Default Credit Card Payment SSE')
    plot_sse('ica', 'housing SSE (left)', 'ICA Wine Quality  SSE')

    plot_clustering_accuracy('ica/Housing', 'ICA Wine Quality  Clustering')
    plot_clustering_accuracy('ica/Perm', 'ICA Default Credit Card Payment Clustering')

    plot_log_liklihood('ica', 'housing log-likelihood', 'ICA Wine Quality  Logliklihood')
    plot_log_liklihood('ica', 'perm log-likelihood', 'ICA Default Credit Card Paymenta Logliklihood')

def plot_rp():
    plot_reconstruction('randomized_projections/housing', 'Wine Quality  Randomized Projections Reconstruction Error')
    plot_reconstruction('randomized_projections/perm', 'Default Credit Card Payment Randomized Projections Reconstruction Error')

    plot_pairwise_distance_corr('randomized_projections/housing', 'Wine Quality  Randomized Projections Pairwise Dist. Corr.')
    plot_pairwise_distance_corr('randomized_projections/perm', 'Default Credit Card Payment Randomized Projections Pairwise Dist. Corr.')

    plot_sse('randomized_projections', 'perm SSE (left)', 'RP Default Credit Card Payment SSE')
    plot_sse('randomized_projections', 'housing SSE (left)', 'RP Wine Quality  SSE')

    plot_clustering_accuracy('randomized_projections/Housing', 'RP Wine Quality  Clustering')
    plot_clustering_accuracy('randomized_projections/Perm', 'RP Default Credit Card Payment Clustering')

    plot_log_liklihood('randomized_projections', 'housing log-likelihood', 'RP Wine Quality  Logliklihood')
    plot_log_liklihood('randomized_projections', 'perm log-likelihood', 'RP Default Credit Card Payment Logliklihood')

def plot_rf():
    plot_feature_importances('random_forest/housing', 'Wine Quality  Random Forests Feature Importances')
    plot_feature_importances('random_forest/perm', 'Default Credit Card Payment Random Forests Feature Importances')

    plot_sse('random_forest', 'perm SSE (left)', 'RF Default Credit Card Payment SSE')
    plot_sse('random_forest', 'housing SSE (left)', 'RF Wine Quality  SSE')

    plot_clustering_accuracy('random_forest/Housing', 'RF Wine Quality  Clustering')
    plot_clustering_accuracy('random_forest/Perm', 'RF Default Credit Card Payment Clustering')

    plot_log_liklihood('random_forest', 'housing log-likelihood', 'RF Wine Quality  Logliklihood')
    plot_log_liklihood('random_forest', 'perm log-likelihood', 'RF Default Credit Card Payment Logliklihood')

def plot_dr():
    plot_pca()
    plot_ica()
    plot_rp()
    plot_rf()


def plot_comparison():
    plot_all_sse('perm SSE (left)', 'Default Credit Card Payment')
    plot_all_sse('housing SSE (left)', 'Wine Quality ')

    plot_all_log_liklihood('perm log-likelihood', 'Default Credit Card Payment')
    plot_all_log_liklihood('housing log-likelihood', 'Wine Quality ')

    plot_all_scoring('housing', 'Wine Quality  k-Means after DR', True)
    plot_all_scoring('housing', 'Wine Quality  Estimation Maximization after DR', False)

    plot_all_scoring('perm', 'Default Credit Card Payment k-Means after DR', True)
    plot_all_scoring('perm', 'Default Credit Card Payment Estimation Maximization after DR', False)

    plot_nn('housing', ' dim red', '# Components', 'Wine Quality  NN after DR') 
    plot_nn_cluster('housing', ' dim red', '# Clusters', 'Wine Quality  NN after Cluster')


plot_clustering()
plot_dr()
plot_comparison()

# plot('NN_OUTPUT/BACKPROP_LOG.txt', 'Backprop NN')

# plot('NN_OUTPUT/RHC_LOG.txt', 'Randomized Hill Climbing NN')

# plot('NN_OUTPUT/SA0.15_LOG.txt', 'Simulated Annealing Cooling .15 NN')
# plot('NN_OUTPUT/SA0.35_LOG.txt', 'Simulated Annealing Cooling .35 NN')
# plot('NN_OUTPUT/SA0.55_LOG.txt', 'Simulated Annealing Cooling .55 NN')
# plot('NN_OUTPUT/SA0.7_LOG.txt', 'Simulated Annealing Cooling .7 NN')
# plot('NN_OUTPUT/SA0.95_LOG.txt', 'Simulated Annealing Cooling .95 NN')

# plot('NN_OUTPUT/GA_50_10_10_LOG.txt', 'Genetic 10 Mate, 10 Mutate NN')
# plot('NN_OUTPUT/GA_50_10_20_LOG.txt', 'Genetic 10 Mate, 20 Mutate NN')
# plot('NN_OUTPUT/GA_50_20_10_LOG.txt', 'Genetic 20 Mate, 10 Mutate NN')
# plot('NN_OUTPUT/GA_50_20_20_LOG.txt', 'Genetic 20 Mate, 20 Mutate NN')

# plot_peaks('Continous Peaks')
# plot_tsp('Traveling Salesman')
# plot_flipflop('Flip Flop')


# iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,elapsed
# 0,0.293247932778,0.292920659286,0.300509176221,0.413502881713,0.414196242171,0.398981670061,0.588158795
# 10,0.068612203824,0.0699373695198,0.0645621181263,0.862775592352,0.86012526096,0.870875763747,7.978658689
# 20,0.0686569920033,0.0699354731105,0.0641716006192,0.862684109414,0.86012526096,0.871690427699,15.104390501
