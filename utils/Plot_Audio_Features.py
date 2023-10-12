from pyAudioAnalysis import MidTermFeatures as aF
import os
import numpy as np
import plotly.graph_objs as go 
import plotly

def plot_audio_features(audio_dirs, feature_names, m_win=1, m_step=1, s_win=0.1, s_step=0.05):
    """
    Plot the audio feature scatterplot.

    Parameters:
    audio_dirs (list of str): List of directories containing audio files.
    feature_names (list of str): List of feature names to be drawn.
    m_win (float): medium term window size.
    m_step (float): medium term window step.
    s_win (float): short term window size.
    s_step (float): medium term window setp.
    """
    class_names = [os.path.basename(d) for d in audio_dirs] 
    features = [] 

    # 段级特征提取
    for d in audio_dirs: 
        f, _, fn = aF.directory_feature_extraction(d, m_win, m_step, s_win, s_step) 
        features.append(f)

    print("Feature shapes:", [f.shape for f in features])

    # 根据特征名称创建特征矩阵
    feature_matrices = []
    for feature in features:
        feature_matrix = np.array([feature[:, fn.index(name)] for name in feature_names])
        feature_matrices.append(feature_matrix)

    # 创建散点图
    plots = []
    for i, f_matrix in enumerate(feature_matrices):
        plots.append(go.Scatter(x=f_matrix[0, :], y=f_matrix[1, :], 
                                name=class_names[i], mode='markers'))

    mylayout = go.Layout(xaxis=dict(title=feature_names[0]),
                         yaxis=dict(title=feature_names[1]))
    plotly.offline.iplot(go.Figure(data=plots, layout=mylayout))