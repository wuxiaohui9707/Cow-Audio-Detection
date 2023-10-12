from pyAudioAnalysis.audioTrainTest import extract_features_and_train, load_model
from pyAudioAnalysis.audioSegmentation import mid_term_file_classification, labels_to_segments
import os

def train_model_and_classify(training_dirs, model_name, classifier_type, test_audio_file, segments_file, mt_win=1, mt_step=0.5, s_win=0.1, s_step=0.05):  # 设置中长期和短期的窗口大小和步长
    # 训练模型
    extract_features_and_train(training_dirs, mt_win, mt_step, s_win, s_step, classifier_type, model_name)

    # 对测试音频文件进行分类
    labels, class_names, _, _ = mid_term_file_classification(
        test_audio_file, 
        model_name, 
        classifier_type, 
        True, 
        segments_file
    )

    # 从保存的模型中加载参数（实际上我们只需要 mt_step）
    _, _, _, _, _, mt_step, _, _, _ = load_model(model_name)

    # 打印合并后的分段信息
    print("\nSegments:")
    segments, classes = labels_to_segments(labels, mt_step)
    for i, segment in enumerate(segments):
        print(f'segment {i} {segment[0]} sec - {segment[1]} sec: {class_names[int(classes[i])]}')

