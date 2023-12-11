import os
from utils.CowAudioAnalyzer import CowAudioAnalyzer
#initialization CowAudioAnalyzer
database_path = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_combination', 'yaml', 'My_Databases.yml')
output_dir = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_combination', 'output')
analyzer = CowAudioAnalyzer(database_path=database_path, output_dir=output_dir, model_name="pyannote/segmentation")

#load dataset
analyzer.load_dataset()

#load model
classes = ["rumination", "Breath", "Burp"]
analyzer.load_pretrained_model(classes)

# set task
analyzer.setup_task(duration=1.5, batch_size=32)

#train model
analyzer.train_model(max_epochs=100)

#save model
model_save_path = os.path.join('/mnt/', 'e', 'Files', 'Acoustic_Data', 'Datasets_combination', 'model', 'model.pth')
analyzer.save_model(model_save_path)

# For inference and segmentation on a test file
test_iter = analyzer.protocol.test()
first_audio_file = next(test_iter)

inference_results = analyzer.run_inference(first_audio_file)
inference_results

analyzer.save_probabilities(first_audio_file, 60, 0, 60, 1.5, 1, 'probability_1')

initial_params = {
    "thresholds": {
        "rumination": {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.0, "min_duration_off": 0.0},
        "Breath": {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.0, "min_duration_off": 0.0},
        #"Grazing": {"onset": 0.5, "offset": 0.4, "min_duration_on":0.0, "min_duration_off": 0.0},
        "Burp": {"onset": 0.7, "offset": 0.6, "min_duration_on": 0.0, "min_duration_off": 0.0},
        #"gulp": {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.005, "min_duration_off": 0.005},
    }
}
analyzer.setup_pipeline(initial_params)
segmentation_results = analyzer.run_segmentation(first_audio_file)
segmentation_results