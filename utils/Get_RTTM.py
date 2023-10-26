from pyannote.core import Annotation

def save_rttm(file, pipeline, output_path):
    timeline = pipeline(file).get_timeline()
    annotation = Annotation()
    for segment in timeline:
        annotation[segment] = "speech"
    with open(output_path, 'w') as f:
        annotation.write_rttm(f)