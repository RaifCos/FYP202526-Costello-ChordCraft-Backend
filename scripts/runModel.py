import json
from chord_cnn_lstm import chord_recognition

def main(audioPath):
    output = chord_recognition.main(audioPath)
    return json.dumps(output)