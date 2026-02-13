import sys
import json
from chord_cnn_lstm import chord_recognition

if __name__ == "__main__":
    audioPath = sys.argv[1]
    output = chord_recognition.main(audioPath)
    print(json.dumps(output))