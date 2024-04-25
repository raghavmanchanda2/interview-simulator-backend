from speechbrain.inference.interfaces import foreign_class
import sounddevice as sd
import wavio


if __name__ == "__main__":
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                               pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
    out_prob, score, index, text_lab = classifier.classify_file(
        "1710561371941.wav")
    print(text_lab)
