import re
import os
import speech_recognition as sr
from config import config
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the desired log level (e.g., INFO, DEBUG, WARNING)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        # Add other handlers (e.g., FileHandler) if needed
    ],
)

logger = logging.getLogger("my_app")
# set logger file to config path if exists
if os.path.exists(config["LOG_FILE_PATH"]):
    logger.addHandler(logging.FileHandler(config["LOG_FILE_PATH"]))

if config['USE_FASTER_WHISPER']:
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en")

    def transcribe_audio(file_path):
        segments, _ = model.transcribe(file_path)
        segments = list(segments)  # The transcription will actually run here.
        transcript = " ".join([x.text for x in segments]).strip()
        return transcript


else:
    from assistanttools.transcribe_gguf import transcribe_gguf

    def transcribe_audio(file_path):
        return transcribe_gguf(whisper_cpp_path=config["WHISPER_CPP_PATH"], model_path=config["WHISPER_MODEL_PATH"], file_path=file_path)

def check_if_vision_mode(transcription):
    """
    Check if the transcription is a command to enter vision mode.
    """
    return any([x in transcription.lower() for x in ["photo", "picture", "image", "snap", "shoot"]])


def check_if_exit(transcription):
    """
    Check if the transcription is an exit command.
    """
    return any([x in transcription.lower() for x in ["stop", "exit", "quit"]])


def check_if_ignore(transcription):
    """
    Check if the transcription should be ignored. 
    This happens if the whisper prediction is "you" or "." or "", or is some sound effect like wind blowing, usually inside parentheses.
    These are things caused by having the fan so close to the microphone, definitely need to fix.
    """
    if transcription.strip().lower() == "you" or transcription.strip() == "." or transcription.strip() == "":
        return True
    if re.match(r"\(.*\)", transcription):
        return True
    return False


def dictate_ollama_stream(stream, early_stopping=False, max_spoken_tokens=250):
    response = ""
    streaming_word = ""
    for i, chunk in enumerate(stream):
        text_chunk = chunk['message']['content']
        streaming_word += text_chunk
        response += text_chunk
        if i > max_spoken_tokens:
            early_stopping = True
            break

        if is_complete_word(text_chunk):
            streaming_word_clean = streaming_word.replace(
                '"', "").replace("\n", " ").replace("'", "").replace("*", "").replace('-', '').replace(':', '').replace('!', '')
            speak(streaming_word_clean)
            streaming_word = ""
    if not early_stopping:
        streaming_word_clean = streaming_word.replace(
            '"', "").replace("\n", " ").replace("'", "").replace("*", "").replace('-', '').replace(':', '').replace('!', '')

        speak(streaming_word_clean)

    return response


def is_complete_word(text_chunk):
    """
    Given the subword outputs from streaming, as these chunks are added together, check if they form a coherent word. If so, return the word.
    """

    if ' ' in text_chunk or all([x not in text_chunk for x in ['a', 'e', 'i', 'o', 'u']]):
        return True
    return False


def remove_parentheses(transcription):
    """
    Remove parentheses and their contents from the transcription.
    """
    return re.sub(r"\(.*\)", "", transcription).strip()

def check_microphone(sounds_path):
    recognizer = sr.Recognizer()
    with sr.Microphone(config["DEVICE_INDEX_MIC"]) as source:
        print("Say something:")
        try:
            audio = recognizer.listen(source, timeout=5)
            # value = recognizer.recognize_whisper(audio)
            # print("You said {}".format(value))
            value = transcribe_audio(
                file_path=f"{sounds_path}audio.wav")
            print("Microphone is working!")
            return True
        except ModuleNotFoundError:
            print("ModuleNotFoundError")
        except sr.WaitTimeoutError:
            print("Timeout: No audio detected.")
        except sr.RequestError as e:
            print(f"Error accessing the microphone: {e}")
        except sr.UnknownValueError:
            print("No audio detected.")
    return False

def speak(text):
    os.system(f"espeak -a {config['SPEECH_VOLUME']} '{text}'")
    logger.info(text)
    # print("speak: ", text)

def log(*text):
    logger.info(*text)
    # print(*text)

def log_error(*text):
    logger.error(*text)
    # print(*text)
