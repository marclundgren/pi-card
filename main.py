import sys
import speech_recognition as sr
import librosa
import os
from assistanttools.actions import get_llm_response, message_history, preload_model
import soundfile as sf
import json
import uuid
from assistanttools.utils import check_if_exit, check_if_ignore, check_microphone, log, log_error, speak, transcribe_audio
from config import config

class WakeWordListener:
    def __init__(self,
                 timeout,
                 phrase_time_limit,
                 sounds_path,
                 wake_word,
                 action_engine,
                 whisper_cpp_path,
                 whisper_model_path):

        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.sounds_path = sounds_path
        self.wake_word = wake_word
        self.action_engine = action_engine
        self.whisper_cpp_path = whisper_cpp_path
        self.whisper_model_path = whisper_model_path

    def listen_for_wake_word(self):
        recognizer = sr.Recognizer()
        speak("Hello. I am ready to assist you.")
        while True:
            with sr.Microphone(config["DEVICE_INDEX_MIC"]) as source:
                log("Awaiting wake word...") 

                try:
                    audio = recognizer.listen(
                        source, timeout=self.timeout // 3, phrase_time_limit=self.phrase_time_limit // 2)
                except sr.WaitTimeoutError:
                    continue
                except:
                    # dont crash if there is an error in recognizing audio
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    log_error(f"Error in recognizing audio. Continuing...")
                    continue
                
                try:
                    log("trying to transcribe audio...")
                    value = transcribe_audio(
                    file_path=f"{self.sounds_path}audio.wav")

                    if value:
                        log("You said: {}".format(value))
                except sr.UnknownValueError:
                    log_error("Could not understand audio")
                    continue
                except:
                    # dont crash if there is an error in recognizing audio
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    log_error("Error in recognizing audio. Continuing...")
                    continue


            try:
                log("trying to write to wav file...")
                with open(f"{self.sounds_path}audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                speech, rate = librosa.load(
                    f"{self.sounds_path}audio.wav", sr=16000)
                sf.write(f"{self.sounds_path}audio.wav", speech, rate)

                transcription = transcribe_audio(
                    file_path=f"{self.sounds_path}audio.wav")

                if transcription:
                    log("Transcription: ", transcription)
                else :
                    log("Transcription: No transcription")

                # check if the wake word is detected
                log("check if wake word is detected.... wake words: ", self.wake_word)
                if any(x in transcription.lower() for x in self.wake_word):
                    speak("Yes?")
                    self.action_engine.run_second_listener(timeout=self.timeout, duration=self.phrase_time_limit)
                    
                # display the transription if the wake word is not detected
                else:
                    log("No wake word detected... Transcription: ", transcription)
                    # speak("I am still listening."))

            except sr.UnknownValueError:
                log_error("Could not understand audio")


class ActionEngine:
    def __init__(
            self,
            sounds_path,
            whisper_cpp_path,
            whisper_model_path,
            ollama_model,
            message_history,
            store_conversations,
            vision_model=None):
        self.sounds_path = sounds_path
        self.whisper_cpp_path = whisper_cpp_path
        self.whisper_model_path = whisper_model_path
        self.ollama_model = ollama_model
        self.message_history = message_history
        self.store_conversations = store_conversations
        self.vision_model = vision_model
        self.conversation_id = str(uuid.uuid4())

    def run_second_listener(self, timeout, duration):
        recognizer = sr.Recognizer()
        while True:
            with sr.Microphone(config["DEVICE_INDEX_MIC"]) as source:
                log("Awaiting query...")
                try:
                    audio = recognizer.listen(
                        source, timeout=timeout, phrase_time_limit=duration)
                except sr.WaitTimeoutError:
                    log("Timeout error")
                    continue

            try:
                log("trying to write to wav file...")
                with open(f"{self.sounds_path}command.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                speech, rate = librosa.load(
                    f"{self.sounds_path}command.wav", sr=16000)
                sf.write(f"{self.sounds_path}command.wav", speech, rate)
                
                log("trying to transcribe audio...")
                transcription = transcribe_audio(
                    file_path=f"{self.sounds_path}command.wav")
                log("Transcription: ", transcription)

                if check_if_ignore(transcription):
                    log("Ignore detected")
                    continue

                if check_if_exit(transcription):
                    log("Exit detected")
                    speak("Program stopped. See you later!")
                    # set message history to empty
                    self.message_history = [self.message_history[0]]
                    return
                else:
                    log("Query detected")
                    os.system(f"play -v .1 sounds/notification.wav") # set this from config path
                    _, self.message_history = get_llm_response(
                        transcription, self.message_history, model_name=self.ollama_model)

                # save appended message history to json
                if self.store_conversations:
                    log("Saving conversation to json")
                    with open(f"storage/{self.conversation_id}.json", "w") as f:
                        json.dump(self.message_history, f, indent=4)

            except sr.UnknownValueError:
                log_error("Could not understand audio")


if __name__ == "__main__":
    if config['START_WITH_TRANSCRIPTION_TEST']:
        log("Testing transcription...")
        transcriptionResult = transcribe_audio(file_path=f"/home/marc/Dev/whisper.cpp/samples/jfk.wav")

        if not transcriptionResult:
            log("Transcription test failed!")
            exit()
        log("Transcription test passed!", transcriptionResult)

    if config['START_WITH_MIC_CHECK']:
        result = check_microphone(config["SOUNDS_PATH"])
        if not result:
            log_error("Microphone is not ready. Please check your microphone and try again.")
            exit()
    
    preload_model(config["LOCAL_MODEL"])

    action_engine = ActionEngine(sounds_path=config["SOUNDS_PATH"],
                                 whisper_cpp_path=config["WHISPER_CPP_PATH"],
                                 whisper_model_path=config["WHISPER_MODEL_PATH"],
                                 ollama_model=config["LOCAL_MODEL"],
                                 message_history=message_history,
                                 store_conversations=config["STORE_CONVERSATIONS"],
                                 vision_model=config["VISION_MODEL"])

    wake_word_listener = WakeWordListener(timeout=config["TIMEOUT"],
                                          phrase_time_limit=config["PHRASE_TIME_LIMIT"],
                                          sounds_path=config["SOUNDS_PATH"],
                                          wake_word=config["WAKE_WORD"],
                                          action_engine=action_engine,
                                          whisper_cpp_path=config["WHISPER_CPP_PATH"],
                                          whisper_model_path=config["WHISPER_MODEL_PATH"])

    
    wake_word_listener.listen_for_wake_word()
