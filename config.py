config = {
    "SOUNDS_PATH": 'sounds/',
    "WAKE_WORD": ["rasp", "berry", "barry", "razbear", "brad", "raster", "test", "wake"],
    "TIMEOUT": 10,
    # longest amount of time the allow a phrase to continue before stopping the recording
    "PHRASE_TIME_LIMIT": 7,
    "USE_FASTER_WHISPER": False,
    "WHISPER_CPP_PATH": "../whisper.cpp/",
    "WHISPER_MODEL_PATH": "../whisper.cpp/models/ggml-tiny.bin",
    "LLAMA_CPP_PATH": "../llama.cpp/",
    # "MOONDREAM_MMPROJ_PATH": "../moondream-quants/moondream2-mmproj-050824-f16.gguf",
    # "MOONDREAM_MODEL_PATH": "../moondream-quants/moondream2-050824-q8.gguf",
    "VISION_MODEL": None, # detr or moondream or None
    "LOCAL_MODEL": "phi3:instruct",  # better responses, higher latency
    "STORE_CONVERSATIONS": True,  # to save in case we you want to analyze later
    "CONDENSE_MESSAGES": True,  # for faster response time
    # number of messages to keep in memory (odd #s work best)
    "TRAILING_MESSAGE_COUNT": 1,
    "SYSTEM_PROMPT": 'You are Pi-Card, a Raspbery Pi Voice Assistant. Answer questions in only a sentence.',
    "SPEECH_VOLUME": 10, # 1 -> 100,
    "START_WITH_MIC_CHECK": True, # if True, will start with a check to see if the microphone is working
    "START_WITH_TRANSCRIPTION_TEST": False, # if True, will start with a test to see if the model is working,
    "LOG_FILE_PATH": "./logs/log.txt", # path to log file,
    "DEVICE_INDEX_MIC": 0, # index of the microphone device,
    "OLLAMA_CHAT_STREAM": True, # if True, will use the ollama chat stream
}
