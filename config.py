config = {
    "SOUNDS_PATH": 'sounds/',
    "WAKE_WORD": ["test", "johhny", "five", "wake"],
    "TIMEOUT": 10,
    # longest amount of time the allow a phrase to continue before stopping the recording
    "PHRASE_TIME_LIMIT": 7,
    "USE_FASTER_WHISPER": False,
    "WHISPER_CPP_PATH": "../whisper.cpp/",
    # "WHISPER_MODEL_PATH": "/home/nkasmanoff/Desktop/whisper.cpp/models/ggml-tiny.en.bin",
    "WHISPER_MODEL_PATH": "../whisper.cpp/models/ggml-tin.bin",
    # ls: cannot access '../whisper.cpp/models/ggml-tiny.en.bin': No such file or directory
    "LLAMA_CPP_PATH": "../llama.cpp/",
    # "MOONDREAM_MMPROJ_PATH": "../moondream-quants/moondream2-mmproj-050824-f16.gguf",
    # "MOONDREAM_MODEL_PATH": "../moondream-quants/moondream2-050824-q8.gguf",
    "VISION_MODEL": None, # detr or moondream or None
    "LOCAL_MODEL": "phi3:instruct",  # better responses, higher latency
    "STORE_CONVERSATIONS": True,  # to save in case we you want to analyze later
    "CONDENSE_MESSAGES": True,  # for faster response time
    # number of messages to keep in memory (odd #s work best)
    "TRAILING_MESSAGE_COUNT": 1,
    "SYSTEM_PROMPT": 'You are Johnny Five, a Raspbery Pi Voice Assistant. Answer questions in only a sentence.',
    "SPEECH_VOLUME": 10, # 1 -> 100,
    "START_WITH_MIC_CHECK": True, # if True, will start with a check to see if the microphone is working
}
