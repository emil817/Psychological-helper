import speech_recognition as sr

#This is file with fuction for converting voice messages to text

def convertTTS(filename):
    recogniser = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_text = recogniser.listen(source)
        try:
            text = recogniser.recognize_google(audio_text, language='ru_RU')
            return text
        except:
            pass

class VoiceMessage():
    def __init__(self, text, id) -> None:
        self.text = text
        self.chat = VoiceMessageChat(id)
class VoiceMessageChat():
    def __init__(self, id) -> None:
        self.id = id