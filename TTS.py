import speech_recognition as sr

def convertTTS(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text,language='ru_RU')
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