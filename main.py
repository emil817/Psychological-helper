import numpy as np
from gigachat import GigaChat
from telebot import TeleBot, types
import requests
import os
import uuid
import json
from joblib import load
import warnings

from Quastionnaries import Questionnaire_agression, Questionnaire_anxiety, Questionnaire_depression
from Quastionnaries import Calculate_func, Questionnaire, Agression_list, Anxiety_list, Depression_list
from TTS import convertTTS, VoiceMessage
from Tokens import TelebotToken, GigaChatToken

warnings.filterwarnings("ignore")

vectoriser = load('Vectoriser.joblib')
clf2 = load('Model.joblib')
count_vectoriser = load('CountVectoriser.joblib')

def analyse_message(text):
    vectorised = vectoriser.transform([text])
    pred = clf2.predict(vectorised)
    return pred[0]

if not os.path.exists("content"):
    os.makedirs("content")

token = TelebotToken
giga = GigaChat(credentials=GigaChatToken, scope="GIGACHAT_API_PERS", verify_ssl_certs=False)

with open("DB.json") as f:
    S = f.read()
DB = json.loads(S)

# DB = [{user: number_of_depressed_messages}, {user: mesaages_history}, {user: test}]
# DB[2] = {user: test};  Tests: 0 - Agression, 1 - Anxiety, 2 - Depression
Test_obj = {}

# Markups for tests
Callbacks = {
    '1_agres': [0, 1], '0_agres': [0, 0],
    '0_anx': [1, 0], '1_anx': [1, 1], '2_anx': [1, 2], '3_anx': [1, 3],
    '0_depress': [2, 0], '1_depress': [2, 1], '2_depress': [2, 2], '3_depress': [2, 3]             
}
markupStart = types.InlineKeyboardMarkup(row_width=2)
markupStart.add(types.InlineKeyboardButton('Да', callback_data='1_start_test'), types.InlineKeyboardButton('Нет', callback_data='0_start_test'))
markupAgression = types.InlineKeyboardMarkup(row_width=2)
markupAgression.add(types.InlineKeyboardButton('Да', callback_data='1_agres'), types.InlineKeyboardButton('Нет', callback_data='0_agres'))
markupAnxiety = types.InlineKeyboardMarkup(row_width=4)
markupAnxiety.add(types.InlineKeyboardButton('0', callback_data='0_anx'), types.InlineKeyboardButton('1', callback_data='1_anx'),
                  types.InlineKeyboardButton('2', callback_data='2_anx'), types.InlineKeyboardButton('3', callback_data='3_anx'))
markupDepression = types.InlineKeyboardMarkup(row_width=4)
markupDepression.add(types.InlineKeyboardButton('0', callback_data='0_depress'), types.InlineKeyboardButton('1', callback_data='1_depress'),
                  types.InlineKeyboardButton('2', callback_data='2_depress'), types.InlineKeyboardButton('3', callback_data='3_depress'))
Marcups = {-1: markupStart, 0: markupAgression, 1: markupAnxiety, 2: markupDepression}


# Starting telegram bot
bot = TeleBot(token)
startKBoard = types.ReplyKeyboardMarkup(row_width=3, resize_keyboard=True)
startKBoard.add(types.KeyboardButton(text="Тест на агрессию"), types.KeyboardButton(text="Тест на тревожность"), types.KeyboardButton(text="Тест на депрессию"))

cx = lambda a, b : round(np.inner(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), 3)

def find_nearest_theme(text):
    agr = cx(count_vectoriser.transform([text]).toarray().tolist()[0], count_vectoriser.transform(Agression_list).toarray().tolist()[0])
    anx = cx(count_vectoriser.transform([text]).toarray().tolist()[0], count_vectoriser.transform(Anxiety_list).toarray().tolist()[0])
    depr = cx(count_vectoriser.transform([text]).toarray().tolist()[0], count_vectoriser.transform(Depression_list).toarray().tolist()[0])

    if agr == max(agr, anx, depr):
        return 0
    elif anx == max(agr, anx, depr):
        return 1
    else:
        return 2


@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Привет, я бот для общения, можешь поговорить со мной', reply_markup=startKBoard)

@bot.message_handler (content_types = ['text'])
def Text (Message):
    if Message.text == "Тест на агрессию" or Message.text == "Тест на тревожность" or Message.text == "Тест на депрессию":
        if Message.text == "Тест на агрессию":
            DB[2][str(Message.chat.id)] = 0
            Test_obj[str(Message.chat.id)] = Questionnaire(Questionnaire_agression)
            bot.send_message(Message.chat.id, "Сейчас я буду скидывать ряд положений, касающихся Вашего поведения. Если они соответствуют имеющейся у Вас тенденции реагировать именно так нажимайте 'Да', если нет, то - 'Нет'")
        elif Message.text == "Тест на тревожность":
            DB[2][str(Message.chat.id)] = 1
            Test_obj[str(Message.chat.id)] = Questionnaire(Questionnaire_anxiety)
            bot.send_message(Message.chat.id, "Сейчас я буду скидывать список общих симптомов тревоги. Пожалуйста, прочтите внимательно описание симптома и отметьте, насколько сильно он вас беспокоил в течение последней недели, включая сегодняшний день, по шкале:\n0 - Совсем не беспокоит\n1 - Слегка. Не слишком меня беспокоит\n2 - Умеренно. Это было неприятно, но я могу это перенести\n3 - Очень сильно. Я с трудом могу это переносить.")
        elif Message.text == "Тест на депрессию":
            DB[2][str(Message.chat.id)] = 2
            Test_obj[str(Message.chat.id)] = Questionnaire(Questionnaire_depression)
            bot.send_message(Message.chat.id, "Сейчас я буду скидывать группы утверждений. Внимательно прочитайте каждую группу утверждений. Затем определите в каждой группе одно утверждение, которое лучше всего соответствует тому, как Вы себя чувствовали НА ЭТОЙ НЕДЕЛЕ И СЕГОДНЯ. Нажмите на кнопку с номером этого утверждения. Прежде, чем сделать свой выбор, убедитесь, что Вы прочли все утверждения в каждой группе.")
        text = Test_obj[str(Message.chat.id)].first_question()
        bot.send_message(Message.chat.id, text, reply_markup=Marcups[DB[2][str(Message.chat.id)]])
        S = json.dumps(DB)
        with open("DB.json", 'w') as f:
            print(S, file=f)
    else:
        pred = analyse_message(Message.text)
        
        if pred == 1:
            if str(Message.chat.id) in DB[0]:
                DB[0][str(Message.chat.id)] += 1
            else:
                DB[0][str(Message.chat.id)] = 1
            print(Message.chat.id, DB[0][str(Message.chat.id)])
        else:
            if not (str(Message.chat.id) in DB[0]):
                DB[0][str(Message.chat.id)] = 0

        answer = ""
        
        if str(Message.chat.id) in DB[1]:
            if DB[0][str(Message.chat.id)] % 3 == 0 and DB[0][str(Message.chat.id)] != 0 and pred == 1:
                answer = "Вы как-то грустно пишите, советую вам обратиться к психологу. "
                response = giga.chat(f"Подбодри человека, который грустит, как будто ты мужчина: {Message.text}")
                answer += response.choices[0].message.content.replace('''"''', '')
                bot.send_message(Message.chat.id, answer)

                DB[2][str(Message.chat.id)] = find_nearest_theme(Message.text)
                bot.send_message(Message.chat.id, 'Не хотите пройти кототкий тест?', reply_markup = Marcups[-1])
                
            else:
                response = giga.chat(f"Ответь позитивно на сообщение, как будто ты мужчина: {Message.text}")
                answer = response.choices[0].message.content.replace('''"''', '')
                #print(response.choices[0].message.content)#
                bot.send_message(Message.chat.id, answer)
        else:
            response = giga.chat(f"Ответь позитивно на сообщение, как будто ты мужчина: {Message.text}")
            answer = response.choices[0].message.content.replace('''"''', '')
            #print(response.choices[0].message.content)#
            bot.send_message(Message.chat.id, answer)

        if str(Message.chat.id) in DB[1]:
            DB[1][str(Message.chat.id)] += [Message.text, answer]
        else:
            DB[1][str(Message.chat.id)] = [Message.text, answer]
        
        S = json.dumps(DB)
        with open("DB.json", 'w') as f:
            print(S, file=f)

# Callback for tests

@bot.callback_query_handler(func=lambda call:True)
def callback(call):
    global test_pos, Test_answ
    if call.message:
        if call.data == '1_start_test':
            if DB[2][str(call.message.chat.id)] == 0:
                Test_obj[str(call.message.chat.id)] = Questionnaire(Questionnaire_agression)
                bot.send_message(call.message.chat.id, "Сейчас я буду скидывать ряд положений, касающихся Вашего поведения. Если они соответствуют имеющейся у Вас тенденции реагировать именно так нажимайте 'Да', если нет, то - 'Нет'")
            elif DB[2][str(call.message.chat.id)] == 1:
                Test_obj[str(call.message.chat.id)] = Questionnaire(Questionnaire_anxiety)
                bot.send_message(call.message.chat.id, "Сейчас я буду скидывать список общих симптомов тревоги. Пожалуйста, прочтите внимательно описание симптома и отметьте, насколько сильно он вас беспокоил в течение последней недели, включая сегодняшний день, по шкале:\n0 - Совсем не беспокоит\n1 - Слегка. Не слишком меня беспокоит\n2 - Умеренно. Это было неприятно, но я могу это перенести\n3 - Очень сильно. Я с трудом могу это переносить.")
            elif DB[2][str(call.message.chat.id)] == 2:
                Test_obj[str(call.message.chat.id)] = Questionnaire(Questionnaire_depression)
                bot.send_message(call.message.chat.id, "Сейчас я буду скидывать группы утверждений. Внимательно прочитайте каждую группу утверждений. Затем определите в каждой группе одно утверждение, которое лучше всего соответствует тому, как Вы себя чувствовали НА ЭТОЙ НЕДЕЛЕ И СЕГОДНЯ. Нажмите на кнопку с номером этого утверждения. Прежде, чем сделать свой выбор, убедитесь, что Вы прочли все утверждения в каждой группе.")
            
            text = Test_obj[str(call.message.chat.id)].first_question()
            bot.send_message(call.message.chat.id, text, reply_markup=Marcups[DB[2][str(call.message.chat.id)]])
        
        elif call.data == '0_start_test':
            bot.send_message(call.message.chat.id, 'Ну ладно')

        elif Callbacks[call.data][0] in range(0, 3):
            text = Test_obj[str(call.message.chat.id)].next_question(Callbacks[call.data][1])
            if text == 0:
                text = Calculate_func[Callbacks[call.data][0]](Test_obj[str(call.message.chat.id)].get_answers())[0]
                bot.send_message(call.message.chat.id, text)
            else:
                bot.send_message(call.message.chat.id, text, reply_markup=Marcups[Callbacks[call.data][0]])

    bot.answer_callback_query(call.id)

@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    filename = str(uuid.uuid4())
    file_name_full="content/"+filename+".ogg"
    file_name_full_converted="content/"+filename+".wav"
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name_full, 'wb') as new_file:
        new_file.write(downloaded_file)
    os.system("ffmpeg -i "+file_name_full+"  "+file_name_full_converted)
    text=convertTTS(file_name_full_converted)
    os.remove(file_name_full)
    os.remove(file_name_full_converted)
    voice_message = VoiceMessage(text[:150], str(message.chat.id))
    Text(voice_message)

@bot.message_handler(content_types=['sticker'])
def voice_processing(message):
    bot.send_message(message.chat.id, "Извините, но я не понимаю стикеры")
    bot.send_sticker(message.chat.id, "CAACAgIAAxkBAAEK6IplcC9lKTwkcJAZGcX2g6oiMFzWDgACGAADwDZPE9b6J7-cahj4MwQ")

while True:
    try:
        bot.polling(none_stop=True, interval=0)
    except requests.exceptions.ReadTimeout:
        continue
