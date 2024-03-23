import numpy as np
from gigachat import GigaChat
from telebot import TeleBot, types
import requests
import os
import uuid
from joblib import load
import warnings

from Quastionnaries import Questionnaire_agression, Questionnaire_anxiety, Questionnaire_depression
from Quastionnaries import Calculate_func, Questionnaire, Agression_list, Anxiety_list, Depression_list
from TTS import convertTTS, VoiceMessage
from Tokens import TelebotToken, GigaChatToken
from DB import SQL_DB, User

warnings.filterwarnings("ignore")

vectoriser = load('Vectoriser.joblib')
clf2 = load('Model.joblib')
count_vectoriser = load('CountVectoriser.joblib')

def analyse_message(text: str) -> np.int64:
    vectorised = vectoriser.transform([text])
    pred = clf2.predict(vectorised)
    return pred[0]

if not os.path.exists("content"):
    os.makedirs("content")

token = TelebotToken
giga = GigaChat(credentials=GigaChatToken, scope="GIGACHAT_API_PERS", verify_ssl_certs=False)

db = SQL_DB()

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
    user_data = db.get_user(str(Message.chat.id))
    if user_data is None:
        db.insert_user(User(tg_id=str(Message.chat.id), count_depressed_messages=0, messages_history="", test=0))
        user_data = User(tg_id=str(Message.chat.id), count_depressed_messages=0, messages_history="", test=0)
    
    if Message.text == "Тест на агрессию" or Message.text == "Тест на тревожность" or Message.text == "Тест на депрессию":
        if Message.text == "Тест на агрессию":
            user_data.test = 0
            Test_obj[str(Message.chat.id)] = Questionnaire(Questionnaire_agression)
            bot.send_message(Message.chat.id, "Сейчас я буду скидывать ряд положений, касающихся Вашего поведения. Если они соответствуют имеющейся у Вас тенденции реагировать именно так нажимайте 'Да', если нет, то - 'Нет'")
        elif Message.text == "Тест на тревожность":
            user_data.test = 1
            Test_obj[str(Message.chat.id)] = Questionnaire(Questionnaire_anxiety)
            bot.send_message(Message.chat.id, "Сейчас я буду скидывать список общих симптомов тревоги. Пожалуйста, прочтите внимательно описание симптома и отметьте, насколько сильно он вас беспокоил в течение последней недели, включая сегодняшний день, по шкале:\n0 - Совсем не беспокоит\n1 - Слегка. Не слишком меня беспокоит\n2 - Умеренно. Это было неприятно, но я могу это перенести\n3 - Очень сильно. Я с трудом могу это переносить.")
        elif Message.text == "Тест на депрессию":
            user_data.test = 2
            Test_obj[str(Message.chat.id)] = Questionnaire(Questionnaire_depression)
            bot.send_message(Message.chat.id, "Сейчас я буду скидывать группы утверждений. Внимательно прочитайте каждую группу утверждений. Затем определите в каждой группе одно утверждение, которое лучше всего соответствует тому, как Вы себя чувствовали НА ЭТОЙ НЕДЕЛЕ И СЕГОДНЯ. Нажмите на кнопку с номером этого утверждения. Прежде, чем сделать свой выбор, убедитесь, что Вы прочли все утверждения в каждой группе.")
        text = Test_obj[str(Message.chat.id)].first_question()
        bot.send_message(Message.chat.id, text, reply_markup=Marcups[user_data.test])
    else:
        pred = analyse_message(Message.text)
        
        if pred == 1:
            user_data.count_depressed_messages += 1
            print(Message.chat.id, user_data.count_depressed_messages)

        answer = ""
        
        if user_data.messages_history != "":
            if user_data.count_depressed_messages % 3 == 0 and user_data.count_depressed_messages != 0 and pred == 1:
                answer = "Вы как-то грустно пишите, советую вам обратиться к психологу. "
                response = giga.chat(f"Подбодри человека, который грустит, как будто ты мужчина: {Message.text}")
                answer += response.choices[0].message.content.replace('''"''', '')
                bot.send_message(Message.chat.id, answer)

                user_data.test = find_nearest_theme(Message.text)
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

        user_data.messages_history += Message.text + ";" + answer + ";"
    
    db.update_user(user_data)

# Callback for tests

@bot.callback_query_handler(func=lambda call:True)
def callback(call):
    global test_pos, Test_answ
    if call.message:
        user_data = db.get_user(str(call.message.chat.id))

        if call.data == '1_start_test':
            if user_data.test == 0:
                Test_obj[str(call.message.chat.id)] = Questionnaire(Questionnaire_agression)
                bot.send_message(call.message.chat.id, "Сейчас я буду скидывать ряд положений, касающихся Вашего поведения. Если они соответствуют имеющейся у Вас тенденции реагировать именно так нажимайте 'Да', если нет, то - 'Нет'")
            elif user_data.test == 1:
                Test_obj[str(call.message.chat.id)] = Questionnaire(Questionnaire_anxiety)
                bot.send_message(call.message.chat.id, "Сейчас я буду скидывать список общих симптомов тревоги. Пожалуйста, прочтите внимательно описание симптома и отметьте, насколько сильно он вас беспокоил в течение последней недели, включая сегодняшний день, по шкале:\n0 - Совсем не беспокоит\n1 - Слегка. Не слишком меня беспокоит\n2 - Умеренно. Это было неприятно, но я могу это перенести\n3 - Очень сильно. Я с трудом могу это переносить.")
            elif user_data.test == 2:
                Test_obj[str(call.message.chat.id)] = Questionnaire(Questionnaire_depression)
                bot.send_message(call.message.chat.id, "Сейчас я буду скидывать группы утверждений. Внимательно прочитайте каждую группу утверждений. Затем определите в каждой группе одно утверждение, которое лучше всего соответствует тому, как Вы себя чувствовали НА ЭТОЙ НЕДЕЛЕ И СЕГОДНЯ. Нажмите на кнопку с номером этого утверждения. Прежде, чем сделать свой выбор, убедитесь, что Вы прочли все утверждения в каждой группе.")
            
            text = Test_obj[str(call.message.chat.id)].first_question()
            bot.send_message(call.message.chat.id, text, reply_markup=Marcups[user_data.test])
        
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
