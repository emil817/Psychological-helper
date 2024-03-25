# Psychological-helper

Чат-бот в Telegram на основе GigaChat от Сбера, для диагностики депрессии, с набором психологических тестов.

Используется модель на SciKit-learn для определения параметров психического здоровья, база данных реализована на SQLite с помощью SQLAlchemy.
Алгоритм работы:
* При получении сообщения текст сообщения анализируется моделью на SciKit-learn.
* Если определяется, что есть нарушения, происходит расчёт косинусных расстояний между векторизованными сообщениями пользователя и списками слов, характерных для агрессии, депрессии и тревожности, для того, чтобы предложить конкретный тест.
* Происходит запрос к GigaChat Api для генерации ответа, затем ответ отправляется пользователю.

На данный момент в боте есть психологические тесты:
* Шкала депрессии А. Бека;
* Шкала тревоги А. Бека;
* Опросник «Ауто- и гетероагрессия» (Е. П. Ильин);

Для запуска необходим файл Tokens.py, содержащий токены для подключения к Telegram-боту и к GigaChat API.
