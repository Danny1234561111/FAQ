import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np

# Загрузка модели и токенизатора
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка модели
bert_model = TFAutoModel.from_pretrained(model_name, from_pt=True)

# Загрузка вашей модели
model = tf.keras.models.load_model('faq_chatbot_model.h5')

# Чтение вопросов из файла
with open('questions.txt', 'r', encoding='utf-8') as q_file:
    questions = q_file.readlines()

# Чтение ответов из файла
with open('answers.txt', 'r', encoding='utf-8') as a_file:
    answers = a_file.readlines()


# Функция для получения эмбеддингов
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='tf')
    outputs = bert_model(**inputs)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()


# Создание набора данных с эмбеддингами
dataset = [(get_embedding(q.strip()), get_embedding(a.strip())) for q, a in zip(questions, answers)]

# Взаимодействие с пользователем
while True:
    user_question = input("Введите ваш вопрос (или 'выход' для завершения): ")
    if user_question.lower() == 'выход':
        break

    # Получение эмбеддинга для вопроса пользователя
    q_emb = get_embedding(user_question.strip())

    # Сравнение с эмбеддингами в наборе данных
    p = []
    for i in range(len(dataset)):
        # Объединение эмбеддингов вопроса и ответа
        emb = np.concatenate([q_emb, dataset[i][1]], axis=1)  # Объединение по оси 1 (по колонкам)

        # Прогнозирование
        prediction = model.predict(tf.convert_to_tensor(emb))  # Убираем np.expand_dims
        p.append([i, prediction[0, 0]])

    p = np.array(p)
    answ = np.argmax(p[:, 1])  # Индекс максимального значения
    print(answers[answ].strip())  # Вывод ответа
