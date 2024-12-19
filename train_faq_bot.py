from transformers import TFAutoModel, AutoTokenizer

# Load the tokenizer
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using PyTorch weights
bert_model = TFAutoModel.from_pretrained(model_name, from_pt=True)

# Example usage
inputs = tokenizer("Hello, how are you?", return_tensors="tf")  # Use 'tf' for TensorFlow
outputs = bert_model(**inputs)
print(outputs)

# Вопросы и ответы
with open('questions.txt', 'r', encoding='utf-8') as q_file:
    questions = q_file.readlines()

# Чтение ответов из файла answers.txt
with open('answers.txt', 'r', encoding='utf-8') as a_file:
    answers = a_file.readlines()

dataset = []
for i in range(len(questions)):
    q_emb = bert_model(tokenizer(questions[i], return_tensors='tf'))[0].numpy().mean(axis=1)
    a_emb = bert_model(tokenizer(answers[i], return_tensors='tf'))[0].numpy().mean(axis=1)
    dataset.append([np.array(q_emb[0]), np.array(a_emb[0])])

# Формирование пар для модели
X, Y = [], []
for i in range(len(dataset)):
    for j in range(len(dataset)):
        X.append(np.concatenate([dataset[i][0], dataset[j][1]]))
        Y.append(1 if i == j else 0)

X = np.array(X)
Y = np.array(Y)

# Обучение модели
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1],)))
model.add(tf.keras.layers.Dense(100, activation='selu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC()])
model.fit(X, Y, epochs=150, class_weight={0: 1, 1: 8})

model.save('faq_chatbot_model.h5')