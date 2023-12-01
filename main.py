from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
# 대화 데이터를 저장할 리스트
conversation_data = []
app = Flask(__name__)

# 대화 데이터를 실시간으로 생성하고 학습에 사용하는 함수
def generate_conversation():
    while True:
        user_input = input("사용자: ")
        conversation_data.append(user_input)  # 사용자 입력 저장
        
        # 챗봇의 응답을 생성하는 로직 작성
        # 미리 정의된 응답
        responses = {
            "안녕": "안녕하세요!",
            "어떻게 지내?": "저는 항상 좋습니다!",
            "무슨 일을 하나요?": "저는 AI 챗봇입니다. 도움을 드릴까요?",
            "안녕히 계세요": "잘 가세요! 다음에 또 만나요."
        }

        def generate_response(user_input):
            # 사용자 입력을 처리(예: 소문자로 변환, 공백 제거 등)
            user_input = user_input.strip().lower()

            # 정의된 응답 중에서 사용자 입력과 일치하는 것이 있는지 확인
            if user_input in responses:
                return responses[user_input]
            else:
                return "죄송합니다. 제가 이해하지 못했습니다."

        while True:
            message = input("User: ")
            print("Chatbot: " + generate_response(message))
def train_model():    
        # 대화 데이터를 학습하는 함수
        questions, answers = zip(*generate_conversation)


        # 텍스트를 숫자로 변환하기 위한 토크나이저
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(questions + answers)
##
        # 질문과 응답을 숫자로 변환
        questions_seq = tokenizer.texts_to_sequences(questions)
        answers_seq = tokenizer.texts_to_sequences(answers)

        # 패딩 처리
        questions_seq = pad_sequences(questions_seq, padding='post')
        answers_seq = pad_sequences(answers_seq, padding='post')

        # 모델 정의
        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index)+1, 128))
        model.add(LSTM(128))
        model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        model.fit(questions_seq, np.expand_dims(answers_seq, -1), epochs=100)
# 실시간 대화 데이터 생성 및 모델 학습 시작
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    conversation_data.append(user_input)

    response = generate_response(user_input)

    return render_template('index.html', user_input=user_input, response=response)

def generate_response(user_input):
    # 응답 생성 로직은 여기에 위치합니다.
    return "챗봇: " + response
generate_conversation()
train_model()
