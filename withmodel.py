import os
import numpy as np
import wave
import speech_recognition as sr
import librosa
import soundfile as sf
import nltk
import re
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Загрузка необходимых данных NLTK
nltk.download('punkt')  # Убедитесь, что пункт установлен

# Ваш токен
token = "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"

def get_audio_file_from_directory(directory):
    """Получить первый аудиофайл из указанной директории."""
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(('.mp3', '.wav', '.m4a')):
            return os.path.join(directory, file_name)
    raise FileNotFoundError("В директории нет поддерживаемых аудиофайлов.")

def convert_to_wav(input_path):
    """Проверка расширения файла и конвертация в WAV при необходимости."""
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension != '.wav':
        output_path = os.path.splitext(input_path)[0] + '.wav'
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format='wav')
        print(f"Файл {input_path} сконвертирован в WAV формат: {output_path}")
        return output_path
    else:
        print("Файл уже в формате WAV.")
        return input_path

def convert_to_mono(input_path, output_path):
    """Конвертация аудио в моно."""
    audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)
    if audio_data.ndim > 1:  # Если стерео
        audio_data = audio_data.mean(axis=0)  # Преобразование в моно
    sf.write(output_path, audio_data, sample_rate)

def recognize_speech(audio_path):
    """Распознавание речи в аудиофайле."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text

def split_speakers(text):
    """Разделение текста на реплики двух спикеров."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Разбиваем текст на предложения
    speaker1 = []
    speaker2 = []
    is_speaker1 = True  # Переключаем спикера

    for sentence in sentences:
        if is_speaker1:
            speaker1.append(sentence)
        else:
            speaker2.append(sentence)
        is_speaker1 = not is_speaker1  # Меняем спикера

    return ' '.join(speaker1), ' '.join(speaker2)

# Пути к файлам
file_directory = r'C:\Users\1\Documents\p\file'
output_text_file = r'C:\Users\1\Documents\p\transcriptions.txt'

try:
    # Получаем первый файл в директории
    input_audio = get_audio_file_from_directory(file_directory)

    # Проверка и конвертация в WAV
    input_audio = convert_to_wav(input_audio)

    # Конвертация в моно
    mono_audio = os.path.splitext(input_audio)[0] + '_mono.wav'
    convert_to_mono(input_audio, mono_audio)

    # Распознаем текст для всего моно-аудио
    recognized_text = recognize_speech(mono_audio)
    print(f"Распознанный текст: {recognized_text}")

    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained("kontur-ai/sbert_punc_case_ru")
    model = AutoModelForTokenClassification.from_pretrained("kontur-ai/sbert_punc_case_ru")

    # Создание пайплайна
    pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)

    # Добавление пунктуации в распознанный текст
    punctuated_output = pipe(recognized_text)

    # Извлечение токенов и объединение их с пробелами
    punctuated_text = ' '.join([token['word'] for token in punctuated_output]).replace(' ##', '').strip()
    print(f"Текст с пунктуацией: {punctuated_text}")

    # Разделяем текст на реплики спикеров
    speaker1_text, speaker2_text = split_speakers(punctuated_text)
    print(f"Спикер 1: {speaker1_text}")
    print(f"Спикер 2: {speaker2_text}")

    # Записываем распознанные фразы в файл
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write("Спикер 1: " + speaker1_text + '\n')
        f.write("Спикер 2: " + speaker2_text + '\n')

except FileNotFoundError as e:
    print(f"Ошибка: {e}")
except PermissionError as e:
    print(f"Ошибка: {e}")
except ValueError as e:
    print(f"Ошибка: {e}")
except Exception as e:
    print(f"Произошла ошибка: {e}")