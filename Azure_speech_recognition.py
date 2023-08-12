'''
Author: MING-CHUNLee mindy80230@gmail.com
Date: 2023-08-06 13:05:34
LastEditors: MING-CHUNLee mindy80230@gmail.com
LastEditTime: 2023-08-12 20:53:35
FilePath: \MediaPipe test\AI_interview\Azure_speech_recognition.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import azure.cognitiveservices.speech as speechsdk

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.speech_recognition_language="zh-TW"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        
        spoken_words = speech_recognition_result.text
        new_spoken_words=spoken_words

        characters_to_remove = ["!", "。", ","] #移除的list

        #移除標點符號
        for char in characters_to_remove:
            new_spoken_words = new_spoken_words.replace(char, "")

        num_words = len(new_spoken_words)
        
        ticks_per_second = speech_recognition_result.duration
        print("Recognized: {}".format(spoken_words))
        print("num_words: {}".format(num_words))
        # 計算每秒鐘平均可以識別的字數
        words_per_second =  (ticks_per_second / 10000000 ) /num_words  # 将刻度轉換為秒

        print("Duration in Ticks: {}".format(words_per_second))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

recognize_from_microphone()