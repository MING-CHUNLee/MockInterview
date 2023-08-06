import speech_recognition as sr

def speech_recognition_continuous():
    # 建立辨識器物件
    recognizer = sr.Recognizer()

    # 使用麥克風作為輸入源
    with sr.Microphone() as source:
        print("請開始說話...")
        while True:
            audio = recognizer.listen(source)

            try:
                # 將語音轉換為文字
                text = recognizer.recognize_google(audio, language="zh-TW")  # 使用 Google 語音辨識 API
                print("辨識結果：", text)
                # 計算語速
                words = len(text.split())
                speech_rate = words / (audio.duration / 60)  # 語速以「一分鐘幾個字」表示
                print("語速：", speech_rate, "字/分鐘")

                if text == "停止":
                    break

            except sr.UnknownValueError:
                print("無法辨識音訊")
            except sr.RequestError as e:
                print("無法取得語音辨識結果；錯誤訊息：", str(e))

# 執行連續語音辨識
speech_recognition_continuous()
