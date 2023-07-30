import speech_recognition as sr
import mediapipe as mp
import cv2
import numpy as np
import threading


def detect_eyes(image):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # 初始化MediaPipe Face Detection模型
    face_detection = mp_face_detection.FaceDetection()

    # 轉換為RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 進行臉部檢測
    results = face_detection.process(image_rgb)

    # 檢測到的臉部數量
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # 繪製臉部檢測框
            cv2.rectangle(image, bbox, (0, 255, 0), 2)

            # 提取眼睛關鍵點
            left_eye_x = int(detection.location_data.relative_keypoints[0].x * iw)
            left_eye_y = int(detection.location_data.relative_keypoints[0].y * ih)
            right_eye_x = int(detection.location_data.relative_keypoints[1].x * iw)
            right_eye_y = int(detection.location_data.relative_keypoints[1].y * ih)

            # 繪製眼睛關鍵點
            cv2.circle(image, (left_eye_x, left_eye_y), 5, (255, 0, 0), -1)
            cv2.circle(image, (right_eye_x, right_eye_y), 5, (255, 0, 0), -1)

            # 在這裡你可以進一步處理眼睛位置的資訊，例如應用Gaze Tracking等

    return image

def speech_recognition_continuous(stop_event):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("請開始說話...")
        while not stop_event.is_set():
            audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio, language="zh-TW")
                print("辨識結果：", text)
                # 計算語速
                words = len(text.split())
                duration = len(audio.frame_data) / audio.sample_rate
                speech_rate = words / (duration / 60)
                print("語速：", speech_rate, "字/分鐘")
                if text == stop_keyword:
                    stop_event.set()

            except sr.UnknownValueError:
                print("無法辨識音訊")
            except sr.RequestError as e:
                print("無法取得語音辨識結果；錯誤訊息：", str(e))



def run_pose_estimation(stop_event):
    import mediapipe as mp
    import cv2
    import numpy as np

    #選擇第一隻攝影機
    cap = cv2.VideoCapture(0)
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    count=0

    # 定義姿勢類型與所對應的索引
    POSE_TYPES = {
        'hand_on_hip': 0,
        'hand_above_head': 1,
        'hand_on_head': 2,
        'hand_on_neck': 3,
        'hand_below_chest': 4,
        'hand_on_chest': 5,
        'hand_crossed_chest':6
    }

    record=np.zeros([2,7], dtype=object)
    record[0][0]='hand_on_hip'
    record[0][1]='hand_above_head'
    record[0][2]='hand_on_head'
    record[0][3]='hand_on_neck'
    record[0][4]='hand_below_chest'
    record[0][5]='hand_on_chest'
    record[0][6]='hand_crossed_chest'
# 載入模型
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while not stop_event.is_set():
            ret, image = cap.read()
            if not ret:
                break
            
            image.flags.writeable = False
            # 轉換為 RGB 格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 得到結果
            pose_results = pose.process(image)

            # 畫出關鍵點
            if pose_results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 取得關鍵點座標
                landmarks = pose_results.pose_landmarks.landmark

                # 將座標轉換為 numpy array 格式
                data = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

                # 取得需要的部分座標
                r_shoulder = data[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_elbow = data[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wrist = data[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # 計算手臂的角度
                vec_1 = np.array([r_shoulder[0] - r_elbow[0], r_shoulder[1] - r_elbow[1]])
                vec_2 = np.array([r_wrist[0] - r_elbow[0], r_wrist[1] - r_elbow[1]])
                cos = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
                angle = np.arccos(cos) * 180 / np.pi



                # 判斷姿勢類型
                if angle < 90:
                    pose_type = 'hand_on_hip'
                elif angle < 140:
                    pose_type = 'hand_above_head'
                elif angle < 160:
                    pose_type = 'hand_on_head'
                elif angle < 180:
                    pose_type = 'hand_on_neck'
                elif angle < 200:
                    pose_type = 'hand_below_chest'
                else:
                    pose_type = 'hand_on_chest'

                # 檢測雙手交叉於胸前的動作
                left_wrist = data[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = data[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # 判斷雙手是否交叉於胸前
                if left_wrist[0] < r_shoulder[0] and right_wrist[0] > r_shoulder[0]:
                    pose_type = 'hand_crossed_chest'

                record[1,POSE_TYPES[pose_type]] += 1

            # 顯示影像
            image = detect_eyes(image)
            cv2.imshow('MediaPipe Pose', image)
            # cv2.imwrite("frame%d.jpg" % count, image) 
            ret,image = cap.read()
            # count += 1
            if cv2.waitKey(1) & 0xFF == ord('q') or stop_event.is_set():
                break

        cap.release()
        print(record)
        cv2.destroyAllWindows()

# 主程式
stop_keyword = "停止"

# 建立停止事件
stop_event = threading.Event()

# 建立語音辨識的執行緒
speech_thread = threading.Thread(target=speech_recognition_continuous, args=(stop_event,))
speech_thread.start()

# 執行肢體辨識
run_pose_estimation(stop_event)

# 等待語音辨識執行緒結束
speech_thread.join()
