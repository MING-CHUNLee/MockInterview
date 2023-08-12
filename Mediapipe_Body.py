import mediapipe as mp
import cv2
import numpy as np

def run_pose_estimation():
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

    no_detection_count = 0
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
        while True:
            # 讀取影像
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

            # 如果沒有畫面標記
            if pose_results.pose_landmarks is None:
                    no_detection_count += 1

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
                print('pose_type:', pose_type)
                

            # 顯示影像
            cv2.imshow('MediaPipe Pose', image)
            # cv2.imwrite("frame%d.jpg" % count, image) 
            ret,image = cap.read()
            count += 1
            if cv2.waitKey(1) & 0xFF  == ord('q'):
                break

        cap.release()
        print(record)
        print('no_detection_count ',no_detection_count )
        cv2.destroyAllWindows()

# 主程式
run_pose_estimation()
