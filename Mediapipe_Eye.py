import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np

# 變數
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0

# 常數
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# 面部輪廓的索引
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# 嘴唇的索引
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# 左眼的索引
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# 右眼的索引
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh

# 創建攝影機物件
camera = cv.VideoCapture(0)

# 面部關鍵點檢測函數
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # 列表[(x, y), (x, y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # 返回每個面部關鍵點的坐標列表
    return mesh_coord

# 歐幾里得距離
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

# 眨眼比例
def blinkRatio(img, landmarks, right_indices, left_indices):
    # 右眼
    # 水平線
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # 垂直線
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # 左眼
    # 水平線
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # 垂直線
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

# 提取眼睛圖像的函數
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # 將彩色圖像轉換為灰度圖像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 獲得圖像的尺寸
    dim = gray.shape

    # 創建遮罩
    mask = np.zeros(dim, dtype=np.uint8)

    # 在掩膜上繪製眼睛的形狀（用白色表示）
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # 在遮罩上繪製眼睛圖像，保留眼睛部分，其餘部分置為灰色
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155

    # 獲得右眼和左眼的最小和最大x和y值
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # 從掩膜中裁剪出眼睛部分
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # 返回裁剪後的眼睛圖像
    return cropped_right, cropped_left

# 眼睛位置估計函數
def positionEstimator(cropped_eye):
    h, w = cropped_eye.shape
    
    # 去除圖像中的噪點
    gaussian_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussian_blur, 3)

    # 將圖像二值化
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # 將眼睛分成三個部分
    piece = int(w / 3)
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # 計算每個部分中的黑色像素數量
    right_part = np.sum(right_piece == 0)
    center_part = np.sum(center_piece == 0)
    left_part = np.sum(left_piece == 0)

    # 將數量最多的部分標記為眼睛位置
    eye_parts = [right_part, center_part, left_part]
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = '' 
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]

    return pos_eye, color

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # 開始計時
    start_time = time.time()
    # 開始視頻迴圈
    while True:
        frame_counter += 1  # 計算幀數
        ret, frame = camera.read()  # 從攝影機獲得幀
        if not ret:
            break  # 無更多幀則退出
        # 調整幀的大小
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            if ratio > 5.5:
                CEF_COUNTER += 1
                utils.colorBackgroundText(frame,  f"Blinks", FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6)

            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
            utils.colorBackgroundText(frame,  f"Total Blinks: {TOTAL_BLINKS}", FONTS, 0.7, (30, 150), 2)

            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()
