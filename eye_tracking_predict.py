from PIL import Image, ImageDraw
import random
import cv2
import numpy as np
import face_recognition
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

cam = cv2.VideoCapture(0)
cam.set(3, 1920)
cam.set(4, 1080)

cv2.namedWindow("test")

data_list = []
target_x = []
target_y = []

current_data = {        
    'left_pupil_x' : None,
    'left_pupil_y' : None,
    'right_pupil_x' : None,
    'right_pupil_y' : None,        
    'left_eye_x_min' : None,
    'left_eye_x_max' : None,
    'left_eye_y_min' : None,
    'left_eye_y_max' : None,
    'right_eye_x_min' : None,
    'right_eye_x_max' : None,
    'right_eye_y_min' : None,
    'right_eye_y_max' : None,
    'left_eyebrow_x' : None,
    'left_eyebrow_y' : None,
    'right_eyebrow_x' : None,
    'right_eyebrow_y' : None,
    'nose_bridge_x' : None,
    'nose_bridge_y' : None,
    'nose_tip_x' : None,
    'nose_tip_y' : None
}

new_target = (960, 540)

df = pd.read_csv('eye_data.csv')

feature_data = df.loc[:, 'left_pupil_x' : 'nose_tip_y']

target_data_x = df.loc[:, ['target_x']]
target_data_y = df.loc[:, ['target_y']]

linreg_x = LinearRegression().fit(feature_data, target_data_x)
linreg_y = LinearRegression().fit(feature_data, target_data_y)

file = open('eye_data.csv', 'a+')

pred_list = []
pred_idx = 0

def generate_point():
    return (10 + random.randrange(1920 - 20), 10 + random.randrange(1080 - 20))

def draw_point(frame):
    x, y = new_target
    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 5)
    cv2.line(frame, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 5)
    cv2.line(frame, (x - 20, y + 20), (x + 20, y - 20), (0, 255, 0), 5)    
    return frame

def detect_pupil(frame, which = 'left'):
    # filter light by color
    # roi[:,:,0] = 0
    # roi[:,:,1] = 0
    # roi[:,:,2] = 0

    
    roi = frame.copy()

    threshold_value = 40
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    for i in range(2):
        
        _, threshold = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)        
        if len(contours) > 0:
            cnt = contours[0]
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            # cv2.imshow("Threshold", threshold)
            # cv2.imshow("gray roi", gray_roi)
            # cv2.imshow("Roi", roi)
            return (x, y)
        # cv2.imshow("Threshold", threshold)
        # cv2.imshow("gray roi", gray_roi)
        # cv2.imshow("Roi", roi)
        print('Increading threshold value:')
        print(threshold_value, end=' -> ')
        threshold_value += 10
        print(threshold_value)
    return None

def cut_img_detect_pupil(frame, location, which = 'left'):
    current_data[which + '_eye_x_min'] = location[0]
    current_data[which + '_eye_x_max'] = location[1]
    current_data[which + '_eye_y_min'] = location[2]
    current_data[which + '_eye_y_max'] = location[3]
    frame = frame.copy()
    fix_value = 20    
    location[0] -= fix_value
    location[1] -= fix_value
    location[2] += fix_value
    location[3] += fix_value    
    pupil_mark = detect_pupil(frame[location[1]:location[3], location[0]:location[2]], which)
    if pupil_mark:
        pupil_position = (pupil_mark[0] + location[0], pupil_mark[1] + location[1])
        mark_eyes(frame, location)
        return pupil_position
    return None

def locate_face_landmarks(face_landmarks, name):
    location = [-1, -1, -1, -1]    
    for i in face_landmarks[name]:        
        if location[0] < 0 or i[0] < location[0]:
            location[0] = i[0]            
        if location[1] < 0 or i[1] < location[1]:
            location[1] = i[1]            
        if location[2] < 0 or i[0] > location[2]:
            location[2] = i[0]            
        if location[3] < 0 or i[1] > location[3]:
            location[3] = i[1]
    for i in range(len(location)):
        location[i] *= 2
    return location

def mark_point(frame, x, y):
    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), 2)
    cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
    cv2.line(frame, (x - 10, y + 10), (x + 10, y - 10), (0, 255, 0), 2)
    return frame

def mark_eyes(frame, eye):
    marked_image = Image.fromarray(frame)
    d = ImageDraw.Draw(marked_image, 'RGBA')
    d.line([(eye[0], eye[1]), (eye[0], eye[3]), (eye[2], eye[3]), (eye[2], eye[1]), (eye[0], eye[1])], fill=(0, 0, 0, 110), width=2)
    return np.asarray(marked_image)

def log_eyes(left_eye, right_eye):
    current_data = {
        'left_eye_x_min' : left_eye[0],
        'left_eye_x_max' : left_eye[1],
        'left_eye_y_min' : left_eye[2],
        'left_eye_y_max' : left_eye[3],
        'right_eye_x_min' : right_eye[0],
        'right_eye_x_max' : right_eye[1],
        'right_eye_y_min' : right_eye[2],
        'right_eye_y_max' : right_eye[3]
    }

def refresh_current_data():
    global current_data
    current_data = {        
        'left_pupil_x' : None,
        'left_pupil_y' : None,
        'right_pupil_x' : None,
        'right_pupil_y' : None,        
        'left_eye_x_min' : None,
        'left_eye_x_max' : None,
        'left_eye_y_min' : None,
        'left_eye_y_max' : None,
        'right_eye_x_min' : None,
        'right_eye_x_max' : None,
        'right_eye_y_min' : None,
        'right_eye_y_max' : None,
        'left_eyebrow_x' : None,
        'left_eyebrow_y' : None,
        'right_eyebrow_x' : None,
        'right_eyebrow_y' : None,
        'nose_bridge_x' : None,
        'nose_bridge_y' : None,
        'nose_tip_x' : None,
        'nose_tip_y' : None
    }

def log_data():    
    global new_target
    for i in current_data:
        if not current_data[i]:
            refresh_current_data()
            print('Missing Data:')
            print(current_data)
            print()
            return
    data_list.append(current_data)
    target_x.append(new_target[0])
    target_y.append(new_target[1])
    new_target = generate_point()
    refresh_current_data()
    print('Data added!')

def log_face(frame):

    refresh_current_data()

    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)[:, :, ::-1]
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")    
    face_landmarks_list = face_recognition.face_landmarks(img, face_locations=face_locations)
    
    if len(face_landmarks_list) < 1:
        return frame

    #print(face_landmarks_list)
    face_landmarks = face_landmarks_list[0]

    left_eye = locate_face_landmarks(face_landmarks, 'left_eye')
    right_eye = locate_face_landmarks(face_landmarks, 'right_eye')

    if len(left_eye) < 1 or len(right_eye) < 1:
        print('Eye(s) not detected.')
        return frame

    left_pupil_position = cut_img_detect_pupil(frame, left_eye, 'left')
    right_pupil_position = cut_img_detect_pupil(frame, right_eye, 'right')

    if left_pupil_position and right_pupil_position:
        current_data['left_pupil_x'] = left_pupil_position[0]
        current_data['left_pupil_y'] = left_pupil_position[1]
        current_data['right_pupil_x'] = right_pupil_position[0]
        current_data['right_pupil_y'] = right_pupil_position[1]

    other_list = ['left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip'] # , 'bottom_lip'
    for name in other_list:
        other_loc = locate_face_landmarks(face_landmarks, name)                
        x = int((other_loc[0] + other_loc[2]) / 2)
        y = int((other_loc[1] + other_loc[3]) / 2)
        current_data[name + '_x'] = x
        current_data[name + '_y'] = y
        frame =  mark_point(frame, x, y)    
    return frame

def warp_perspective_matrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    nums = src.shape[0]
    A = np.zeros((2*nums, 8))
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i,:] = [A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1,:] = [0,0,0,A_i[0], A_i[1],1,-A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 

    A = np.mat(A)
    warpMatrix = A.I * B #Get the a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    

    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #input a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

four_point_collect = []
four_point_target = [[30, 30], [1890, 30], [30, 1050], [1890, 1050]]

point_collect = []

while len(four_point_collect) < 4:

    new_target = four_point_target[len(four_point_collect)]
    
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    frame = log_face(frame)
    
    missing = False
    for i in current_data:
        if not current_data[i]:
            missing = True
    if not missing:
        frame_df = pd.DataFrame([current_data])
        pred_x = linreg_x.predict(frame_df)
        pred_y = linreg_y.predict(frame_df)
        
        if len(pred_list) < 10:
            pred_list.append((pred_x, pred_y))
            new_target = (pred_x, pred_y)
        else: 
            pred_list[pred_idx] = (pred_x, pred_y)
            pred_idx = (1 + pred_idx) % 10
            sum_x = 0
            sum_y = 0
            for i in pred_list:
                sum_x += i[0]
                sum_y += i[1]
            mean_x = int(sum_x / 10)
            mean_y = int(sum_y / 10)
            point_collect = [mean_x, mean_y]
            

    frame = draw_point(frame)
    
    k = cv2.waitKey(1)
    
    if k%256 == 32:
        four_point_collect.append(point_collect)
        print(point_collect)
        
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    cv2.imshow("test", frame)

warp_matrix = warp_perspective_matrix(np.array(four_point_target), np.array(four_point_collect))

print(warp_matrix)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    
    frame = log_face(frame)
    
    missing = False
    for i in current_data:
        if not current_data[i]:
            missing = True
    if not missing:
        frame_df = pd.DataFrame([current_data])
        pred_x = linreg_x.predict(frame_df)
        pred_y = linreg_y.predict(frame_df)
        
        if len(pred_list) < 10:
            pred_list.append((pred_x, pred_y))
            new_target = (pred_x, pred_y)
        else: 
            pred_list[pred_idx] = (pred_x, pred_y)
            pred_idx = (1 + pred_idx) % 10
            sum_x = 0
            sum_y = 0
            for i in pred_list:
                sum_x += i[0]
                sum_y += i[1]
            mean_x = int(sum_x / 10)
            mean_y = int(sum_y / 10)
            
            fixed_point = np.dot(warp_matrix, np.array([[mean_x], [mean_y], [1]]))
            
            new_target = (fixed_point[0], fixed_point[1])            

            print([mean_x, mean_y], '->', new_target)
        

    frame = draw_point(frame)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    cv2.imshow("test", frame)
    
cam.release()
cv2.destroyAllWindows()

