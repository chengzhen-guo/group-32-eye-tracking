from PIL import Image, ImageDraw
import random
import cv2
import numpy as np
import face_recognition
from win32api import GetSystemMetrics
from fractions import gcd

print("Width =", GetSystemMetrics(0))
print("Height =", GetSystemMetrics(1))

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

print(cv2.WND_PROP_FULLSCREEN)
print(cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

data_list = []
target_x = []
target_y = []
image_l = []
image_r = []
image_name_l = []
image_name_r = []

current_data = {
    'id': None,
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

new_target = [12, 9]
new_target_direction = 'right'

do_collect = False
animation_continue = False


file = open('eye_data.csv', 'a+')

read_file = open('eye_data.csv', 'r')

data_id = len(read_file.read().split('\n'))

image_left = None
image_right = None

if len(read_file.read()) < 1:
#if len(file.read()) < 1:
    line = ''
    for i in current_data:
        line += i + ','
    line += 'target_x,target_y\n'
    file.write(line)

def generate_point():
    return (10 + random.randrange(width - 20), 10 + random.randrange(height - 20))

def generate_animation_point():
    global new_target_direction, animation_continue, new_target, do_collect

    if not animation_continue:
        return    

    if new_target[0] + 60 >= width:
        print('new_target', new_target)
        new_target_direction = 'left'
    elif new_target_direction == 'left' and new_target[0] - 60 <= 0:
        print('new_target', new_target)
        new_target_direction = 'right'        
        new_target[1] += 60
    
    if new_target_direction == 'right':
        new_target[0] += 60
    elif new_target_direction == 'left':
        new_target[0] -= 60

    animation_continue = False

    if new_target[1] >= height:
        print('Collection Over!')
        do_collect = False
        new_target = [12, 9]

def draw_point(frame):
    x, y = new_target
    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 5)
    cv2.line(frame, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 5)
    cv2.line(frame, (x - 20, y + 20), (x + 20, y - 20), (0, 255, 0), 5)
    return frame

def detect_pupil(frame, which = 'left'):    
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
            return (x, y)
        print('Increading threshold value:')
        print(threshold_value, end=' -> ')
        threshold_value += 10
        print(threshold_value)
    return None

def cut_img_detect_pupil(frame, location, which = 'left'):
    global image_left, image_right
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
        if which == 'left':
            image_left = frame[location[1]:location[3], location[0]:location[2]]
        else:
            image_right = frame[location[1]:location[3], location[0]:location[2]]
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
    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 0), 2)
    cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 0), 2)
    cv2.line(frame, (x - 10, y + 10), (x + 10, y - 10), (255, 255, 0), 2)
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
        'id' : data_id,
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
    global new_target, animation_continue, data_id
    for i in current_data:
        if not current_data[i]:
            refresh_current_data()
            print('Missing Data:')
            print(current_data)
            print()
            return
        
    line = ''
    for j in current_data:
        line += str(current_data[j]) + ','
    line += str(new_target[0]) + ','
    line += str(new_target[1]) + '\n'
    file.write(line)
    cv2.imwrite('eye_images/' + str(data_id) + '_l.jpg', image_left)
    cv2.imwrite('eye_images/' + str(data_id) + '_r.jpg', image_right)
    
    #cv2.imwrite(str(data_id) + '_l', image_left)
    #cv2.imwrite(str(data_id) + '_r', image_right)
    
    target_x.append(new_target[0])
    target_y.append(new_target[1])    
    
    animation_continue = True
    generate_animation_point()

    data_id += 1
    refresh_current_data()
    #print('Data added!')

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

while True:
    ret, frame = cam.read()
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break    
    if k%256 == 32:
        # SPACE pressed record data
        do_collect = not do_collect
        animation_continue = not animation_continue
    
    frame = cv2.flip(frame, 1)

    if do_collect:
        frame = log_face(frame)    
        log_data()
    frame = draw_point(frame)
    cv2.imshow("test", frame)
    
cam.release()
cv2.destroyAllWindows()

print()
print('Recording data...')
'''
for i in range(len(data_list)):
    line = ''
    for j in data_list[i]:
        line += str(data_list[i][j]) + ','
    line += str(target_x[i]) + ','
    line += str(target_y[i]) + '\n'
    file.write(line)
    cv2.imwrite('eye_images/' + image_name_l[i], image_l[i])
    cv2.imwrite('eye_images/' + image_name_r[i], image_r[i])
'''
file.close()

'''
image_l.append(image_left)
image_r.append(image_right)
image_name_l.append(str(data_id) + '_l.jpg')
image_name_r.append(str(data_id) + '_r.jpg')
'''
