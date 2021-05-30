import cv2
import sys
import numpy as np
import face_recognition
import screeninfo
import random
import time 
from scipy.ndimage.interpolation import shift

# pislantas kepek
feligcsukva = cv2.resize(cv2.imread("feligcsukva.png",0), (480, 480))
#feligcsukva = np.hstack([feligcsukva, feligcsukva])
felignyitva = cv2.resize(cv2.imread("felignyitva.png",0), (480, 480))
#felignyitva = np.hstack([felignyitva, felignyitva])
csukva = cv2.resize(cv2.imread("csukva.png",0), (480, 480))
#csukva = np.hstack([csukva, csukva])

pupil = cv2.resize(cv2.imread("pupil4.png"), (480, 480))

# meret szorzo
multi = 2

# pislogas
blink_frec = 150

# video beolvasas
video_capture = cv2.VideoCapture(0)

# szem pozicio
x_list = []
y_list = []
no_face = 0
blink = random.randint(20,30)
blink_seq = [felignyitva, feligcsukva, csukva, csukva, feligcsukva, felignyitva]

# megjeleníto kepernyo
screen_id = 1
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height
print(width, height)

window_name = 'szem_1'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

window_name_2 = 'szem_2'
cv2.namedWindow(window_name_2, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name_2, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name_2, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.moveWindow(window_name_2, 480,0)


while True:
    # alap kep
    img = np.zeros((width, height, 3))+255
    
    # kep olvasas
    ret, frame = video_capture.read()
    width_cap, height_cap, tmp = frame.shape
    
    small_frame = cv2.resize(frame, (0, 0), fx=1/multi, fy=1/multi)
    rgb_small_frame = small_frame[:, :, ::-1]
    faces = face_recognition.face_locations(rgb_small_frame)
    face_size = []
    
    for (top, right, bottom, left) in faces:
        top *= multi
        right *= multi
        bottom *= multi
        left *= multi
        face_size.append(bottom-top)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
    if len(face_size)>0:
        top, right, bottom, left = faces[np.argmax(face_size)]
        cv2.rectangle(frame, (left*multi, top*multi), (right*multi, bottom*multi), (255, 255, 255), 2)
        x_list.append(left+right)
        y_list.append(top+bottom)
        ######
        #image = np.roll(np.roll(pupil, int(140-280*(np.mean(x_list))/width_cap)+50, axis = 1), int(-140+280*(np.mean(y_list))/height_cap)+50, axis = 0)
        #image = np.hstack([image, image])
        ######
        if len(x_list)>10:
            x_list.pop(0)
            y_list.pop(0)
    else:    
        no_face+=1
        ###############
        #image = shift(pupil, (0,0,0), cval=255) #cv2.circle(img,(411,411), 100, (0,0,0), -1)
        #image = np.hstack([image, image])
        ###############
        if no_face > 50:
            x_list.append((width_cap+11)//2)
            y_list.append((height_cap-125)//2)
            x_list.pop(0)
            y_list.pop(0)
            #print(x_list)
            
    try:
        image = np.roll(np.roll(pupil, int(70-140*(np.mean(x_list))/width_cap)+25, axis = 1), int(-70+140*(np.mean(y_list))/height_cap)+25, axis = 0)
        #image = np.hstack([image, image])
        #cv2.circle(img,(600*(int(640-np.mean(x_list)))//width_cap-50, 600*(int(np.mean(y_list)))//height_cap+250), 100, (0,0,0), -1)
    except ValueError:
        image = shift(pupil, (0,0,0), cval=255) #cv2.circle(img,(411,411), 100, (0,0,0), -1)
        #image = np.hstack([image, image])
        print("Nincs arc....!")
        
    # debughoz
    #cv2.imshow('Video', frame)
    
    print(blink)
    blink -=1
    if blink < 1:
        print(image.shape, blink_seq[-blink].shape)
        image = cv2.bitwise_and(image,image,mask = blink_seq[-blink])
        time.sleep(0.05)
        if blink < -4:
            print("blink")
            blink = random.randint(blink_frec,blink_frec+50)
        
    cv2.imshow(window_name, image)
    cv2.imshow(window_name_2, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
