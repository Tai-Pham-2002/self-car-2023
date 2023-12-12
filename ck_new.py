# Import socket module
import socket
import cv2
import numpy as np
import torch
import time
import json
import base64
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2

import numpy as np
import matplotlib.patches as patches
# Load a model
model = YOLO('update_weght_seg.pt')  # pretrained YOLOv8n model
model_detect = YOLO('sign_best.pt')

global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0


# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed

#-----------------PID Controller-------------------#
error_arr = np.zeros(5)
pre_t = time.time()
MAX_SPEED = 60

def PID(error, p, i, d):
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)

if __name__ == "__main__":
    try:
        cls = ''
        count = 0
        coor_thres = 0
        momentum_flag = False
        pre_class = ''
        class_count = 0
        while True:
            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message = bytes(f"{sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(1000000000)
            data_recv = json.loads(data)
            
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
           
            # print(current_speed, current_angle)
            # print(image.shape)            
            # try:
            Img_pid = np.zeros((384,640), dtype='uint8')
            Img_turn = np.zeros((384,640), dtype='uint8') 
            
            # FOR PID ERROR
            start_point_pid = (0, 306) 
            end_point_pid   = (640, 306) 
            
            # FOR DETECT WHEN TO TURN
            start_point_turn = (0, 290) 
            end_point_turn   = (640, 290)
            line_image_turn  = 0
            line_array_turn  = 0
            # FOR LINE ATTRIBUTE
            color     = (255, 255, 255) 
            thickness = 1
            
            # MODEL PREDICT AND SEGMENT 
            predict = model.predict(image)  # return a list of Results objects
            segment = predict[0].masks.data.squeeze(0).cpu().numpy()*255
            
            # DRAWING LINE
            line_image_pid  = cv2.line(Img_pid, start_point_pid, end_point_pid, color, 1) 
            # line_image_turn = cv2.line(Img_turn, start_point_turn, end_point_turn, color, 1) 
                
            new_Img_pid     = line_image_pid * segment 
            # new_Img_turn    =line_image_turn * segment
            # Display the image 
            
            # Find the coordinates of non-zero pixels
            line_array_pid  = np.column_stack(np.where(new_Img_pid > 0))
            # line_array_turn = np.column_stack(np.where(new_Img_turn > 0))
        
            # Extract start and end points
            if line_array_pid.size == 0:
                results = model(image)
                for r in results:
                    box = r.boxes.xyxy.cpu().numpy()
                    y = list(box.flatten())
                    coor_thres = int(y[1] + ((y[3] - y[1])/2))

                start_point_pid = (0, coor_thres) 
                end_point_pid   = (640, coor_thres) 
                line_image_pid  = cv2.line(Img_pid, start_point_pid, end_point_pid, color, 1) 
                new_Img_pid     = line_image_pid * segment 
                line_array_pid  = np.column_stack(np.where(new_Img_pid > 0))
                start_point_pid = tuple(line_array_pid[0])
                end_point_pid   = tuple(line_array_pid[-1])
                momentum_flag = True
                
            else:
                start_point_pid = tuple(line_array_pid[0])
                end_point_pid   = tuple(line_array_pid[-1])
            
            


            # cv2.imshow('segment', segment)
            
            
            # cv2.imshow('new_Img_turn', new_Img_turn)
            # cv2.imshow('origin', image)
            
            detect_flag = 0
            # Var for manage
            turn_flag = 0
            # if (end_point_turn[1] - start_point_turn[1]) >= 630:
            #     turn_flag = 1
            
            line_image_turn = cv2.line(Img_turn, start_point_turn, end_point_turn, color, 1)
            new_Img_turn    =line_image_turn * segment
            line_array_turn = np.column_stack(np.where(new_Img_turn > 0))
            
            # if (sendBack_Speed >= 56):
            #     start_point_turn = (0, 290) 
            #     end_point_turn   = (640, 290)
            #     line_image_turn = cv2.line(Img_turn, start_point_turn, end_point_turn, color, 1)
            #     new_Img_turn    =line_image_turn * segment
            #     line_array_turn = np.column_stack(np.where(new_Img_turn > 0))
            # elif (sendBack_Speed < 56):
            #     start_point_turn = (0, 290) 
            #     end_point_turn   = (640, 290)
            #     line_image_turn = cv2.line(Img_turn, start_point_turn, end_point_turn, color, 1)
            #     new_Img_turn    =line_image_turn * segment
            #     line_array_turn = np.column_stack(np.where(new_Img_turn > 0))
                
            
                
            if line_array_turn.size == 0:
                a= 0
            else:
                start_point_turn = tuple(line_array_turn[0])
                end_point_turn   = tuple(line_array_turn[-1])

            if (end_point_turn[1] - start_point_turn[1]) >= 635:
                turn_flag = 1 

            pre_class = cls
            results = model_detect(image, show = True)
            for r in results:
                box = r.boxes.cls.cpu().numpy()
                prob = r.boxes.conf.cpu().numpy().astype('float64')
                print('prob', prob)
                if box.size == True:
                    cls  = names[int(box[0])]
                    if (pre_class == cls):
                        class_count += 1
                    elif (pre_class != cls):

                        if (class_count == 0):
                            cls = cls
                        else:
                            print('class: ',cls)
                            cls = pre_class
                            print('class_change: ',cls)
                            class_count = 0

                            
                    detect_flag = 1
                    Control(0,-5)
                    
                    

            # PID ERROR

            middle_point_of_lane = int((start_point_pid[1] + end_point_pid[1]) / 2)
            # DIVIDE PID
            middle_point_of_left_lane = int((start_point_pid[1] + middle_point_of_lane) / 2)
            middle_point_of_right_lane = int((end_point_pid[1] + middle_point_of_lane) / 2)
            
            middle_point_of_image = int(640 / 2)
        
            names= ["no_turn_left", "turn_left", "no_turn_right", "turn_right", "go_straight_ahead"]
            Pid_error = middle_point_of_lane - middle_point_of_image
            
            sendBack_angle = PID(Pid_error,0.3,0.0000000015,0.07)
            
            if (Pid_error  < 0 )and (momentum_flag ==    1):
                if ((end_point_pid[1] - start_point_pid[1]) < 320):
                    sendBack_angle = -25
               
                
            elif (Pid_error > 0 )and (momentum_flag ==1):
                if ((end_point_pid[1] - start_point_pid[1]) < 320):
                    sendBack_angle = 25
              
                # else:
                #     # Pid_error = middle_point_of_lane  -  60 - middle_point_of_image
                #     sendBack_angle = 20
            # if detect_flag ==1:
            #     if (cls == 'turn_right') or (cls == 'no_turn_left'):
            #         Pid_error_left =  middle_point_of_left_lane - (middle_point_of_image - 20)  
            #         Pid_error = Pid_error_left

            #     elif (cls == 'turn_left') or (cls == 'no_turn_right'):
            #         Pid_error_right = middle_point_of_right_lane - (middle_point_of_image + 20)  
            #         Pid_error = Pid_error_right
    
            cv2.line(new_Img_pid, (middle_point_of_image, new_Img_pid.shape[0]), (middle_point_of_image, 310), color, thickness)  
            cv2.line(new_Img_pid, (middle_point_of_image, new_Img_pid.shape[0]), (middle_point_of_lane, 310), color, thickness)  
            # cv2.imshow('new_Img_pid', new_Img_pid)
            # cv2.imshow('new_Img_turn', new_Img_turn)
            
            # if (sendBack_Speed < 60):
            #     sendBack_angle = PID(Pid_error,0.25,0.0000015,0.05)
            
            
        
                
            # print(abs(sendBack_angle))
            if turn_flag == 1:
                class_count = 0
                count  = 4
                if (cls == 'turn_right') or (cls == 'no_turn_left'):
                    if current_speed < 55:
                        Control(15,125)
                    else:
                        Control(23,130)

                elif (cls == 'turn_left') or (cls == 'no_turn_right'):
                    if current_speed < 55:
                        Control(-15,130)
                    else:
                        Control(-23,130)

                elif (cls == 'go_straight_ahead'):
                    Control(0,130)
                    
                
            else:
                count -= 1
                if count > 0:
                    if (cls == 'turn_right') or (cls == 'no_turn_left'):
                        if current_speed < 55:
                            Control(20,130)
                        else:
                            Control(25,130)
                    

                    elif (cls == 'turn_left') or (cls == 'no_turn_right'):
                        if current_speed < 55:
                            Control(-20,130)
                        else:
                            Control(-25,130)
                elif count == 0:
                    
                    cls = 0
        
                else:    
                    if abs(sendBack_angle) < 5 and abs(sendBack_angle) > 2:
                        if (current_speed > 66):
                            Control(0, -30)
                        # else:
                        #     Control(0, 135)
                        Control(0, 140)
                    elif abs(sendBack_angle) <= 2:
                        if (current_speed > 66):
                            Control(0, -30)
                        # else:
                        #     Control(0, 135)
                        Control(0, 140)
                    else:
                        Control(sendBack_angle, -10)
                        Control(sendBack_angle, 100)
                
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # except:
            #     print("Turn off PID")

    finally:
        cv2.destroyAllWindows()
        print('closing socket')
        s.close()