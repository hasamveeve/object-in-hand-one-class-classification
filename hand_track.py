import cv2
import mediapipe
import math 


showcrop = False

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture("../VIDEOS/outNew1.mp4")

frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0
remove_counter = 0

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=15)
count_mask = 0
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=5) as hands:
    while (True):

        ret, frame = capture.read()
        count += 1
        count_mask += 1
        center_points_cur_frame = []
        
        #cv2.rectangle(frame, (80, 60), (620, 450), (0, 255, 0), 2)       

        roi = frame[140: 920,300: 800]
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks != None:
            h, w, c = frame.shape


            for handLandmarks in results.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for normalizedLandmark in handLandmarks.landmark:
                    #normalizedLandmark = handLandmarks.landmark[point]
                    x1, y1 = int(normalizedLandmark.x * w), int(normalizedLandmark.y * h)
                    pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                              normalizedLandmark.y,
                                                                                              frameWidth, frameHeight)

                    #cv2.circle(frame, pixelCoordinatesLandmark, 5, (0, 255, 0), -1)
                    if x1 > x_max:
                        x_max = x1
                    if x1 < x_min:
                        x_min = x1
                    if y1 > y_max:
                        y_max = y1
                    if y1 < y_min:
                        y_min = y1
                        
                        
                    
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            #sd = cv2.resize(cropped_frame,(x_max-x_min, y_max-y_min))

            if showcrop and x_min>0 and y_min > 0:
                cv2.imshow("cropped", cropped_frame)
                cv2.destroyWindow("cropped")
                cv2.imshow("cropped", cropped_frame)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)       
            cv2.imshow("Mask", mask)
            if x_min>20:
                x_min -= 20
            if y_min>20:
                y_min -= 20

            #save_img = cv2.resize(mask[y_min:y_max+20, x_min:x_max+20], (200, 200))
            
            #cv2.imwrite("masks/mask-"+str(count_mask)+".png", save_img)
            
            #center-Points
            cx = int(x_min + (x_max-x_min) / 2)
            cy = int(y_min + (y_max-y_min) / 2)
            
            center_points_cur_frame.append((cx,cy))
            if count <= 2:
                for pt in center_points_cur_frame:
                    for pt2 in center_points_prev_frame:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                        if distance < 200:
                            tracking_objects[track_id] = pt
                            track_id += 1

            else:

                tracking_objects_copy = tracking_objects.copy()
                center_points_cur_frame_copy = center_points_cur_frame.copy()

                for object_id, pt2 in tracking_objects_copy.items():
                    object_exists = False
                    for pt in center_points_cur_frame_copy:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                        # Update IDs position
                        if distance < 200:
                            tracking_objects[object_id] = pt
                            object_exists = True
                            if pt in center_points_cur_frame:
                                center_points_cur_frame.remove(pt)
                            continue

                    # Remove IDs lost
                    if not object_exists:
                        remove_counter+=1
                        if remove_counter >10:
                            remove_counter = 0
                            tracking_objects.pop(object_id)

                # Add new IDs found
                for pt in center_points_cur_frame:
                    tracking_objects[track_id] = pt
                    track_id += 1

            for object_id, pt in tracking_objects.items():
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
            
            center_points_prev_frame = center_points_cur_frame.copy()



            
            
        cv2.imshow('Test hand', frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
capture.release()