import mediapipe as mp
import cv2 as cv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

video = "video.mp4"

vidcap = cv.VideoCapture(video)

winwidth = 480
winheight = 750

# if you want to save the video directly
#out = cv.VideoWriter('hand_tracking.mp4',-1,30.0, (winwidth,winheight))


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break   

        # Process the frame for hand tracking
        processFrames = hands.process(frame)

        frame.flags.writeable = True
        # Draw landmarks on the frame
        if processFrames.multi_hand_landmarks:
            for hand_landmarks in processFrames.multi_hand_landmarks:
                x, y = hand_landmarks.landmark[8].x * winwidth, hand_landmarks.landmark[8].y * winheight
                x1, y1 = hand_landmarks.landmark[12].x * winwidth, hand_landmarks.landmark[12].y * winheight
   
                cv.circle(frame, (int(x), int(y)), 18, (202, 156, 225), -1)
                cv.circle(frame, (int(x1), int(y1)), 18, (207, 255, 229), -1)


        # Resize the frame to the desired window size 
        resized_frame = cv.resize(frame, (winwidth, winheight))

        # saving the video
        # out.write(resized_frame)

        
        # Display the resized frame
        cv.imshow('Hand Tracking', resized_frame)

        # Exit loop by pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vidcap.release()
# out.release()
cv.destroyAllWindows()
