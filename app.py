from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp


app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

camera = cv2.VideoCapture(0)


def func():
    with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while True:
                
            ## read the camera frame
            success,frame=camera.read()
            if not success:
                break
            else:
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame)
                    # Draw the hand annotations on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                    
                    # Initially set finger count to 0 for each cap
                    fingerCount = 0
                    if results.multi_hand_landmarks:
                         for hand_landmarks in results.multi_hand_landmarks:
                              handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                              handLabel = results.multi_handedness[handIndex].classification[0].label
                              handLandmarks = []
                              for landmarks in hand_landmarks.landmark:
                                handLandmarks.append([landmarks.x, landmarks.y])
                                # Test conditions for each finger: Count is increased if finger is 
                                #   considered raised.
                                # Thumb: TIP x position must be greater or lower than IP x position, 
                                #   deppeding on hand label.
                              if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                                fingerCount = fingerCount+1
                              elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                                fingerCount = fingerCount+1


                                # Other fingers: TIP y position must be lower than PIP y position, 
                                #   as image origin is in the upper left corner.
                              if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
                                fingerCount = fingerCount+1
                              if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
                                fingerCount = fingerCount+1
                              if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
                                fingerCount = fingerCount+1
                              if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
                                fingerCount = fingerCount+1


                              mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                    # Display finger count
                    cv2.putText(frame, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

                    ret,buffer=cv2.imencode('.jpg',frame)
                    frame=buffer.tobytes()         
                    
            

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/", methods = ['GET', 'POST'])
def hello():
    return render_template('index.html')


@app.route('/video', methods= ['GET', 'POST'])
def video():
    return Response(func(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=8000)  

