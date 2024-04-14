import cv2
import mediapipe as mp
from detect_pose import detectPose
from classify_pose import classifyPose
import os


def webCam(stop_flag):

    mp_pose = mp.solutions.pose

    images = []
    for image in os.listdir('poses'):
        name, ext = os.path.splitext(image)
        images.append(name)
        print(name)

    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)


    # Initialize a resizable window.
    # cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    small_height = 0
    small_width = 0

    # Iterate until the webcam is accessed successfully.
    while stop_flag.value:
        
        # Read a frame.
        ok, frame = camera_video.read()

        
        # Check if frame is not read properly.
        if not ok:
            
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)

        label = ''
        
        # Check if the landmarks are detected.
        if landmarks:
            
            # Perform the Pose Classification.
            frame, label = classifyPose(landmarks, frame, display=False)

        if 'Close' in label:
            for entry in images:
                if entry in label:
                    small_image = cv2.imread(f'poses/{entry}.jpeg')
                    # Get the dimensions of the small image
                    small_height, small_width, _ = small_image.shape
                    small_height, small_width, _ = small_image.shape
                    small_height = int(small_height * 0.30)  # Reduce height to 50%
                    small_width = int(small_width * 0.20)  # Reduce width to 50%
                    small_image = cv2.resize(small_image, (small_width, small_height))
            x_offset = frame.shape[1] - small_width - 10  
            y_offset = frame.shape[0] - small_height - 10
            frame[y_offset:y_offset+small_height, x_offset:x_offset+small_width] = small_image

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()