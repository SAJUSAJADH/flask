from angleCalculation import calculateAngle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt



def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''

    mp_pose = mp.solutions.pose
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'


    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'

    
     #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the plank pose.
    #---------------------------------------------------------------------------------------------------------------- 
    if left_shoulder_angle > 165 and left_shoulder_angle < 195 and right_shoulder_angle > 165 and right_shoulder_angle < 195 and \
       left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195 and \
       left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:

        label = 'Plank Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the Extended Hand-To-Big-Toe pose.
    #----------------------------------------------------------------------------------------------------------------
    if left_knee_angle > 85 and left_knee_angle < 95 and left_shoulder_angle > 175 and left_shoulder_angle < 185 and \
    right_knee_angle > 85 and right_knee_angle < 95 and right_shoulder_angle > 175 and right_shoulder_angle < 185:
        
        label = 'Extended Hand-To-Big-Toe Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the chair pose.
    #----------------------------------------------------------------------------------------------------------------
    if left_knee_angle > 145 and right_knee_angle > 145 and \
    left_shoulder_angle > 175 and left_shoulder_angle < 185 and right_shoulder_angle > 175 and right_shoulder_angle < 185:
        
        label = 'Chair Pose'
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0) 

    else:

        # Check if the user's pose is close to the target pose
        if label == left_knee_angle < 90 or right_knee_angle < 90:
            label = 'Close to Warrior2 Pose'
        elif label == left_knee_angle < 160 or right_knee_angle < 160:
            label = 'Close to T Pose'
        elif label == left_knee_angle < 315 or right_knee_angle < 315 or left_knee_angle > 318 or right_knee_angle > 318:
            label = 'Close to Tree Pose'
        elif label ==left_shoulder_angle < 165 or right_shoulder_angle < 165 or left_elbow_angle < 165 or right_elbow_angle < 165 or left_knee_angle < 165 or right_knee_angle < 165:
            label = 'Close to Plank Pose'
        elif label ==left_knee_angle < 85 or left_knee_angle > 95 or right_knee_angle < 85 or right_knee_angle > 95:
            label = 'Close to Extended Hand-To-Big-Toe Pose'
        elif label ==left_knee_angle < 145 or right_knee_angle < 145:
            label = 'Close to Chair Pose'
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        # plt.figure(figsize=[10,10])
        # plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        color = (0, 255, 0) 
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label