import cv2

from output import thin

def capture_and_process():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the captured frame
        cv2.imshow('Frame', frame)

        # Check for key press
        key = cv2.waitKey(1)
        
        # Press spacebar to capture and process the image
        if key == ord(' '):
            # Close the camera
            cap.release()
            cv2.destroyAllWindows()
            
            # Pass the captured image to the main function in output.py
            thin(frame)
            break
        
        # Press esc to exit without capturing
        elif key == 27:
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_process()