import cv2

# Open webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        break

    out.write(frame)  # Save frame to file
    cv2.imshow("Webcam Recording", frame)  # Display the frame

    # Press 'q' to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()
