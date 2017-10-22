import cv2
import os

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('project_video_processed.mp4',fourcc, 25.0, (1280,720))

files = os.listdir("video_images")
files.sort()

for filename in files:
    frame = cv2.imread("video_images/"+filename)
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()
