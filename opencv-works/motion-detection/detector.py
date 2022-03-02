import cv2, time
import pandas as pd
from datetime import datetime

# assign initial frame to None
first_frame = None

# List when any moving object appear
status_list = [None, None]

# saving time of movemeent
times = []

# initialize dataframe, for Start and End of movement
df = pd.DataFrame(columns=["Start", "End"])

# capturing video
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
	# reading frame/image from video
	check, frame = video.read()
	status = 0

    # Converting color image to gray_scale image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Converting gray scale image to GaussianBlur (so that change can be find easily)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	if first_frame is None:
		first_frame = gray
		continue

	# calculating difference b/w first_frame  and current frame(gaussian blur image)
	delta_frame = cv2.absdiff(first_frame, gray)

	# if difference is greater than 30, it will show as white color(255)
	thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)

	# finding contour of moving object
	cnts, _ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour) < 10000:
			continue
		status = 1

		(x, y, w, h) = cv2.boundingRect(contour)
		# making rectangle around moving object
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

	# appending status of motion
	status_list.append(status)

	status_list = status_list[-2:]

	# appending start time of motion
	if status_list[-1] == 1 and status_list[-2] == 0:
		times.append(datetime.now())

	# Appending End time of motion
	if status_list[-1] == 0 and status_list[-2] == 1:
		times.append(datetime.now())

	# Displaying image in gray_scale
	cv2.imshow("Gray Frame", gray)

	# Displaying the difference in currentframe to
	# the first_frame
	cv2.imshow("Difference Frame", delta_frame)

	# Displaying the black and white image in which if
	# intensity difference greater than 30 it will appear white
	cv2.imshow("Thresh", thresh_delta)

	# Displaying color frame with contour of motion of object
	cv2.imshow("Color Frame", frame)

	key = cv2.waitKey(1)
	# if 'q' entered whole process will stop
	if key == ord('q'):
		# if something is moving, then it append the end time of movement
		if status == 1:
			times.append(datetime.now())
		break

# Appending time of motion in DataFrame
for i in range(0, len(times), 2):
	df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

# Creating a CSV file in which time of movements will be saved
df.to_csv("Time.csv")

video.release()
cv2.destroyAllWindows()


video.release()
cv2.destroyAllWindows()
