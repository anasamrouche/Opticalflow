import horn_schunck_rs
import numpy as np
import rich as r
import cv2

current_image = np.array([
    [1, 2, 1, 0, 1],
    [2, 1, 2, 1, 2],
    [1, 1, 1, 0, 3],
    [2, 2, 0, 1, 2],
    [1, 1, 3, 1, 2],
    [1, 1, 2, 3, 2]
], dtype=np.float32)

next_image = np.array([
    [1, 1, 1, 0, 2],
    [2, 1, 2, 1, 0],
    [1, 1, 1, 0, 3],
    [1, 2, 0, 1, 0],
    [1, 2, 3, 2, 1],
    [1, 2, 3, 1, 2]
], dtype=np.float32)


video = cv2.VideoCapture("./tests/pingponghd.mp4")
frameCount = 80
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(video.get(cv2.CAP_PROP_FPS))

temp_buffer = np.empty((frameCount, frameHeight, frameWidth, 3), np.float32)
video_buffer = np.empty((frameCount, frameHeight, frameWidth), np.float32)

count = 0
keep_going = True
while count < frameCount and keep_going:
    keep_going, temp_buffer[count] = video.read()
    count += 1

video_buffer = np.linalg.norm(temp_buffer, axis=3)

r.print(video_buffer.shape)

optical_flow_x, optical_flow_y = horn_schunck_rs.optical_flow(video_buffer, 50, 20)
optical_flow_x = optical_flow_x.astype(np.uint8)
optical_flow_y = optical_flow_y.astype(np.uint8)

output = cv2.VideoWriter("outputhd.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frameWidth, frameHeight), isColor=False)
for frame in optical_flow_x:
    output.write(frame)

output.release()