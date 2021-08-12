# Kaushik Goud Chandapet
# 1536199, Mechatronics
# University of Siegen, Germany

# Remote Photoplethysmography
# Version 1.0 (August 2021)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from imutils.video import VideoStream
import pickle
from xlsxwriter import Workbook
import scipy.signal as sig

plt.ion()  # Set interactive mode on
fig = plt.figure()
plt.xlabel("Frames")
plt.ylabel("Pixels")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Excel parameters
fr = []
blue = []
red = []
green = []

# plotting parameters
b_plot = []
g_plot = []
r_plot = []
f_plot = []

# Using Video-capture to get the fps value.
capture = cv2.VideoCapture(0)
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()
print(fps)

# Using Video-Stream to continuously run Webcam
vs = VideoStream().start()

# Using Video-capture to run video file
# cap = cv2.VideoCapture("WIN_20210712_16_25_07_Pro.mp4")

frame_count = 0  # frames count
update = 0  # plot update

is_update = False


# Butterworth forward-backward band-pass filter
def bandpass(signal, fs, order, fc_low, fc_hig, debug=False):
    """Butterworth forward-backward band-pass filter.

    :param signal: list of ints or floats; The vector containing the signal samples.
    :param fs: float; The sampling frequency in Hz.
    :param order: int; The order of the filter.
    :param fc_low: int or float; The lower cutoff frequency of the filter.
    :param fc_hig: int or float; The upper cutoff frequency of the filter.
    :param debug: bool, default=False; Flag to enable the debug mode that prints additional information.

    :return: list of floats; The filtered signal.
    """
    nyq = 0.5 * fs  # Calculate the Nyquist frequency.
    cut_low = fc_low / nyq  # Calculate the lower cutoff frequency (-3 dB).
    cut_hig = fc_hig / nyq  # Calculate the upper cutoff frequency (-3 dB).
    bp_b, bp_a = sig.butter(order, (cut_low, cut_hig), btype="bandpass")  # Design and apply the band-pass filter.
    bp_data = list(sig.filtfilt(bp_b, bp_a, signal))  # Apply forward-backward filter with linear phase.
    return bp_data


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        image = vs.read()
        # success, image = cap.read()
        if image is None:
            # If loading a video, use 'break' instead of 'continue'.
            continue
            # break
        height, width, _ = image.shape
        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        processed_img = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert the RGB image to BGR.
        if processed_img.multi_face_landmarks:
            for face_landmarks in processed_img.multi_face_landmarks:
                landmark_points = []
                for i in range(0, 468):
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    p = [x, y]
                    landmark_points.append([x, y])
                # Set ROI points
                forehead = np.array((
                    landmark_points[9], landmark_points[107], landmark_points[66], landmark_points[105],
                    landmark_points[104], landmark_points[103],
                    landmark_points[67], landmark_points[109], landmark_points[10],
                    landmark_points[338], landmark_points[297], landmark_points[332],
                    landmark_points[333], landmark_points[334], landmark_points[296],
                    landmark_points[336]))
                left_cheek = np.array((landmark_points[266], landmark_points[426], landmark_points[436],
                                       landmark_points[416], landmark_points[376],
                                       landmark_points[352], landmark_points[347], landmark_points[330]))
                right_cheek = np.array((landmark_points[36], landmark_points[206], landmark_points[216],
                                        landmark_points[192], landmark_points[147],
                                        landmark_points[123], landmark_points[117], landmark_points[118],
                                        landmark_points[101]))
                # Alternate ROIs
                left_cheek1 = np.array((landmark_points[266], landmark_points[426], landmark_points[411],
                                        landmark_points[346], landmark_points[347], landmark_points[330]))
                right_cheek1 = np.array((landmark_points[36], landmark_points[206], landmark_points[187],
                                         landmark_points[117], landmark_points[118], landmark_points[101]))

                # Draw ROI's on the image
                cv2.polylines(image, [forehead], True, (0, 255, 255), 2)
                cv2.polylines(image, [left_cheek1], True, (0, 255, 255), 2)
                cv2.polylines(image, [right_cheek1], True, (0, 255, 255), 2)

                # mask the image and crop the ROI with black background
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [forehead, left_cheek1, right_cheek1], (255))
                crop_img = cv2.bitwise_and(image, image, mask=mask)

                # eliminate the black pixels and get mean of RGB for each frame
                b, g, r = cv2.split(crop_img)
                indices_list = np.where(np.any(crop_img != [0, 0, 0], axis=-1))
                roi_pixel_img = crop_img[indices_list]

                # Append the current frame's RGB to plotting parameters
                b_plot.append(roi_pixel_img[:, 0].mean())
                g_plot.append(roi_pixel_img[:, 1].mean())
                r_plot.append(roi_pixel_img[:, 2].mean())
                frame_count += 1
                f_plot.append(frame_count)

                # Draw the face mesh on the image
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                cv2.imshow('MediaPipe FaceMesh', image)
                cv2.imshow('MediaPipe Masked pixel crop', crop_img)

                # Plot the graph 3 times a sec (10 new records each time)
                if frame_count % int(fps / 3) == 0:
                    is_update = True  # New frame has come

                    # plot the RGB signals
                    plt.plot(f_plot, b_plot, 'b', label='Blue')
                    plt.plot(f_plot, g_plot, 'g', label='Green')
                    plt.plot(f_plot, r_plot, 'r', label='Red')
                    plt.pause(0.01)
                    # Save the plot as png file
                    fig.savefig('rPPGplotLive.png', dpi=100)
                    update += 1

                elif update > 2:
                    # After 3 plots push the reading to Excel parameters and clear plotting parameters
                    if is_update:
                        if update == 3:
                            blue.extend(b_plot)
                            green.extend(g_plot)
                            red.extend(r_plot)
                            fr.extend(f_plot)
                        else:
                            blue.extend(b_plot[(len(b_plot) - int(fps / 3)):(len(b_plot) - 1)])
                            green.extend(g_plot[(len(g_plot) - int(fps / 3)):(len(g_plot) - 1)])
                            red.extend(r_plot[(len(r_plot) - int(fps / 3)):(len(r_plot) - 1)])
                            fr.extend(f_plot[(len(f_plot) - int(fps / 3)):(len(f_plot) - 1)])

                        del b_plot[0:int(fps / 3)]
                        del g_plot[0:int(fps / 3)]
                        del r_plot[0:int(fps / 3)]
                        del f_plot[0:int(fps / 3)]

                        is_update = False  # we added the new frame to our list structure

        # Break using esc key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    # cap.release()
    capture.release()
    vs.stop()

    # Hold plot and plot the filtered signals
    plt.ioff()
    # 2nd order butterworth bandpass filtering
    bp_r_plot = bandpass(red, fps, 2, 0.5, 3)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
    bp_g_plot = bandpass(green, fps, 2, 0.5, 2)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
    bp_b_plot = bandpass(blue, fps, 2, 0.5, 2)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
    plt.plot(fr, bp_r_plot, 'r', label='BPFiltered_Red')
    # plt.plot(fr, bp_g_plot, 'g', label='BPFiltered_Green')
    # plt.plot(fr, bp_b_plot, 'b', label='BPFiltered_Blue')
    # plt.legend()
    plt.show()

    # Export to pickle byte stream object
    tabular_rgb = np.array([np.array(fr), np.array(red), np.array(green), np.array(blue)])
    pickle.dump(tabular_rgb, open("save.p", "wb"))
    # _pickle_imported_obj = pickle.load(open("save.p", "rb"))

    # Export to Excel file
    book = Workbook('rPPGLive.xlsx')
    sheet = book.add_worksheet()
    row = 0
    col = 0

    sheet.write(row, col, 'Frames')
    sheet.write(row + 1, col, 'Blue mean')
    sheet.write(row + 2, col, 'Green mean')
    sheet.write(row + 3, col, 'Red mean')
    col += 1

    for f, b, g, r in zip(fr, blue, green, red):
        sheet.write(row, col, f)
        sheet.write(row + 1, col, b)
        sheet.write(row + 2, col, g)
        sheet.write(row + 3, col, r)
        col += 1
    book.close()