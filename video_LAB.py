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
import heartpy as hp
from scipy import signal


plt.ion()  # Set interactive mode on
fig = plt.figure(1)
plt.xlabel("Time(ms)")
plt.ylabel("Pixels")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Excel parameters
time_stamp = []
luminance = []
b_star = []
a_star = []

# plotting parameters
l_plot = []
a_plot = []
b_plot = []
t_plot = []

# Set source_mp4 Video
source_mp4 = '02-base.mp4'
source_csv = '02-base PPG.csv'

# Using Video-capture to get the fps value.
capture = cv2.VideoCapture(source_mp4)
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()
print(fps)

# Using Video-Stream to continuously run Webcam
# vs = VideoStream().start()

# Using Video-capture to run video file
cap = cv2.VideoCapture(source_mp4)

frame_count = 0  # frames count
time_count = 0  # time in milliseconds
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

# Fast Fourier Transform
def fft(data, fs, scale="mag"):
   # Apply Hanning window function to the data.
   data_win = data * np.hanning(len(data))
   if scale == "mag":  # Select magnitude scale.
     mag = 2.0 * np.abs(np.fft.rfft(tuple(data_win)) / len(data_win))  # Single-sided DFT -> FFT
   elif scale == "pwr":  # Select power scale.
     mag = np.abs(np.fft.rfft(tuple(data_win)))**2  # Spectral power
   bin = np.fft.rfftfreq(len(data_win), d=1.0/fs)  # Calculate bins, single-sided
   return bin, mag


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # image = vs.read()
        success, image = cap.read()
        if image is None:
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break
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
                cv2.polylines(image, [left_cheek], True, (0, 255, 255), 2)
                cv2.polylines(image, [right_cheek], True, (0, 255, 255), 2)

                # mask the image and crop the ROI with black background
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [forehead], (255))
                crop_img = cv2.bitwise_and(image, image, mask=mask)

                # Convert to Lab color space, eliminate the black pixels and get mean of LAB for each frame
                b, g, r = cv2.split(crop_img)
                lab_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
                indices_list = np.where(np.any(lab_img != [0, 128, 128], axis=-1))
                roi_pixel_img = lab_img[indices_list]

                # Append the current frame's LAB to plotting parameters
                l_plot.append(roi_pixel_img[:, 0].mean())
                a_plot.append(roi_pixel_img[:, 1].mean())
                b_plot.append(roi_pixel_img[:, 2].mean())
                frame_count += 1
                time_count += (1000/fps)
                t_plot.append(time_count)

                # Draw the face mesh on the image
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                cv2.imshow('MediaPipe FaceMesh', image)
                # cv2.imshow('MediaPipe Masked pixel crop', crop_img)

                # Plot the graph 4 times a sec (15 new records each time)
                if frame_count % 15 == 0:
                    is_update = True  # New frame has come

                    # plot the LAB signals
                    plt.plot(t_plot, l_plot, 'b', label='luminance')
                    plt.plot(t_plot, a_plot, 'g', label='a_star')
                    plt.plot(t_plot, b_plot, 'r', label='b_star')
                    plt.pause(0.01)
                    # Save the plot as png file
                    fig.savefig('rPPGplotVideo.png', dpi=100)
                    update += 1

                elif update > 2:
                    # After 3 plots push the reading to Excel parameters and clear plotting parameters
                    if is_update:
                        if update == 3:
                            luminance.extend(l_plot)
                            a_star.extend(a_plot)
                            b_star.extend(b_plot)
                            time_stamp.extend(t_plot)
                        else:
                            luminance.extend(l_plot[(len(l_plot) - 15):len(l_plot)])
                            a_star.extend(a_plot[(len(a_plot) - 15):len(a_plot)])
                            b_star.extend(b_plot[(len(b_plot) - 15):len(b_plot)])
                            time_stamp.extend(t_plot[(len(t_plot) - 15):len(t_plot)])

                        del l_plot[0:15]
                        del a_plot[0:15]
                        del b_plot[0:15]
                        del t_plot[0:15]

                        is_update = False  # we added the new frame to our list structure

        # Break using esc key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    capture.release()
    # vs.stop()

    # Hold plot and plot the filtered signals
    plt.ioff()
    # 2nd order butterworth bandpass filtering
    bp_b_plot = bandpass(b_star, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
    bp_a_plot = bandpass(a_star, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
    bp_l_plot = bandpass(luminance, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
    # plt.plot(time_stamp, bp_b_plot, 'r', label='BPFiltered_b_star')
    plt.plot(time_stamp, bp_a_plot, 'g', label='BPFiltered_a_star')
    # plt.plot(time_stamp, bp_l_plot, 'b', label='BPFiltered_luminance')
    plt.title("Raw and Filtered Signals")
    # plt.legend()
    # plt.show()

    # Calculate and display FFT
    X_fft, Y_fft = fft(bp_a_plot, fps, scale="mag")
    fig2 = plt.figure(2)
    plt.plot(X_fft, Y_fft)
    plt.title("FFT of filtered Signal")
    fig2.savefig('FFTplotVideo.png', dpi=100)
    # plt.show()

    # Welch's Periodogram
    f_set, Pxx_den = signal.welch(bp_a_plot, fps)
    fig3 = plt.figure(3)
    plt.semilogy(f_set, Pxx_den)
    plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    fig3.savefig('WelchplotVideo.png', dpi=100)
    # plt.show()

    # Calculate Heart Rate and Plot
    working_data, measures = hp.process(bp_a_plot, fps)
    plot_object = hp.plotter(working_data, measures, show=False, title= 'Final_Heart Rate Signal Peak Detection')
    plot_object.savefig('bpmPlotVideo.png', dpi=100)

    hrdata = hp.get_data(source_csv, column_name='Signal')
    timerdata = hp.get_data(source_csv, column_name='Time')
    working_data1, measures1 = hp.process(hrdata, hp.get_samplerate_mstimer(timerdata))
    plot_object1 = hp.plotter(working_data1, measures1, show=False, title= 'Original_Heart Rate Signal Peak Detection')
    plot_object1.savefig('bpmPlotOriginal.png', dpi=100)
    plt.show()

    # Export to pickle byte stream object
    tabular_rgb = np.array([np.array(time_stamp), np.array(b_star), np.array(a_star), np.array(luminance)])
    pickle.dump(tabular_rgb, open("video_save.p", "wb"))
    # _pickle_imported_obj = pickle.load(open("save.p", "rb"))

    # Export to Excel file
    book = Workbook('rPPGVideo.xlsx')
    sheet = book.add_worksheet()
    row = 0
    col = 0

    sheet.write(row, col, 'Time')
    sheet.write(row, col + 1, 'luminance mean')
    sheet.write(row, col + 2, 'a_star mean')
    sheet.write(row, col + 3, 'b_star mean')
    row += 1

    for f, b, g, r in zip(time_stamp, luminance, a_star, b_star):
        sheet.write(row, col, f)
        sheet.write(row, col + 1, b)
        sheet.write(row, col + 2, g)
        sheet.write(row, col + 3, r)
        row += 1
    book.close()
