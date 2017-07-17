# Kevin Perry
# July, 2016
# kevinperry@gatech.edu

import cv2
import numpy as np
import dlib
import time
from scipy import signal

# Constants
window_name = 'Pulse Observer'
buffer_max_size = 300
min_hz = 0.83  # 50 BPM
max_hz = 3.33  # 200 BPM
graph_height = 200
min_frames = 100
show_fps = True  # Controls whether the FPS is displayed in top-left of GUI window.

# Lists for storing video frame data
values = []
times = []


# Creates the specified Butterworth filter and applies it.
# See:  http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


# Gets the region of interest for the forehead.
def get_forehead_roi(face_points):
    # Store the points in a Numpy array so we can easily get the min and max for x and y via slicing
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Forehead area between eyebrows
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)


# Gets the region of interest for the nose.
def get_nose_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Nose and cheeks
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)
    return int(left), int(right), int(top), int(bottom)


# Gets region of interest that includes forehead, eyes, and nose.
# Note:  Combination of forehead and nose performs better.  This is probably because this ROI includes
#	     the eyes, and eye blinking adds noise.
def get_full_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Only keep the points that correspond to the internal features of the face (e.g. mouth, nose, eyes, brows).
    # The points outlining the jaw are discarded.
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(np.min(points[17:47, 0]))
    min_y = int(np.min(points[17:47, 1]))
    max_x = int(np.max(points[17:47, 0]))
    max_y = int(np.max(points[17:47, 1]))

    center_x = min_x + (max_x - min_x) / 2
    # center_y = min_y + (max_y - min_y) / 2
    left = min_x + int((center_x - min_x) * 0.15)
    right = max_x - int((max_x - center_x) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y
    return int(left), int(right), int(top), int(bottom)


def sliding_window_demean(signal, num_windows):
    window_size = int(round(len(signal) / num_windows))
    demeaned = np.zeros(signal.shape)
    for i in xrange(0, len(signal), window_size):
        if i + window_size > len(signal):
            window_size = len(signal) - i
        slice = signal[i:i + window_size]
        if slice.size == 0:
            print 'Empty Slice: size={0}, i={1}, window_size={2}'.format(signal.size, i, window_size)
            print slice
        demeaned[i:i + window_size] = slice - np.mean(slice)
    return demeaned


def get_avg(roi1, roi2):
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    return avg


# Draws the heart rate graph in the GUI window.
def draw_graph(signal, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / buffer_max_size
    scale_factor_y = 30
    midpoint_y = graph_height / 2
    for i in xrange(0, signal.shape[0] - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    return graph


# Draws the heart rate text (BPM) in the GUI window.
def draw_bpm(bpm_str, bpm_width, bpm_height):
    bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
    # Draw gray line to separate graph from BPM display
    bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                   thickness=2)
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height / 2 + bpm_text_base)
    cv2.putText(bpm_display, bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2.7, color=(0, 255, 0), thickness=2)
    bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                     thickness=1)
    bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
    bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
    cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 255, 0), thickness=1)
    return bpm_display


# Draws the current frames per second in the GUI window.  This can be turned off by setting the
# "show_fps" constant to False.
def draw_fps(frame, fps):
    cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=-1)
    cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (5, 20), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 255, 0))
    return frame


# Main function.
def run_pulse_observer(detector, predictor, webcam, window):
    last_bpm = 0

    # cv2.getWindowProperty() returns -1 when window is closed by user.
    while cv2.getWindowProperty(window, 0) == 0:
        r, frame = webcam.read()

        # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
        # The original frame will be used to compute heart rate.
        view = np.array(frame)

        # Detect face using dlib
        faces = detector(frame, 0)
        if len(faces) == 1:
            face_points = predictor(frame, faces[0])

            # Get the regions of interest.
            fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
            nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

            # Draw green rectangles around our regions of interest (ROI)
            cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
            cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

            # Slice out the regions of interest (ROI) and average them
            fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
            nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
            avg = get_avg(fh_roi, nose_roi)

            # Add value and time to lists
            values.append(avg)
            times.append(time.time())

            # Buffer is full, so pop the value off the top
            if len(times) > buffer_max_size:
                values.pop(0)
                times.pop(0)

            curr_buffer_size = len(times)

            # Heart rate graph gets 75% of window width.  BPM gets 25%.
            graph_width = int(view.shape[1] * 0.75)
            bpm_display_width = view.shape[1] - graph_width

            # Don't try to compute pulse until we have at least the min. number of frames (e.g. 60)
            if curr_buffer_size > min_frames:
                # Smooth the signal by detrending and demeaning
                detrended = signal.detrend(np.array(values), type='linear')
                demeaned = sliding_window_demean(detrended, 15)

                # Compute relevant times
                time_elapsed = times[-1] - times[0]
                fps = curr_buffer_size / time_elapsed  # frames per second

                # Filter signal with Butterworth bandpass filter
                filtered = butterworth_filter(demeaned, min_hz, max_hz, fps, order=5)

                # Compute FFT
                fft = np.abs(np.fft.rfft(filtered))

                # Generate list of frequencies that correspond to the FFT values
                freqs = fps / curr_buffer_size * np.arange(curr_buffer_size / 2 + 1)

                # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
                # because they correspond to impossible BPM values.
                while True:
                    max_idx = fft.argmax()
                    bps = freqs[max_idx]
                    if bps < min_hz or bps > max_hz:
                        print 'BPM of {0} was discarded.'.format(bps * 60.0)
                        fft[max_idx] = 0
                    else:
                        bpm = bps * 60.0
                        break

                # It's impossible for the heart rate to change more than 10% between samples,
                # so use a weighted average to smooth the BPM with the last BPM.
                if last_bpm > 0:
                    bpm = (last_bpm * 0.9) + (bpm * 0.1)
                last_bpm = bpm

                graph = draw_graph(filtered, graph_width, graph_height)
                bpm_display = draw_bpm(str(int(round(bpm))), bpm_display_width, graph_height)

                if show_fps:
                    view = draw_fps(view, fps)

            else:
                # If there's not enough data to compute HR, show an empty graph with loading text and
                # the BPM placeholder
                graph = np.zeros((graph_height, graph_width, 3), np.uint8)
                pct = int(round(float(curr_buffer_size) / min_frames * 100.0))
                loading_text = 'Computing pulse: ' + str(pct) + '%'
                loading_size, loading_base = cv2.getTextSize(loading_text, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                                             fontScale=1, thickness=1)
                loading_x = int((graph_width - loading_size[0]) / 2)
                loading_y = int(graph_height / 2 + loading_base)
                cv2.putText(graph, loading_text, (loading_x, loading_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1, color=(0, 255, 0), thickness=1)
                bpm_display = draw_bpm('--', bpm_display_width, graph_height)

            graph = np.hstack((graph, bpm_display))
            view = np.vstack((view, graph))

        else:
            # No faces detected, so we must clear the lists of values and timestamps.  Otherwise there will be a gap
            # in timestamps when a face is detected again.
            del values[:]
            del times[:]

        cv2.imshow(window, view)
        key = cv2.waitKey(1)
        # Exit if user presses the escape key
        if key == 27:
            break


def main():
    detector = dlib.get_frontal_face_detector()
    # Predictor pre-trained model can be downloaded from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    webcam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)
    run_pulse_observer(detector, predictor, webcam, window_name)
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
