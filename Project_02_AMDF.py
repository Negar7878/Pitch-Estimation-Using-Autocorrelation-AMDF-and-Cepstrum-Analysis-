import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to perform center clipping on an input array
def center_clipping_function(input_array):
    CL = 0.2 * max(np.abs(input_array))
    new_array = np.where(input_array >= CL, input_array - CL, np.where(input_array <= -CL, input_array + CL, 0))
    return new_array

# Function to perform three-level center clipping on an input array
def three_level_center_clipping_function(input_array):
    CL = 0.2 * max(np.abs(input_array))
    new_array = np.where(input_array >= CL, 1, np.where(input_array <= -CL, -1, 0))
    return new_array

# Function to calculate frames from an input signal
def calculation_of_frames(input_signal):
    frames = librosa.util.frame(input_signal, 
                                frame_length=frame_size_samples,
                                hop_length=frame_shift_samples)
    return frames

# Function to extract voiced frames based on energy threshold
def voiced_frame_extraction(frames, energy):
    threshold = 0.12 * np.max(energy)
    voiced_frame_indices = np.where(energy > threshold)[0]
    return frames[:, voiced_frame_indices]

# Function to calculate the Average Magnitude Difference Function (AMDF) for each frame
def amdf(frames, max_shift, frame_size, frame_size_samples):
    pitch_frame_amdf = []
    
    for frame in frames.T:
        amdf_values = np.zeros(max_shift + 1)
        for k in range(1, max_shift + 1):
            amdf_values[k] = np.mean(np.abs(frame[:-k] - frame[k:]))
        
        valleys = find_peaks(-amdf_values, distance=53)[0]
        pitch = np.mean(np.diff(valleys[0:5]))
        
        pitch_frame_amdf.append(1 / (pitch * frame_size / frame_size_samples))
    
    return np.array(pitch_frame_amdf)

# Parameters
fs = 16000
frame_size = 0.025                # frame size in seconds (30 ms)
frame_shift = 0.5 * frame_size    # frame shift in seconds (15 ms)

# Convert frame sizes and hop sizes to samples
frame_size_samples  = int(frame_size * fs)      # frame size in samples
frame_shift_samples = int(frame_shift * fs)     # frame shift in samples
max_shift = frame_size_samples - 20

# Load the speech signal
speech_signal, sr = librosa.load('Input_audio.wav')

# Resample the speech signal
sampled_speech_signal = librosa.resample(speech_signal, orig_sr=sr, target_sr=fs)

# Apply center clipping and three-level center clipping
clipped_signal = center_clipping_function(sampled_speech_signal)
three_level_clipped_signal = three_level_center_clipping_function(sampled_speech_signal)

# Calculate frames for original, center-clipped, and three-level center-clipped signals
frames_original_speech = calculation_of_frames(sampled_speech_signal)
frames_clipped_speech = calculation_of_frames(clipped_signal)
frames_three_level_clipped_speech = calculation_of_frames(three_level_clipped_signal)

# Calculate energy for each frame
energy_original_speech = np.sum(frames_original_speech**2, axis=0)

# Separate voiced and unvoiced frames based on the energy threshold
voiced_frames_original_speech = voiced_frame_extraction(frames_original_speech, energy_original_speech)
voiced_frames_clipped_speech = voiced_frame_extraction(frames_clipped_speech, energy_original_speech)
voiced_frames_three_level_clipped_speech = voiced_frame_extraction(frames_three_level_clipped_speech, energy_original_speech)

# Calculate pitch using AMDF for each frame
pitch_frame_of_original_speech = amdf(voiced_frames_original_speech, max_shift, frame_size, frame_size_samples)
pitch_frame_of_clipped_signal = amdf(voiced_frames_clipped_speech, max_shift, frame_size, frame_size_samples)
pitch_frame_of_three_level_clipped_signal = amdf(voiced_frames_three_level_clipped_speech, max_shift, frame_size, frame_size_samples)

# Plot pitch for each frame
plt.subplot(3, 1, 1)
plt.scatter(range(len(voiced_frames_original_speech.T)), pitch_frame_of_original_speech, label='AMDF')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 400)
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.scatter(range(len(voiced_frames_clipped_speech.T)), pitch_frame_of_clipped_signal, label='Center Clipped AMDF')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 400)
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.scatter(range(len(voiced_frames_three_level_clipped_speech.T)), pitch_frame_of_three_level_clipped_signal, label='Three Level Center Clipped AMDF')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 400)
plt.legend()
plt.grid()

plt.show()

# Calculate average pitch for voiced parts of speech
pitch_AMDF = np.mean(np.nan_to_num(pitch_frame_of_original_speech, nan=0.0))
pitch_center_clipped_AMDF = np.mean(np.nan_to_num(pitch_frame_of_clipped_signal, nan=0.0))
pitch_3_level_clipped_AMDF = np.mean(np.nan_to_num(pitch_frame_of_three_level_clipped_signal, nan=0.0))

print(f'Average pitch of voiced parts of speech based on AMDF: {pitch_AMDF:.2f} Hz')
print(f'Average pitch of voiced parts of speech based on Center Clipped AMDF: {pitch_center_clipped_AMDF:.2f} Hz')
print(f'Average pitch of voiced parts of speech based on Three Level Clipped AMDF: {pitch_3_level_clipped_AMDF:.2f} Hz')
