import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import medfilt

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
    return frames[:, voiced_frame_indices], voiced_frame_indices

# Function to calculate pitch for each frame using autocorrelation
def calculation_of_pitch_for_each_frame_with_autocorrelation(frames, frame_size, frame_size_samples):
    pitch_frame_autocorrelation = []

    # Calculating the pitch for each voiced frame
    for v_frame in frames.T:
        frame = librosa.autocorrelate(v_frame, axis=-1)

        peaks = find_peaks(frame, distance=53)
        pitch = np.mean(np.diff(peaks[0][0:5]))
    
        pitch_frame_autocorrelation.append(1 / (pitch * frame_size / frame_size_samples))
    
    return np.array(pitch_frame_autocorrelation)

# Parameters
fs = 16000
frame_size = 0.025  # frame size in seconds (25 ms)
frame_shift = 0.5 * frame_size  # frame shift in seconds (12.5 ms)

# Load the speech signal
speech_signal, sr = librosa.load('Input_audio.wav')

# Resample the speech signal
sampled_speech_signal = librosa.resample(speech_signal, orig_sr=sr, target_sr=fs)

# Apply center clipping and three-level center clipping
clipped_signal = center_clipping_function(sampled_speech_signal)
three_level_clipped_signal = three_level_center_clipping_function(sampled_speech_signal)

# Plot the original, center-clipped, and three-level center-clipped signals
plt.subplot(3, 1, 1)
plt.plot(range(len(sampled_speech_signal)), sampled_speech_signal)
plt.title('Sampled Original Signal')
plt.xlabel('Sample')

plt.subplot(3, 1, 2)
plt.plot(range(len(clipped_signal)), clipped_signal)
plt.title('Center Clipped Signal')
plt.xlabel('Sample')

plt.subplot(3, 1, 3)
plt.plot(range(len(three_level_clipped_signal)), three_level_clipped_signal)
plt.title('Three-level Center Clipped Signal')
plt.xlabel('Sample')

plt.tight_layout()
plt.show()

# Convert frame sizes and hop sizes to samples
frame_size_samples = int(frame_size * fs)      # frame size in samples
frame_shift_samples = int(frame_shift * fs)    # frame shift in samples

# Calculate frames for original, center-clipped, and three-level center-clipped signals
frames_original_speech = calculation_of_frames(sampled_speech_signal)
frames_clipped_speech = calculation_of_frames(clipped_signal)
frames_three_level_clipped_speech = calculation_of_frames(three_level_clipped_signal)

# Calculate energy for each frame
energy_original_speech = np.sum(frames_original_speech**2, axis=0)
energy_clipped_speech = np.sum(frames_clipped_speech**2, axis=0)
energy_3_level_clipped_speech = np.sum(frames_three_level_clipped_speech**2, axis=0)

# Separate voiced and unvoiced frames based on the energy threshold
voiced_frames_original_speech, voiced_indices = voiced_frame_extraction(frames_original_speech, energy_original_speech)
voiced_frames_clipped_speech = frames_clipped_speech[:, voiced_indices]
voiced_frames_three_level_clipped_speech = frames_three_level_clipped_speech[:, voiced_indices]

# Create time axis for plotting
time = np.arange(0, len(sampled_speech_signal) / fs, frame_shift)[:-2]

# Plot original speech signal and its energy
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(sampled_speech_signal)) / fs, sampled_speech_signal)
plt.title('Original Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, energy_original_speech, label='Energy')
plt.plot([0,len(sampled_speech_signal)/fs],[0.12 * np.max(energy_original_speech),0.12 * np.max(energy_original_speech)])
plt.title('Energy versus Time')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.annotate(text=f"threshold for voiced frame energy = {0.12 * np.max(energy_original_speech):.3f}",
             xytext=(0.7,0.5),
             textcoords="figure fraction",
             xy=(0.80*len(sampled_speech_signal) / fs, 0.12 * np.max(energy_original_speech)),
             xycoords = "data",
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Calculate pitch for each frame using autocorrelation
pitch_frame_of_original_speech = calculation_of_pitch_for_each_frame_with_autocorrelation(voiced_frames_original_speech, frame_size, frame_size_samples)
pitch_frame_of_clipped_signal = calculation_of_pitch_for_each_frame_with_autocorrelation(voiced_frames_clipped_speech, frame_size, frame_size_samples)
pitch_frame_of_three_level_clipped_signal = calculation_of_pitch_for_each_frame_with_autocorrelation(voiced_frames_three_level_clipped_speech, frame_size, frame_size_samples)

# Plot pitch for each frame
plt.subplot(3, 1, 1)
plt.scatter(range(len(voiced_frames_original_speech.T)), pitch_frame_of_original_speech, label='Autocorrelation')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 400)
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.scatter(range(len(voiced_frames_clipped_speech.T)), pitch_frame_of_clipped_signal, label='Center Clipped Autocorrelation')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 400)
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.scatter(range(len(voiced_frames_three_level_clipped_speech.T)), pitch_frame_of_three_level_clipped_signal, label='Three-level Center Clipped Autocorrelation')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 400)
plt.legend()
plt.grid()

plt.show()

# Calculate average pitch for voiced parts of speech
pitch_autocorrelation = np.mean(pitch_frame_of_original_speech)
pitch_center_clipped_autocorrelation = np.mean(np.nan_to_num(pitch_frame_of_clipped_signal, nan=0.0))
pitch_3_level_clipped_autocorrelation = np.mean(np.nan_to_num(pitch_frame_of_three_level_clipped_signal, nan=0.0))

print(f'Average pitch of voiced parts of speech based on autocorrelation: {pitch_autocorrelation:.2f} Hz')
print(f'Average pitch of voiced parts of speech based on center_clipped_autocorrelation: {pitch_center_clipped_autocorrelation:.2f} Hz')
print(f'Average pitch of voiced parts of speech based on three_level_clipped_autocorrelation: {pitch_3_level_clipped_autocorrelation:.2f} Hz')
