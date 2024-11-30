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
    new_array = np.where(input_array >= CL, float(1), np.where(input_array <= -CL, float(-1), float(0)))
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

# Function to calculate cepstrum for each frame
def cepstrum_calculation(frames, fs, frame_size, frame_size_samples, frame_shift_samples):
    cepstra_pitch = []
    for frame in frames.T:
        # Calculate the Constant-Q Transform (CQT)
        cqt = np.abs(librosa.core.cqt(y=frame, sr=fs, n_bins=84, bins_per_octave=12, hop_length=frame_shift_samples))

        # Take the logarithm of the magnitude spectrum
        log_cqt = np.log1p(cqt)

        # Take the inverse Fourier transform to get the cepstrum
        cepstrum = np.abs(librosa.core.istft(log_cqt, hop_length=frame_shift_samples))

        # Find peaks in the cepstrum to estimate pitch
        peaks = find_peaks(cepstrum, height=0.2 * (max(cepstrum)), distance=50)[0]
        pitch = np.mean(np.diff(peaks[0:2]))
       
        cepstra_pitch.append(1 / (pitch * frame_size / frame_size_samples))

    return np.array(cepstra_pitch)

# Parameters
fs = 16000
frame_size = 0.03                 # frame size in seconds (30 ms)
frame_shift = 0.5 * frame_size    # frame shift in seconds (15 ms)

# Convert frame sizes and hop sizes to samples
frame_size_samples  = int(frame_size * fs)      # frame size in samples
frame_shift_samples = int(frame_shift * fs)     # frame shift in samples

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

# Calculate pitch using cepstrum for each frame
pitch_frame_of_original_speech = cepstrum_calculation(voiced_frames_original_speech, fs, frame_size, frame_size_samples, frame_shift_samples)
pitch_frame_of_clipped_signal = cepstrum_calculation(voiced_frames_clipped_speech, fs, frame_size, frame_size_samples, frame_shift_samples)
pitch_frame_of_three_level_clipped_signal = cepstrum_calculation(voiced_frames_three_level_clipped_speech, fs, frame_size, frame_size_samples, frame_shift_samples)

# Plot pitch for each frame
plt.subplot(3, 1, 1)
plt.scatter(range(len(voiced_frames_original_speech.T)), pitch_frame_of_original_speech, label='Cepstrum')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 350)
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(range(len(voiced_frames_clipped_speech.T)), pitch_frame_of_clipped_signal, label='Center Clipped Cepstrum')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 350)
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(range(len(voiced_frames_three_level_clipped_speech.T)), pitch_frame_of_three_level_clipped_signal, label='Three Level Center Clipped Cepstrum')
plt.xlabel('Voiced Frame Number')
plt.ylabel('Pitch')
plt.ylim(0, 350)
plt.legend()

plt.show()

# Calculate average pitch for voiced parts of speech
pitch_cepstrum = np.mean(np.nan_to_num(pitch_frame_of_original_speech, nan=0.0))
pitch_center_clipped_cepstrum = np.mean(np.nan_to_num(pitch_frame_of_clipped_signal, nan=0.0))
pitch_3_level_clipped_cepstrum = np.mean(np.nan_to_num(pitch_frame_of_three_level_clipped_signal, nan=0.0))

print(f'Average pitch of voiced parts of speech based on cepstrum: {pitch_cepstrum:.2f} Hz')
print(f'Average pitch of voiced parts of speech based on Center Clipped Cepstrum: {pitch_center_clipped_cepstrum:.2f} Hz')
print(f'Average pitch of voiced parts of speech based on Three Level Clipped Cepstrum: {pitch_3_level_clipped_cepstrum:.2f} Hz')
