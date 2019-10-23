import numpy as np
import librosa
from playsound import playsound
import logging
logging.basicConfig(level=logging.DEBUG)

def applyDelay(signal, delay, fs):
  delayedSingal = np.zeros(int(fs * delay))
  outputSignal = np.hstack([delayedSingal, signal])
  return outputSignal

if __name__ == '__main__':
  rawAudio, rawAudio_fs = librosa.load('always.flac')

  initialDelay = 0.2
  leftDelay = initialDelay
  rightDelay = initialDelay + 0.2/1000 # (less than 0.6 msec)

  leftAudio = applyDelay(rawAudio, leftDelay, rawAudio_fs)
  rightAudio = applyDelay(rawAudio, rightDelay, rawAudio_fs)

  outAudio_length = min(leftAudio.shape[0], rightAudio.shape[0])
  leftAudio_trimmed = leftAudio[:outAudio_length].reshape(-1,1)
  rightAudio_trimmed = rightAudio[:outAudio_length].reshape(-1,1)
  outputAudio = np.asfortranarray(np.hstack([leftAudio_trimmed, rightAudio_trimmed]))

  logging.info("Saving the output sound {}".format(outputAudio.shape))
  librosa.output.write_wav('out.wav', outputAudio, rawAudio_fs)
  
  logging.info("Play output sound")
  playsound('out.wav')