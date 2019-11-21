import visr
import rrl
import objectmodel
import glob

from DynamicHrirRenderer import DynamicHrirRenderer
from utils import sofaExtractDelay, sph2cart

import numpy as np
import matplotlib.pyplot as plt

import librosa
import os

def render(audioSignal, fs,
           sofaFile,
           movementSeq,
           blockSize = 256,
           useDynamicITD = False,
           useDynamicILD = False,
           useHRIRinterpolation = True,
           useCrossfading = True,
           useInterpolatingConvolver = True):
    numBinauralObjects = 1
    numOutputChannels = 2

    numBlocks  = audioSignal.shape[0] // blockSize
    signalLength = blockSize * numBlocks
    
    context = visr.SignalFlowContext(period=blockSize, samplingFrequency=fs)

    renderer = DynamicHrirRenderer(context, "DynamicHrirRenderer", None,
                                   numberOfObjects=numBinauralObjects,
                                   sofaFile=sofaFile,
                                   headTracking=False,
                                   dynamicITD=useDynamicITD,
                                   dynamicILD=useDynamicILD,
                                   hrirInterpolation=useHRIRinterpolation,
                                   filterCrossfading=useCrossfading,
                                   interpolatingConvolver=useInterpolatingConvolver)

    result, messages = rrl.checkConnectionIntegrity(renderer)
    if not result:
        raise ValueError(messages)

    # Set input/output placeholders
    inputSignal = np.zeros((numBinauralObjects, signalLength), dtype=np.float32)
    inputSignal[0,:] = audioSignal[:signalLength]
    outputSignal = np.zeros((numOutputChannels, signalLength), dtype=np.float32)
    
    # Set initial position
    az = 0
    el = 0
    r = 1
    ps = objectmodel.PointSource(0)
    ps.position = sph2cart(np.array([az,el,r]))
    ps.level = 1.0
    ps.channels = [ps.objectId]

    # Set input flow
    flow = rrl.AudioSignalFlow(renderer)
    paramInput = flow.parameterReceivePort('objectVector')                 
    ov = paramInput.data()
    ov.set([ps])
    paramInput.swapBuffers()

    # Generate synthetic movement sequence

    for blockIdx in range(numBlocks):
        # Update params
        az = movementSeq[int(blockIdx % movementSeq.shape[0])]
        el = 0
        r = 1
        ps.position = sph2cart(np.array([az,el,r]))
        ov = paramInput.data()
        ov.set([ps])
        paramInput.swapBuffers()

        # process
        inputBlock = inputSignal[:, blockIdx*blockSize:(blockIdx+1)*blockSize]
        outputBlock = flow.process(inputBlock)
        outputSignal[:, blockIdx*blockSize:(blockIdx+1)*blockSize] = outputBlock
        
    outputSignal = np.asfortranarray(outputSignal)
    return outputSignal, fs

def generateStaticDataset(audioPaths, outputDir, 
                          sofaFile='hrir/HRIR_L2354.sofa',
                          n=100):
    os.makedirs(outputDir, exist_ok=True)

    fp = open(os.path.join(outputDir, 'annotation.csv'), 'w')
    fp.write("{},{}\n".format('file','azimuth'))
    fp.flush()

    for audioPath in audioPaths[:1]:
        filename = os.path.split(audioPath)[1]
        x, fs = librosa.load(audioPath)
        librosa.output.write_wav(os.path.join(outputDir, filename.split('.')[0] + '.wav'), x, fs)
        for i in range(n):
            print("Generating {} [i = {}]".format(audioPath, i))
            outputPath = os.path.join(outputDir, filename.split('.')[0] + '_{:02d}'.format(i) + '.wav')

            randomAzimuth = 2 * np.pi * np.random.rand()
            movementSeq = np.array([randomAzimuth])
            fp.write("{},{}\n".format(outputPath, randomAzimuth))
            fp.flush()

            out_x, out_fs = render(x, fs, sofaFile, movementSeq)
            librosa.output.write_wav(outputPath, out_x, out_fs)
    fp.close()

if __name__ == '__main__':
    audioPaths = glob.glob(os.path.join('carsSound/*'))
    generateStaticDataset(audioPaths, 'processedCarsSound')