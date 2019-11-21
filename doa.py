import numpy as np
import matplotlib.pyplot as plt
from SOFASonix import SOFAFile
import librosa
import scipy.signal
from tqdm import tqdm

def findDirection(x_l, x_r, sofaFile, blockSize=128):
    """
    A localization algorithm based on head-related transfer functions

    GERHARD, Maike, et al. "A localization algorithm based on head-related transfer functions."
    """
    sofa = SOFAFile.load(sofaFile)
    positions = sofa.SourcePosition
    ir = sofa.data_ir
    ir_l = ir[:,0,:]
    ir_r = ir[:,1,:]

    numBlocks = x_l.shape[0] // blockSize
    directions = np.empty((numBlocks, 3))
    for block in tqdm(range(numBlocks)):
        bestPosition = None
        minDist = np.inf
        x_l_block = np.fft.fft(x_l[block * blockSize: (block+1) * blockSize])
        x_r_block = np.fft.fft(x_r[block * blockSize: (block+1) * blockSize])

        for idx in range(len(positions)):
            position = positions[idx]
            (_, elev, _) = position
            if np.abs(elev) > 1e-1: 
                continue

            # Find optimal signal 
            x_p = np.empty((blockSize,))
            ir_l_freq = np.fft.fft(ir_l[idx])
            ir_r_freq = np.fft.fft(ir_r[idx])
            for k in range(blockSize):
                x_p[k] = (x_l_block[k] * np.conj(ir_l_freq[k]) + x_r_block[k] * np.conj(ir_r_freq[k])) / (np.abs(ir_l_freq[k])**2 + np.abs(ir_r_freq[k])**2)
            x_p = np.fft.fft(x_p)

            # Compute distance
            dist = np.linalg.norm(np.concatenate((x_l_block - x_p * ir_l_freq, x_r_block - x_p * ir_r_freq)))

            if dist < minDist:
                minDist = dist
                bestPosition = position
        
        print(bestPosition[0])
        directions[block] = bestPosition

    return directions



if __name__ == '__main__':
    x, fs = librosa.load('processedCarsSound/g2BSnZZZtRs_410000_420000_00.wav', mono=False)
    sofaFile = 'hrir/HRIR_L2354.sofa'
    directions = findDirection(x[0], x[1], sofaFile)

    gt_az = 1.5486159839802436
    r = np.linspace(0, 10.0, directions.shape[0])

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar', facecolor='#d5de9c')
    ax.plot(gt_az * np.ones_like(r), r, color='blue', ls='--', label='ground truth')
    ax.plot(directions[:,0] / 360.0 * (2 * np.pi), r, '.', color='red', label='prediction')
    ax.legend()
    plt.savefig('out.png', dpi=300)
    plt.show()  