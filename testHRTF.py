import argparse
import numpy as np
import time
import openal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--hrtf', type=str)
    args = parser.parse_args()

    if args.hrtf:
        hrtf_name = args.hrtf
    else:
        hrtf_name = None

    openal.oalInit()

    openal.oalInitHRTF(requested_hrtf=hrtf_name)

    listener = openal.Listener()
    listener.set_position((0,0,0))

    source = openal.oalOpen(args.filename)
    source.relative = True

    source.set_position((0.,0.,-1.))

    source.play()

    angle = 0
    while source.get_state() == openal.AL_PLAYING:
        time.sleep(0.005)

        angle += 0.01 * np.pi * 0.5
        if angle > np.pi:
            angle -= np.pi * 2.0

        source.set_position((np.sin(angle), 0., -np.cos(angle)))
        source.set_stereo_angles((np.pi/6.0-angle, -np.pi/6.0-angle))

    openal.oalQuit()