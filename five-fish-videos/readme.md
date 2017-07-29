The videos are too big, so I can't upload them to github.

JD:  I used ffmpeg to compress.  command:
ffmpeg -i video-simulated-five-fish-real.avi -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" video-simulated-five-fish-real.mp4

(the -vf flag dealt with the error for an odd number of x-dimension pixels)
