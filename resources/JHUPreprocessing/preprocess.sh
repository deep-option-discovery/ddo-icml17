#!/bin/bash

for f in *.avi
do
ffmpeg -i $f -an -vcodec rawvideo processed-$f
done
