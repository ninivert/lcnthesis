#!/bin/bash
# extracts first and last frame of every video (files ending in *.mp4)
# output image has max 400px height
# this is done for the PDF render

for filename in *.mp4; do
	echo "$filename";
	ffmpeg -y -i $filename -sseof -1 -i $filename -map 0:v -vframes 1 -q:v 1 -qmin 1 "$filename-1.jpg" -map 1:v -q:v 1 -qmin 1 -update 1 "$filename-2.jpg";
	# convert "$filename-1.jpg" -resize 'x500>' "$filename-1.jpg";
	# convert "$filename-2.jpg" -resize 'x500>' "$filename-2.jpg";
done