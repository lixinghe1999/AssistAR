
# first convert to 720p, then 480p, then 144p
# ffmpeg -i example.mp4 -vf scale=1280:720 720p.mp4
# ffmpeg -i example.mp4 -vf scale=640:480 480p.mp4
# ffmpeg -i example.mp4 -vf scale=176:144 144p.mp4

# INPUTs=('720p.mp4' '480p.mp4' '144p.mp4')
# for i in "${INPUTs[@]}"
# do
# OUTPUT="128kbps_${i}"
# ffmpeg -i $i -b:v 128k -c:v libx264 $OUTPUT
# done

INPUTs=('720p.mp4' '480p.mp4' '144p.mp4')
for i in "${INPUTs[@]}"
do
OUTPUT="CRF41_${i}"
ffmpeg -i $i -crf 41 -c:v libx264 $OUTPUT
done