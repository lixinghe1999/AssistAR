import os
data_folder = 'dataset/v2/clips'
output_folder = 'dataset/v2/downsampled'
ffmpeg_command1 = 'ffmpeg -i {} -c:v h264  -b:v 1M -x264-params keyint=30:scenecut=0 -s 640x480 {} -y'
ffmpeg_command2 = 'ffmpeg -i {} -vf fps=10 {} -y'
for clip in os.listdir(data_folder):
    clip_path = os.path.join(data_folder, clip)
    output_path = os.path.join(output_folder, clip)
    _ffmpeg_command = ffmpeg_command1.format(clip_path, output_path)
    os.system(_ffmpeg_command)

    os.makedirs(output_path[:-4], exist_ok=True)
    _ffmpeg_command = ffmpeg_command2.format(output_path, output_path[:-4] + '/%04d.jpg')
    os.system(_ffmpeg_command)
    ori_size, out_size = os.path.getsize(clip_path)/ 1024 / 1024, os.path.getsize(output_path)/ 1024 / 1024
    frames_size = sum([os.path.getsize(os.path.join(output_path[:-4], frame)) for frame in os.listdir(output_path[:-4])]) / 1024 / 1024
    print('Original:', ori_size, 'Downsampled:', out_size, 'frames', frames_size, 'ratio', out_size/ori_size)
    break