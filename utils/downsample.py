import os
def downsample_clip(data_folder = 'dataset/v2/clips',  output_folder = 'dataset/v2/downsampled', clip=None, split_frames=True):
    '''
    Simulate the downsample process of the video clips on smart glasses
    '''
    ffmpeg_command1 = 'ffmpeg -i {} -c:v h264  -b:v 150k -x264-params keyint=30:scenecut=0 -vf "scale=640:-2" {} -y -loglevel quiet'
    ffmpeg_command2 = 'ffmpeg -i {} -vf fps=30 {} -y -loglevel quiet'
    clip_path = os.path.join(data_folder, clip)
    output_path = os.path.join(output_folder, clip)
    _ffmpeg_command = ffmpeg_command1.format(clip_path, output_path)
    os.system(_ffmpeg_command)
    ori_size, out_size = os.path.getsize(clip_path), os.path.getsize(output_path)
    print('Original:', ori_size, 'Downsampled:', out_size, 'ratio', out_size/ori_size)
    if split_frames: # split video to frames
        os.makedirs(output_path[:-4], exist_ok=True)
        _ffmpeg_command = ffmpeg_command2.format(output_path, output_path[:-4] + '/%04d.jpg')
        os.system(_ffmpeg_command)
        frames_size = sum([os.path.getsize(os.path.join(output_path[:-4], frame)) for frame in os.listdir(output_path[:-4])]) 
        print('Original:', ori_size, 'frames_size', frames_size, 'ratio', frames_size/ori_size)
        os.remove(output_path)
        
def split_frames(data_folder = 'dataset/v2/clips',  output_folder = 'dataset/v2/downsampled', clip=None, split_frames=True):
    '''
    Simulate the downsample process of the video clips on smart glasses
    '''
    ffmpeg_command2 = 'ffmpeg -i {} -vf fps=30 {} -y -loglevel quiet'
    clip_path = os.path.join(data_folder, clip)
    output_path = os.path.join(output_folder, clip)

    os.makedirs(output_path[:-4], exist_ok=True)
    _ffmpeg_command = ffmpeg_command2.format(clip_path, output_path[:-4] + '/%04d.jpg')
    os.system(_ffmpeg_command)

    ori_size = os.path.getsize(clip_path)
    frames_size = sum([os.path.getsize(os.path.join(output_path[:-4], frame)) for frame in os.listdir(output_path[:-4])])
    print('Original:', ori_size, 'frames_size', frames_size, 'ratio', frames_size/ori_size)

def roi_h264(input_video='input_video.mp4', method='blur', roi=(100, 100, 300, 200)):
    '''
    please use the following to make sure opencv is ok with h264
    conda install -c conda-forge opencv
    '''
    import cv2
    import os

    # Ope`n the video file
    cap = cv2.VideoCapture(input_video)

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the region of interest (ROI)
    roi_x, roi_y, roi_width, roi_height = roi
    roi_ratio = (roi_height * roi_width) / (width * height)

    # codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*[chr(codec & 0XFF), chr((codec >> 8) & 0XFF), chr((codec >> 16) & 0XFF), chr((codec >> 24) & 0XFF)])
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 

    out_roi = cv2.VideoWriter(input_video.replace('.mp4', '_roi.mp4'), fourcc, fps, (width, height))
    blur_kernel_size = (15, 15)

    while True:
        # Read a frame from the input video
        ret, frame = cap.read()
        if not ret:
            break
        # Create a copy of the frame for processing
        processed_frame = frame.copy() 

        if method == 'blur':
            # Apply blur to the non-ROI area
            processed_frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            processed_frame[0:roi_y, :] = cv2.GaussianBlur(frame[0:roi_y, :], blur_kernel_size, 0)
            processed_frame[roi_y+roi_height:, :] = cv2.GaussianBlur(frame[roi_y+roi_height:, :], blur_kernel_size, 0)
            processed_frame[:, 0:roi_x] = cv2.GaussianBlur(frame[:, 0:roi_x], blur_kernel_size, 0)
            processed_frame[:, roi_x+roi_width:] = cv2.GaussianBlur(frame[:, roi_x+roi_width:], blur_kernel_size, 0)
        elif method == 'mask':
            # Apply zeros to the non-ROI area
            processed_frame[0:roi_y, :] = 0
            processed_frame[roi_y+roi_height:, :] = 0
            processed_frame[:, 0:roi_x] = 0
            processed_frame[:, roi_x+roi_width:] = 0
        else: # do nothing
            pass
        # Write the processed frame to the output video
        out_roi.write(processed_frame)

    # Release the video capture and writer
    cap.release()
    out_roi.release()
    ori_size, roi_size = os.path.getsize(input_video), os.path.getsize(input_video.replace('.mp4', '_roi.mp4'))
    print('ratio', round(roi_size/ori_size,3 ), 'ROI ratio', round(roi_ratio,3))
def crop_first_second(input_video='input_video.mp4', output_video='output_video.mp4'):
    import ffmpeg
    # Load the input video
    input_video = ffmpeg.input(input_video)

# Crop the first second of the video
    cropped_video = input_video.filter('trim', duration=1)

    # Output the cropped video
    output_video = ffmpeg.output(cropped_video, output_video)
    ffmpeg.run(output_video)
def evaluation_psnr(input_video='input_video.mp4', output_video='output_video.mp4', roi=(100, 100, 300, 200)):
    import cv2
    cap1 = cv2.VideoCapture(input_video)
    cap2 = cv2.VideoCapture(output_video)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if the videos have the same dimensions
    if width1 != width2 or height1 != height2:
        print("Error: The two videos have different dimensions.")
        exit()

    # Initialize the PSNR sum and frame count
    psnr_sum = 0
    frame_count = 0

    # Loop through the frames and calculate the PSNR
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break
        if roi is not None:
            psnr = cv2.PSNR(frame1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], frame2[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])
        else:
            psnr = cv2.PSNR(frame1, frame2)
        psnr_sum += psnr
        frame_count += 1

    # Calculate the average PSNR
    avg_psnr = psnr_sum / frame_count

    print(f"The PSNR: {avg_psnr:.2f} dB under {roi} ROI")

    # Release the video capture objects
    cap1.release()
    cap2.release()
    return avg_psnr

if __name__ == '__main__':
    # down_sample(split_frames=True)
    # crop_first_second('dataset/example/orange_pick.mp4', 'dataset/example/orange_pick_cropped.mp4')
    # blur_ffmpeg('dataset/example/orange_pick_cropped.mp4')
    
    roi_h264('dataset/example/orange_pick_cropped.mp4', method='mask', roi=(100, 100, 300, 200))
    evaluation_psnr('dataset/example/orange_pick_cropped.mp4', 'dataset/example/orange_pick_cropped_roi.mp4', roi=(100, 100, 300, 200)) 
    evaluation_psnr('dataset/example/orange_pick_cropped.mp4', 'dataset/example/orange_pick_cropped_roi.mp4', roi=None) 

    roi_h264('dataset/example/orange_pick_cropped.mp4', method='blur', roi=(100, 100, 300, 200))
    evaluation_psnr('dataset/example/orange_pick_cropped.mp4', 'dataset/example/orange_pick_cropped_roi.mp4', roi=(100, 100, 300, 200)) 
    evaluation_psnr('dataset/example/orange_pick_cropped.mp4', 'dataset/example/orange_pick_cropped_roi.mp4', roi=None) 