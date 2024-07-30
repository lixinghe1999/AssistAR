from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget
import matplotlib.pyplot as plt

def request_detection(image_path, prompt):
    # Step 1: initialize the config
    token = "66a4d663b6a166e87e473eb50a0a7bc4"
    # token = None
    config = Config(token)

    # Step 2: initialize the client
    client = Client(config)

    # Step 3: run the task by DetectionTask class
    # image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
    # if you are processing local image file, upload them to DDS server to get the image url
    image_url = client.upload_file(image_path)

    task = DetectionTask(
        image_url=image_url,
        prompts=[TextPrompt(text=prompt)],
        targets=[DetectionTarget.BBox],  # detect both bbox and mask
        model=DetectionModel.GDino1_5_Pro,  # detect with GroundingDino-1.5-Pro model
    )

    client.run_task(task)
    result = task.result


    objects = result.objects  # the list of detected objects
    return objects
def parser_api(objects, image_path):
    '''
    input
    bbox [x1, y1, x2, y2]
    output
    ref: [class_name]
    box: [x1, y1, x2, y2
    '''
    image = plt.imread(image_path)
    img_height, img_width = image.shape[:2]
    ref = [obj.category for obj in objects]
    box = [[obj.bbox[0] / img_width, obj.bbox[1] / img_height, obj.bbox[2] / img_width, obj.bbox[3] / img_height] for obj in objects]
    return ref, box
if __name__ == '__main__':

    prompt = '.'.join(['cat', 'napkin', 'blanket', 'phone charger', 'boot', 'sofa', 'bowl', 'waste container', 'laptop charger', 'stool'])
    objects = request_detection("EA4C18B4867AABAFF1F2BC161B77909F_01_14.jpg", prompt)
    ref, box = parser_api(objects, "EA4C18B4867AABAFF1F2BC161B77909F_01_14.jpg")
    print(ref)
    print(box)
