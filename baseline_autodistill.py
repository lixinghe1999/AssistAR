from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundingDINO(ontology=CaptionOntology({"adhesive tape": "adhesive tape"}))

# label all images in a folder called `context_images`
base_model.label(
  input_folder="./dataset/EgoObjects_mini/images",
  output_folder="./dataset/EgoObjects_mini/groundingdino_output",
)

# target_model = YOLOv8("yolov8n.pt")
# target_model.train("./dataset/EgoObjects_mini/groundingdino_output/data.yaml", epochs=10)
# run inference on the new model
# pred = target_model.predict("./dataset/EgoObjects_mini/images/3E8E6F25C49EB51513B4D2B75DBBE0A5_01_6.jpg", confidence=0.5)
# print(pred)