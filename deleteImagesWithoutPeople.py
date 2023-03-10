import glob
import os
import sys

from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering

if len(sys.argv) < 2:
    print("Please provide a directory path as an argument.")
    sys.exit(1)

path = sys.argv[1]  # Get the path argument from the command line

print("loading blip model")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Use the glob module to find all image files in the path
image_files = glob.glob(os.path.join(path, '*.jpg')) + \
    glob.glob(os.path.join(path, '*.png'))

print("found {} images, start parsing...".format(len(image_files)))

question = "How many people are in the picture?"
for image_file in image_files:
    raw_image = Image.open(image_file)
    inputs = processor(raw_image, text=question, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(out[0], skip_special_tokens=True)
    img_name = os.path.basename(image_file)
    if int(answer) == 0:
        print("delete {}".format(image_file))
        os.remove(image_file)
    else:
        print("{}: found {} people".format(img_name, answer))
