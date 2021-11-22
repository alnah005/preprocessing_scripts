import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from collections import OrderedDict
import models.crnn as crnn
import os
import json


model_path = "netCRNN_15_24780.pth"
main_dir = "../umnDatasetSample_cropped/"
json_path = "detections_cropped.json"
output_json_path = "detections_cropped_with_text.json"
alphabet = "!\"#&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz| \\$@~£–—êüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ`\{\}"
json_detection_path_delimiter='/'

assert os.path.isfile(json_path)
with open(json_path, 'r') as openfile:
    detections = json.load(openfile)



def load_model_and_converter(pretrained_model_path, letters):
    model = crnn.CRNN(128, 1, len(letters)+1, 256, use_transformer=True)
    if torch.cuda.is_available():
        model = model.cuda()
        state_dict = torch.load(pretrained_model_path)
    else:
        state_dict = torch.load(pretrained_model_path,map_location=torch.device('cpu'))
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if "module." == k[:7]:
            name = k[7:]  # remove `module.`
        state_dict_rename[name] = v
    model.load_state_dict(state_dict_rename)

    converter = utils.strLabelConverter(letters)
    return model, converter

def load_image(img_path,transformer):
    image = Image.open(os.path.join(main_dir,img_path)).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    return image

model, converter = load_model_and_converter(model_path,alphabet)
model.eval()
transformer = dataset.resizeNormalize((720, 128))
for index,detection in enumerate(detections):
    img_name = detection["im_path"].split(json_detection_path_delimiter)[-1]
    image = load_image(os.path.join(main_dir,img_name),transformer)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    detections[index]["text"] = sim_pred


with open(output_json_path, "w") as outfile:
    json.dump(detections,outfile,indent=2)