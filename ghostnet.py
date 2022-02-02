import os
from PIL import Image
from torchvision import transforms
import torch
from dataset import CovidImageDataset
import h5py
from tqdm import tqdm
import itertools

# preferences
data_base = '/hkfs/work/workspace/scratch/im9193-health_challenge/data'
data_base = '/home/dmeier/AI-HERO/data'
seed_worker=42

dataset = CovidImageDataset(
    os.path.join(data_base, 'train.csv'),
    os.path.join(data_base, 'imgs'), transform='resize_rotate_crop',rgb_mode=True)

sample_size = len(dataset)
trainset = torch.utils.data.random_split(dataset, [sample_size, len(dataset)-sample_size])[0]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=20,
                                          worker_init_fn=seed_worker)
                                              
# load model           
model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
model.eval()

if torch.cuda.is_available():
    model.to('cuda')

model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

outputs = []
labels = []
for input_batch, label_batch in tqdm(trainloader):
    # move the input and model to GPU for speed if available
    #if torch.cuda.is_available():
    #    input_batch = input_batch.to('cuda')


    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output, dim=0)
    labels.append(label_batch)
    outputs.append(probabilities)
#    print(probabilities.shape)

outputs = torch.cat(outputs)
labels = torch.cat(labels)

f = h5py.File('output.h5', 'w')
f.create_dataset("imgs", data=outputs)
f.create_dataset("labels", data=labels)
f.close()
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.

#print(probabilities)

# Read the categories
#with open("imagenet_classes.txt", "r") as f:
#    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
#top5_prob, top5_catid = torch.topk(probabilities, 10)
#for i in range(top5_prob.size(0)):
#    print(categories[top5_catid[i]], top5_prob[i].item())
