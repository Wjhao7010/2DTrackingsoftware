import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, cam, transforms=None, min_area = 30):
        self.transforms = transforms
        self.img_dir = os.path.join(path, cam, 'imgs')
        self.min_area = min_area

        # Load annotations
        ann_path = os.path.join(path, cam, "gt.txt")
        self.df = pd.read_csv(ann_path, sep=",")
        self.tags = ['background','zebrafish']

        # Get a sorted list of paths of annotated images - images without annotations are ignored
        self.imgs = sorted(self.df['filename'].unique())

    def __getitem__(self, idx):
        # load images
        img_file = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_file)#.convert("RGB")
        image_id = torch.tensor([idx])

        # Load bounding boxes and labels from the given frame
        bb = self.df[self.df['filename'] == self.imgs[idx]]

        boxes = []
        labels = []
        # Check for bounding boxes in the image
        for i in range(len(bb)):
            # Read the bounding box positions
            xmin = bb.loc[bb.index[i],'left']
            ymin = bb.loc[bb.index[i],'top']
            xmax = bb.loc[bb.index[i],'left'] + bb.loc[bb.index[i],'width']
            ymax = bb.loc[bb.index[i],'top'] + bb.loc[bb.index[i],'height']

            if (xmax-xmin)*(ymax-ymin) < self.min_area: # In order to not take totally occluded bounding boxes into account.
                continue


            boxes.append([xmin, ymin, xmax, ymax])

            # Convert labels from annotation tag to index number
            #labels.append(self.tags.index(bb.loc[bb.index[i],'Annotation tag']))
            #ipdb.set_trace()
            #labels.append(self.tags.index(bb.loc[bb.index[i],'id'])) # FIX
            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Set target attributes
        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

#    def get_events(self,delimiter):
#        """
#        An event is specified by the first part of the filename before a given delimiter
#
#        Example
#            filename: 2019_05_10_14_55_57-00000.png
#            delimiter: -
#            extension: .png
#            image number: 00000
#            event: 2019_05_10_14_55_57
#
#        """
#        events = []
#        for f in self.imgs:
#            tmp = f[:(f.find(delimiter))]
#            # Append unique event-names to the list
#            if tmp in events: continue
#            events.append(tmp)
#
#        return events
#
    def __len__(self):
        return len(self.imgs)
