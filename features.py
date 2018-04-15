import torch.nn as nn
import torchvision.models as models
import vqa.VQA as VQA
from progressbar import Bar, ETA, Percentage, ProgressBar
import pickle

dataset = VQA(annotation_file="",question_file="")
widgets = ['Generating Features', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
           ' ', ETA()]
pbar = ProgressBar(widgets=widgets)
if __name__ == '__main__':
    data = []
    net = models.resnet18()
    net.eval()
    net.feature = True
    image_ids = dataset.getImgIds()
    for i in pbar(image_ids):
        image = cv2.imread(os.path.join(root,'COCO_train2014_'+ str(i).zfill(12) + '.jpg'))
        """
        Normalization stuff here
        """
        data.append(net(image))
    print("Done")
    pickle.dump(data,open("features.pkl","wb"))
