import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import geopandas

from utils_cifar import get_test_data_loader

try:
    from sagemaker_inference import environment
except:
    from sagemaker_training import environment
    
from sagemaker_inference import (
    content_types,
    decoder,
    encoder,
    errors,
    utils,
)

import boto3
import ast
import json

s3_client = boto3.client('s3')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

device = "cuda" if torch.cuda.is_available() else "cpu"


# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_fn(model_dir):
    """
    Load the model for inference
    """

    logger.info("model_fn")
    
    model = Net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    input_data = data.to(device)
    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            output = model.forward(input_data)

    return output

def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor,
        depending if cuda is available.
    """
#     np_array = decoder.decode(input_data, content_type)
    logger.info("request type: {}".format(content_type))

    request = ast.literal_eval(input_data)
    s3_obj = s3_client.get_object(Bucket=request['data_source'].split('/')[2], Key='/'.join(request['data_source'].split('/')[3:]))
    body = json.loads(s3_obj['Body'].read())
    logger.info(body)
        
    testloader = get_test_data_loader()
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    logger.info("GroundTruth: ", " ".join("%4s" % classes[labels[j]] for j in range(4)))
    np_array = images.numpy()
    tensor = torch.FloatTensor(
        np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    
    return tensor.to(device)

def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized
    Returns: output data serialized
    """
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy().tolist()
    
    logger.info("accept: {}".format(accept))
    for content_type in utils.parse_accept(accept):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(prediction, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(accept)