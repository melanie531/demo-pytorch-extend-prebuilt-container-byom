# demo-pytorch-extend-prebuilt-container-byom

This example demonstrates how to extend our prebuilt pytorch container for bring your own model hosting. 

In the *pytorch_local_mode_cifar_10.ipynb* notebook, there are three parts:
1. Part one shows how to extend our prebuilt pytorch container;
2. Part two shows how to train a Pytorch model using prebuilt pytorch training container;
3. Part three shows how to host the pretrained model on SageMaker using the extended hosting container and how to update the entry point script to include custom functions to preprocess the data, such as reading data from S3.