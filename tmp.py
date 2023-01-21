import json
import boto3
from PIL import Image
import numpy as np


im = Image.open('daisy.jpg')
im_resized = im.resize((28, 28))
im_np = np.array(im_resized)
im_np = im_np / 255.0
im_np = im_np.tolist()



# Create a SageMaker client
client = boto3.client('sagemaker-runtime', region_name= 'ap-south-1')

# Set the endpoint name and model to test
endpoint_name = 'https://runtime.sagemaker.ap-south-1.amazonaws.com/endpoints/pytorch-inference-2023-01-17-15-13-39-342'

# Send a test request to the endpoint
response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=json.dumps(im_np))

# Print the response
print(response['Body'].read())