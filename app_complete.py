import os
import time
import json
import boto3
import pandas as pd
import logging 
import streamlit as st
import time
from botocore.exceptions import ClientError


st.set_page_config(layout="wide")
logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
BUCKET='ENTER NAME OF S3 BUCKET'
ENDPOINT='ENTER NAME OF DEPLOYED MODEL ENDPOINT'
   
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

def query(request,params):
    s3_client = boto3.client('s3')
    runtime_sm_client=boto3.client('sagemaker-runtime')
    inputs = dict(   
    input_payload=json.dumps(request),
    )

    mode=params['controlnet']

    # invoke the setup_conda model to create the shared conda environment
      
    payload = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [1],
                "datatype": "BYTES",
                "data": ["hello"],  # dummy data not used by the model
            }
        ]
    }

    response = runtime_sm_client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/octet-stream",
        Body=json.dumps(payload),
        TargetModel="setup_conda.tar.gz",
    )
    
    #payload must be in the structure specified in the config.pbtxt file
    payload = {
        "inputs": [
            {"name": name, "shape": [1, 1], "datatype": "BYTES", "data": [data]}
            for name, data in inputs.items()
        ]
    }
    
    try:
        response = runtime_sm_client.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/octet-stream",
            Body=json.dumps(payload),
            TargetModel=f"{mode}.tar.gz", # specify the target model to run inference on
            Accept="application/json"
        )
    except:
        time.sleep(2)
        response = runtime_sm_client.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/octet-stream",
            Body=json.dumps(payload),
            TargetModel=f"{mode}.tar.gz", # specify the target model to run inference on
            Accept="application/json"
        )
    output = json.loads(response["Body"].read().decode("utf8"))["outputs"]   
    
    image=s3_client.get_object(Bucket=BUCKET, Key=f"{request['output']}/{output[0]['data'][0].split('/',4)[-1]}")["Body"].read()
    tech=s3_client.get_object(Bucket=BUCKET, Key=f"{request['output']}/{mode}_{output[0]['data'][0].split('/',4)[-1]}")["Body"].read()
    return image, tech
                
def action_doc(params):
    st.title('Unleashing Creativity: How Generative AI enhances guided ad-creatives content generation with AWS')
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader('Upload an image')
        if file is not None:
            file_name=str(file.name)
            st.image(file)
                   
            
    with col2:        
        with st.expander("Sample Prompt"):
            st.write("Prompt: metal orange colored car, complete car, colour photo, outdoors in a pleasant landscape, realistic, high quality  \nNegative prompt: cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, blurry, bad anatomy, bad proportions" )
        input_question = st.text_input('**Please pass a prompt:**', '')
        neg_prompt = st.text_input('**Negative prompt (Optional):**', '')
        if st.button('Generate Image') and len(input_question) > 3 and file is not None:
       
            s3_client = boto3.client('s3')
            s3_client.put_object(Body=file, Bucket=BUCKET, Key=file_name)            
            n_p="cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, blurry, bad anatomy, bad proportions"
            n_p=neg_prompt if neg_prompt else n_p
            request={"prompt":input_question,
             "negative_prompt":n_p,
             "image_uri":f's3://{BUCKET}/{file_name}',
             "scale": params['scale'],
             "steps":params['steps'],
             "low_threshold":params['low_thresh'],
             "high_threshold":params['high_thresh'],
             "seed": params['seed'],
             "output":"output"
            }
            image, tech=query(request,params)
            st.write('<p style="font-size:22px; color:blue;">Generated Image</p>',unsafe_allow_html=True)
            st.image(image)
            st.write(f'<p style="font-size:22px; color:blue;">{params["controlnet"].upper()}</p>',unsafe_allow_html=True)
            st.image(tech)     
            
def app_sidebar():
    with st.sidebar:
        st.write('## How to use:')
        description = """This app lets you bring an image and modify it using prompts. 
                        Take advantage of the Stable Diffusion model and ControlNet techniques to reimagine your image."""
        st.write(description)
        st.write('---')
        st.write('### User Preference')
        controlnet = st.selectbox('Choose ControlNet Technique', options=['canny','depth','mlsd', 'scribble', 'hed','openpose'])
        scale = st.slider('scale', min_value=0., max_value=2., value=0.5, step=0.1)
        steps = st.slider('steps', min_value=0., max_value=50., value=20., step=1.)
        low_thresh = st.slider('low_threshold', min_value=0., max_value=500., value=100., step=10.)
        high_thresh = st.slider("high_threshold", min_value=0., max_value=1000., value=200., step=10.)
        seed = st.slider('seed', min_value=0., max_value=1000., value=100., step=10.)
        params = {'scale':scale, 'steps':steps, 'low_thresh':low_thresh, 'high_thresh':high_thresh,'seed':seed, 'controlnet':controlnet}
        return params 
        
        
def main():
    params=app_sidebar()
    action_doc(params)


if __name__ == '__main__':
    main()
