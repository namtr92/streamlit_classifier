import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
from torchvision import io
from streamlit_cropper import st_cropper

import torch.nn as nn
st.set_page_config(layout="wide")
classes = ['black',
 'brown',
 'conveyor-hole',
 'embose-tape',
 'glue',
 'ink-blue',
 'ink-green',
 'ink-red',
 'ink-yellow',
 'metal-tape',
 'others',
 'tear',
 'texture',
 'white-clump',
 'white-tape',
 'yellow']

# Load the pre-trained DenseNet121 model
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(512, len(classes)) 
# model.load_state_dict(torch.load('deploy/resnet18_pad.pth',map_location=torch.device('cpu')))

import timm
model = timm.create_model('deit_tiny_patch16_224',pretrained=False,num_classes=len(classes))
model.load_state_dict(torch.load('deploy/deit_tiny_pad.pth',map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict(image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Streamlit app
def main():
    st.title("Image Classifier with ResNet18")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    aspect_dict = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "2:3": (2, 3),
        "Free": None
    }
    aspect_ratio = aspect_dict[aspect_choice]
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        with col1:
            cropped_img = st_cropper(image, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
        with col2:
            prediction = predict(cropped_img)
            st.write("Prediction:", classes[prediction])
            st.image(cropped_img, caption="Uploaded Image.", use_column_width='auto')
            # Make predictions
            
        

if __name__ == "__main__":
    main()
