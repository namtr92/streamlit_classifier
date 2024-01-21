import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import densenet121
from torchvision import io

# Load the pre-trained DenseNet121 model
model = densenet121(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    st.title("Image Classifier with DenseNet121")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Make predictions
        prediction = predict(image)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
