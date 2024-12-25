import torch, cv2, os
from torchvision import models, transforms
from PIL import Image

# class dictionary
classes = [
    'Expressionism', 
    'Fauvism',
    'High_Renaissance',
    'Impressionism',
    'Mannerism_Late_Renaissance',
    'Naive_Art_Primitivism',
    'New_Realism',
    'Northern_Renaissance',
    'Pointillism',
    'Pop_Art',
    'Post_Impressionism',
    'Realism',
    'Rococo',
    'Romanticism',
    'Symbolism'
]

##### START OF AI GENERATED CODE (OpenAI ChatGPT) #####

# Transform image into correct formatting
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for consistency
    transforms.ToTensor(),          # Convert to tensor formatting
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on ImageNet stats
])

##### END OF AI GENERATED CODE (OpenAI ChatGPT) #####

# initialize the model
model = models.resnet50(pretrained=False) # want to use my own weights/classes

# modify to match the number of classes
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classes))  # plug in 15 custom classes

# load model's weights
model.load_state_dict(torch.load('model_10_epoch.pth'))

# set the model to evaluation
model.eval()

##### START OF AI GENERATED CODE (OpenAI ChatGPT) #####

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to test the model on an image
def test_model(image_path):
    # Open an image
    image = Image.open(image_path)
    
    # Apply the transformations to the image
    image = transform(image)
    
    # Add an extra batch dimension
    image = image.unsqueeze(0)  # Adding batch dimension: [1, 3, 224, 224]
    
    # Move the image to the same device as the model
    image = image.to(device)

    # Perform inference
    with torch.no_grad():  # Disable gradient tracking for inference
        output = model(image)
    
    # Get the predicted class (index of the max value)
    _, predicted = torch.max(output, 1)
    
    # Get the predicted class label from the manually defined classes
    class_idx = predicted.item()
    class_name = classes[class_idx]  # Use the manually defined classes list
    
    return(f"Predicted Class: {class_name}")

##### END OF AI GENERATED CODE (OpenAI ChatGPT) #####

for filename in os.listdir('TestImages'):
    file_path = os.path.join('TestImages', filename)

    # image for display
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # resize
    ratio = 500.0 / img.shape[0]
    dim = (int(img.shape[1] * ratio), 500)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # display
    cv2.imshow(test_model(file_path), img)

    # exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()