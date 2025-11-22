# Predict flower name from an image with predict.py along with the probability 
# of that name. That is, you'll pass in a single image /path/to/image 
# and return the flower name and class probability.

# Basic usage: python predict.py /path/to/image checkpoint

# Options: 

#          * Return top K most likely classes: 
#            python predict.py input checkpoint --top_k 3 

#          * Use a mapping of categories to real names: 
#            python predict.py input checkpoint --category_names cat_to_name.json 

#          * Use GPU for inference: python predict.py input checkpoint --gpu


import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image

def get_input_args():

    parser = argparse.ArgumentParser(
        description="Predict flower name for an image using a pretrained model.")

    parser.add_argument("image_path", help="Path to input image")

    parser.add_argument("checkpoint", help="Path to model checkpoint")

    parser.add_argument("--top_k", type=int, default=5, 
                        help="Return top K most likely classes")
    
    parser.add_argument("--category_names", type=str, default=None, 
                        help="Path to JSON file mapping categories to names")
    
    parser.add_argument("--gpu", action="store_true", 
                        help="Use GPU if available")
    
    return parser.parse_args()


def load_checkpoint(filepath, device='cpu'):
    ''' Takes a checkpoint file as input and returns the model
    '''
    checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint['arch']

    if arch == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.fc = checkpoint['classifier']

    elif arch == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        model.classifier = checkpoint['classifier']

    elif arch == 'densenet121':
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        model.classifier = checkpoint['classifier']

    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    model.eval()

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image = Image.open(image)

    apply_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = apply_transformations(image)

    return image_tensor.numpy()


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using
        a trained deep learning model.
    '''

    # set model to evaluation mode
    model.eval()
    model.to(device)

    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0) # add dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    # convert logits to probabilities
    probabilities = torch.softmax(output, dim=1)

    # get top k probabilities and indices
    top_probabilities, top_indices = probabilities.topk(topk)
    top_probabilities = top_probabilities.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    # convert from indices to flower types using model.class_to_idx
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]

    return top_probabilities, top_classes


if __name__ == "__main__":
    args = get_input_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    if args.gpu and device != 'cuda':
        print("GPU requested but CUDA not available. Using CPU.")

    model = load_checkpoint(args.checkpoint, device=device)

    probs, classes = predict(args.image_path, model, 
                             device=device, topk=args.top_k)

    # map to flower names if json provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        names = [cat_to_name[c] for c in classes]
    else:
        names = classes

    # print results
    print("\nTop Predictions:")
    for i in range(len(names)):
        print(f"{i + 1}: {names[i]} with probability {probs[i]:.4f}")
    print()
