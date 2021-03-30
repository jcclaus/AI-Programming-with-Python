# PROGRAMMER: John Claus
# DATE CREATED: 12/20/2020                          
# REVISED DATE: 
# PURPOSE: This program predicts the type of flower from an image via class selection 
# from a previously trained NN model. This program also gives the user inputed topk      
# probailities and respect flower type of the picture with a default value of 5.
# The pretrained NN is loaded from an exiting checkpoint created during the NN training.
# Additionally, the user can select whether the GPU will be used to make the prediction.
# If the GPU is unavailable, the program will default to using the cpu regardless of the 
# user selection. The user can also select the file path for the json file used
# to covert the classes to the name of flower.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py data_dir image <image file path> checkpoint <checkpoint file path> 
#             --gpu <True or False> --topk <# of top predictions>
#             --json <json file path> 
#   Example call:
#    python predict.py Orange_Dahlia.jpg checkpoint --gpu False --topk 3 --json cat_to_name.json   
##

# Imports python modules
from time import time, sleep
from get_input_args import predict_input_args
from train import build_model
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os

def load_model(checkpoint):
    """
    This funtion builds the NN model with its associated weights from the 
    checkpoint folder. It returns this model and the class to name dictionary for
    later use in the identification of the flower type.
    """
    # Checks that directory exists. If not, uses the default directory
    if os.path.isdir(checkpoint) == False:
        print("Checkpoint path does not exist. Using the default checkpoint folder.\n")
        checkpoint = 'checkpoint'
    # Loads the number of hidden neurons
    hidden = torch.load(checkpoint + '/hidden.pth')
    # Loads the VGG type used
    arch = torch.load(checkpoint + '/vgg_ver.pth')
    # Builds the initial model
    model, hidden = build_model(arch, hidden)
    # Converts to the cpu to upload
    model.to('cpu')
    # Loads the state dictionary and weights from checkpoint.pth file
    state_dict = torch.load(checkpoint + '/checkpoint.pth')
    # Loads the state dictionary into the model
    model.load_state_dict(state_dict)
    # Loads the class to index map into the class_to_idx dictionary
    class_to_idx = torch.load(checkpoint + '/class_to_idx.pth')
    return model, class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Preprocesses the image to resize, center crop, normalize and covert to a tensor
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Calls the preprocess on the image file
    img_tensor = preprocess(image)
    # Insures the the image tensor doesn't require gradient descent
    img_tensor.requires_grad_(False)

    return img_tensor

def predict(image_path, model, class_to_idx, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Defines the device dependent upon what is currently operating, assigns the model
    # to that device and sets the model to evaluation mode
    device = torch.device("cuda" if ((torch.cuda.is_available()) and (gpu == False)) else "cpu")
    model.to(device)
    model.eval()
    
    # Verify that the filepath exists. If not, then uses the default flower picture
    if os.path.isfile(image_path) == False:
        print("The file path entered for the flower does not exist. Using the default image.\n")
        image_path = 'Orange_Dahlia.jpg'
    # Opens up the image using PIL, processes that image from the process_image() function
    # into a tensor, adds an additional dimension with unsqueeze(), and asigns the tensor object
    # to the device type being used  
    img_pil = Image.open(image_path)
    img_tensor = process_image(img_pil)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    
    # Pushes the image tensor forward through the NN, coverts the log output into 
    # probabilities, finds the topk number (5) of top probabilites and predictions,
    # converts those to the non-GPU format, and converts the tensor objects into a numpy lists
    out_logps = model.forward(img_tensor)
    out_ps = torch.exp(out_logps)
    top_prob, top_index = out_ps.topk(topk)  
    top_prob = top_prob.to("cpu")
    top_index = top_index.to("cpu")
    top_prob = top_prob.detach().numpy().tolist()[0] 
    top_index = top_index.detach().numpy().tolist()[0]
    
    # Reverses the dictionary keys and values of class to index into index to class
    idx_to_class = {val: key for key, val in class_to_idx.items()}  
    # Uses the index to class dictionary to get a top category list from the top index list
    top_cats = [idx_to_class[index] for index in top_index]
    
    return top_prob, top_cats


def main():
    # Collect start time to measure program speed
    start_time = time()
    # Collect input arguments
    in_arg = predict_input_args()
    # Load the trained model for use in the prediction
    loaded_model, class_to_index = load_model(in_arg.checkpoint)
    # Calls the prediction function to get lists of the top
    # categories and their associated probabilities
    probs, cats = predict(in_arg.image, loaded_model, class_to_index, in_arg.topk, in_arg.gpu)
    # Load the category to name dictionary
    with open(in_arg.json, 'r') as f:
        cat_to_name = json.load(f)
    # Converts the categories into the specific flower names
    flowers = [cat_to_name[cat] for cat in cats]
    # Print topk flowers and probabilities
    n = 0
    for flower in flowers:
        print("{}. Flower: {:20}  Probability: {:.3f}   ".format(n+1,flower,probs[n]))
        n += 1
    # Collect the end time to measure program speed
    end_time = time()
    # Determine total program time and print
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Call to main function
if __name__ == "__main__":
    main()    