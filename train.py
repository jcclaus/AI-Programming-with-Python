# PROGRAMMER: John Claus
# DATE CREATED: 12/20/2020                          
# REVISED DATE: 
# PURPOSE: Trains a user selected TorchVision VGG model which has had the classifier section modified
#          to incorporate a custon hidden layer with a user defined number of units. The user can also
#          define whether this training set will be done with a GPU enabled or not.Additionally, the 
#          user can define the training rate and the number epochs used to train the data set. At each
#          epoch, the program will print out the training loss, the validation loss, and the validation
#          accuracy. 
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py data_dir --lr <learning rate> --model <vgg19 or vgg16> 
#             --hidden <# of neurons in hidden layer> --epochs <# of training epochs>
#             --gpu <True or False> --chk <checkpoint folder>
#   Example call:
#    python train.py flowers --lr 0.005 --model vgg16 --hidden 2048 --epochs 8 --gpu False --chk alt_checkpoint
##

# Imports python modules
from time import time, sleep
from get_input_args import train_input_args
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os

def load_data(data_dir):
    """
    Loads the training, validation, and testing data from the data_dir directory
    and associated specific directories for the taining, validation, and testing data.
    
    Transforms the loaded data either as a standard or training transform. The standard 
    transform will be used on the validation and test sets and only where it resizes, 
    centers, nortmalizes, and coverts the images to tensors. The train transform is used 
    for the training data. This transform also radomly rotates,resizes, and flips the images
    so that the train set may better train the network.
    
    The image_dataset dictionary uses three keys: 'train', 'valid', and 'test' which
    call the collection of transformed images.
    
    The dataloaders dictionary uses three keys: 'train', 'valid', and 'test' which
    call the 64 batches of shuffled labels and images at a time.
    """
    if os.path.isdir(data_dir) == False:
        print("Data directory does not exist. Using the default directory.\n")
        data_dir = 'flowers'
    # Define the training, validation, and testing directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the standard and training image transforms
    standard_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    # Define the image_datasets dictionary
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=train_transform),
                 'valid': datasets.ImageFolder(valid_dir, transform=standard_transform),
                 'test' : datasets.ImageFolder(valid_dir, transform=standard_transform)}
    
    # Define the dataloaders dictionary
    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
              'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
              'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)}
    
    # Moves the class to index map to an output variable
    class_to_index = image_datasets['train'].class_to_idx
    
    # Returns the dataloaders dictionary and class to index map
    return dataloaders, class_to_index

def build_model(model_type, hidden):
    """
    Loads the VGG model type of either vgg16 or vgg19 and then adds a new classification section with given number 
    of neurons in the hidden layer. If the model type entered is not vgg16 or vgg19, it will return an error
    message and default to the vgg16 model. This also checks if the number hidden layer 2 neurons is less than 1.
    If it is, then it returns an error and uses the default of 4096. The new classifier has an input layer that 
    matches the original's and steps down with 2 layers to a output of 102 possible flower classifications. It
    also has a drop rate of 25% included to improve help training overcome local minima. 
    """
    
    # Configures TorchVision VGG model type from input
    if (model_type == 'vgg19'):
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        if (model_type != 'vgg16'):
            print("Incorrect model type. Using default of VGG16.\n")
            
    # Insures model parameters are not modified  
    for param in model.parameters():
        param.requires_grad = False
    
    # Error checking for hidden layer 2 count value
    if (hidden <= 0):
        hidden = 4096
        print("Hidden layer neuron count cannot be less than 1, using default value of 4096.\n")
        
    # New classifier definition
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden)),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.25)),
                          ('fc2', nn.Linear(hidden, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Replaces the original model classifier with the newly created one
    model.classifier = classifier
    
    # Returns the new model
    return model, hidden

def train_model(model, dataloaders, gpu, epochs, learn_rate, sched_step = 4, sched_gamma = 0.1, print_error = True):
    """
    This function trains the given model input with the user provided dataloaders, epochs, learn_rate, schedule steps,
    and schedule gamma. It uses negative likelihood loss (NLL) for the error criterion. An Adam optimizer with a user
    defined learning rate. The learning rate reduces by the sched_gamma amount every sched_step epochs. The number of
    epochs used to train the NN are user defined as is whether the GPU is enabled and will be used. At each epoch, the 
    function tests the weights by using a validation set independent of the training data
    """
    # Calculates the length of the validation and training data for error calculations     
    valid_length = len(dataloaders['valid'])
    train_length = len(dataloaders['train'])
    # Sets the type of device the model is using (GPU ~ cuda vs not GPU - ~ cpu )
    device = torch.device("cuda" if ((torch.cuda.is_available()) and (gpu == True)) else "cpu")
    # Initializes the type of error measured 
    criterion = nn.NLLLoss()
    # Initializes the optimizer with type Adam and learning rate 
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    # Initializes the scheduler to reduce the learning every scheduled step by gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    # Sets the model to run on the device configured above
    model.to(device);
    # Cycles through the training and validation through the number epochs
    for epoch in range(epochs):
        # Sets the model to train
        model.train()
        # Steps the scheduler 
        scheduler.step()
        # Initializes the running training error for each epoch
        running_train = 0
        # Cycles through all of the batches of training data
        for train_inputs, train_labels in dataloaders['train']:
            # Converts the taining inputs and labels to the current device
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            # Zeroes the gradients
            optimizer.zero_grad()
            # Forward feeds the training data through the current model
            train_logps = model.forward(train_inputs)
            # Calculates the training loss from the error function
            train_loss = criterion(train_logps, train_labels)
            # Tallies the training losses through all of the batches
            running_train += train_loss.item()
            # Backpropogates the training errors in the NN
            train_loss.backward()
            # Steps the optimzer
            optimizer.step()
        # Initializes the validation variables to zero
        running_loss = 0
        accuracy = 0
        # Switches the model to evaluate for validation
        model.eval()
        # Ensures no gradient calculations during validation
        with torch.no_grad():
            # Cycles through all of the batches of validation data
            for valid_inputs, valid_labels in dataloaders['valid']:
                # Converts the validation inputs and labels to the current device
                valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)  
                # Forward feeds the validation data through the current model
                valid_logps = model.forward(valid_inputs)
                # Calculates the validation loss from the error function
                valid_loss = criterion(valid_logps, valid_labels)
                # Tallies the validation losses through all of the batches
                running_loss += valid_loss.item()
                # Converts the probabiliites from the logrithmic values
                valid_ps = torch.exp(valid_logps)
                # Calculates the first class and probability from topk
                top_prob, top_class = valid_ps.topk(1, dim=1)
                # Cummulates matches when the class matches
                matches = top_class == valid_labels.view(*top_class.shape)
                # Cummulates the accuracy from the average of the matches
                accuracy += torch.mean(matches.type(torch.FloatTensor)).item()  
                
        # Prints the Validation losses and accuracy at each training epoch if enabled
        if print_error == True:
            print("Epoch:{}/{}   ".format(epoch+1,epochs) + 
                  "Training loss:{:.3f}   ".format(running_train/train_length) +
                  "Validation loss:{:.3f}   ".format(running_loss/valid_length) + 
                  "Validation accuracy:{:.3f}".format(accuracy/valid_length))
    
    return model
    
def save_model(model, class_to_index, hidden, arch, checkpoint):
    """
    This function saves the model's state dictionary to a checkpoint file and also saves the 
    class to index dictionary to an additional checkpoint file. Both files can be later used for
    testing and prediction without the need to retrain the NN
    """
    # Creates a new checkpoint directory if it dooesn't exist
    if os.path.isdir(checkpoint) == False:
        os.mkdir(checkpoint) 
    # Converts to non-GPU so that the data can be stored
    model.to("cpu")
    # Saves the state dictionary and weights to the file checkpoint.pth
    torch.save(model.state_dict(), checkpoint + '/checkpoint.pth')
    # Saves the class to index map to the file class_to_idx.pth
    torch.save(class_to_index, checkpoint + '/class_to_idx.pth')
    # Saves the hidden count to the file hidden.pth
    torch.save(hidden, checkpoint + '/hidden.pth')    
    # Saves the vgg architecture type to the file vgg_ver.pth
    torch.save(arch, checkpoint + '/vgg_ver.pth')
    
def main():
    # Collect start time to measure program speed
    start_time = time()
    # Collect input arguments
    in_arg = train_input_args()
    # Load the image data into the dataloaders and the class to index dictionary
    data_loaders, class_to_idx = load_data(in_arg.data_dir)
    # Create the model and add the new classification layer
    untrained_model, hidden_count = build_model(in_arg.model, in_arg.hidden)
    # Train and validate the data from the data_loaders
    trained_model = train_model(untrained_model, data_loaders, in_arg.gpu, in_arg.epochs, in_arg.lr)
    # Save the trained model
    save_model(trained_model, class_to_idx, hidden_count, in_arg.model, in_arg.chk)
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
    