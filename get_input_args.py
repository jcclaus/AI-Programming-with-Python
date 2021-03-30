# PROGRAMMER: John Claus
# DATE CREATED: 12/20/2020                           
# REVISED DATE: 
# PURPOSE: Create 2 functions that retrieve the  command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. These functions are specific to the train.py 
#          and predict.py programs.
#
##
# Imports python modules
import argparse

def train_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the train.py program from a terminal window. This function uses Python's 
    argparse module to created and defined these 7 command line arguments. If 
    the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Data directory as dat_dir with default value of 'flower'
      2. Learning rate as --lr with default value 0.001
      3. TorchVision VGG model used as --model with default value 'vgg16'
      4. Number of hidden units as --hidden with default value 4096
      5. Number of training epochs as --epochs with default value 8
      6. GPU Enabled as --gpu with default value True
      7. Checkpoint folder name as --chk with default value 'checkpoint'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type = str, help = 'Path to the data directory') 
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate (0.001)') 
    parser.add_argument('--model', type = str, default = 'vgg16', help = 'TorchVision VGG model used: vgg19 or (vgg16)') 
    parser.add_argument('--hidden', type = int, default = 4096, help = 'Number of hidden units (4096)') 
    parser.add_argument('--epochs', type = int, default = 8, help = 'Number of training epochs (8)') 
    parser.add_argument('--gpu', type = bool, default = True, help = '(True) = GPU enabled, False = GPU disabled') 
    parser.add_argument('--chk', type = str, default = 'checkpoint', help = 'Checkpoint folder name') 
    return parser.parse_args()

def predict_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the predict.py program from a terminal window. This function uses Python's 
    argparse module to created and defined these 5 command line arguments. If 
    the user fails to provide some or all of the 5 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:    
      1. Path to the folder and file name of image to be checked as image with default value 'orange-dahlia.png'
      2. Path to the folder of ML checkpoint used as checkpoint with default value 'checkpoint'
      3. GPU Enabled as --gpu with default value False
      4. Top number of class predicitions as --topk with default value 5
      5. Path to the folder and file name of JSON map used as --json with default value 'cat_to_name.json'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type = str, help = 'Path to the folder and file name of image to be checked') 
    parser.add_argument('checkpoint', type = str, help = 'Checkpoint folder name') 
    parser.add_argument('--gpu', type = bool, default = False, help = 'True = GPU enabled, False = GPU disabled') 
    parser.add_argument('--topk', type = int, default = 5, help = 'Top number of class predicitions') 
    parser.add_argument('--json', type = str, default = 'cat_to_name.json', help = 'Path to the folder and file name of JSON map used') 
    return parser.parse_args()