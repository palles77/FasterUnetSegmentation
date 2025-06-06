import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

def print_segmentation_parameters(
    segmentation_parameters: dict, 
    indent: int = 0, 
    header_info: str = "Segmentation parameters"):
    """
    Print the segmentation parameters dictionary and any nested dictionaries.

    Args:
        segmentation_parameters (dict): A dictionary of segmentation parameters.
        indent (int): The number of spaces to indent each level of nested dictionaries.
        header_info (str): The header text to print before the dictionary.
    """
    # print header
    print(header_info + ":")
    
    # print dictionary
    for key, value in segmentation_parameters.items():        
        
        if isinstance(value, dict):
            print_segmentation_parameters(value, indent+2, "")
        else:
            print(" " * indent + f"{key}: {value}")