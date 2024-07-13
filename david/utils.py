import os
import pickle
import torch

def cache_function_call(file_path, func, use_torch=False, *args, **kwargs):
    # Check if the results file exists
    if os.path.exists(file_path):
        # Load the results from the file
        if use_torch:
            results = torch.load(file_path, map_location=torch.device('cpu'))
            print(f"Results loaded from disk using torch.load: {file_path}")
        else:
            with open(file_path, 'rb') as file:
                results = pickle.load(file)
            print(f"Results loaded from disk using pickle: {file_path}")
    else:
        # Run the function to get the results
        results = func(*args, **kwargs)
        
        # Move results to CPU if it's a torch tensor
        if use_torch and isinstance(results, torch.Tensor):
            results = results.cpu()
        
        # Save the results to the file
        if use_torch:
            torch.save(results, file_path)
            print(f"Results computed and saved to disk using torch.save: {file_path}")
        else:
            with open(file_path, 'wb') as file:
                pickle.dump(results, file)
            print(f"Results computed and saved to disk using pickle: {file_path}")
        
    return results