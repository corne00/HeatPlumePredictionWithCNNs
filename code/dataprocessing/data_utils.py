import numpy as np
import torch


class NormalizeTransform:
    def __init__(self,info:dict,out_range = (0,1)):
        self.info = info
        self.out_min, self.out_max = out_range 

    def __call__(self,data, type = "Inputs"):
        for prop, stats in self.info[type].items():
            index = stats["index"]
            if index < data.shape[0]:
                self.__apply_norm(data,index,stats)
            else:
                print(f"Index {index} might be in training data but not in this dataset")
        return data
    
    def reverse(self,data,type = "Labels"):
        # output for out of place operation, required for training
        data = torch.swapaxes(data,0,1)
        output = torch.zeros_like(data)
        for _, stats in self.info[type].items():
            index = stats["index"]
            output[index] = self.__reverse_norm(data,index,stats)
        return output
    
    def __apply_norm(self,data,index,stats):
        norm = stats["norm"]
        
        def rescale():
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - stats["min"]) / delta * (self.out_max - self.out_min) + self.out_min
        
        if norm == "LogRescale":
            data[index] = np.log(data[index] - stats["min"] + 1)
            rescale()
        elif norm == "Rescale":
            rescale()
        elif norm == "Standardize":
            data[index] = (data[index] - stats["mean"]) / stats["std"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['norm']}' not recognized")
        
    def __reverse_norm(self,data,index,stats):
        # if len(data.shape) == 4: # TODO not relevant for allin1 
        #     assert data.shape[0] <= data.shape[1], "Properties must be in 0th dimension; batches pushed to 1st dimension"
        norm = stats["norm"]

        def rescale(data, index, stats, out_min, out_max):
            delta = stats["max"] - stats["min"]
            return (data[index] - out_min) / (out_max - out_min) * delta + stats["min"]
        
        if norm == "LogRescale":
            return np.exp(rescale(data, index, stats, self.out_min, self.out_max)) + stats["min"] - 1
        elif norm == "Rescale":
            return rescale(data, index, stats, self.out_min, self.out_max)
        elif norm == "Standardize":
            return data[index] * stats["std"] + stats["mean"]
        elif norm is None:
            return data[index]
        else:
            raise ValueError(f"Normalization type '{stats['norm']}' not recognized")
