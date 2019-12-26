"""prints currently alive Tensors and Variables"""
import sys
import gc
import torch

def dump_tensors(f=sys.stdout):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print((type(obj), obj.size()), file=f)
        except:
            pass
