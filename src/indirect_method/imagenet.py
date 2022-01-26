import torchvision
from .utils_dataset import return_dataloaders, get_dataset_with_index
from robustbench.model_zoo import model_dicts as all_models_rb
from robustbench.model_zoo.enums import ThreatModel, BenchmarkDataset
from torch.utils.data import Dataset
from robustbench.data import load_imagenet

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def get_dataloaders(opt, mode):
    assert(mode!='train')
    assert(mode!='test')
    
    imagenet_preprocessing_str = all_models_rb[BenchmarkDataset(opt.dataset_to_use)][ThreatModel(opt.use_robust_bench_threat)][opt.use_robust_bench_model]['preprocessing']
    
    im_data = load_imagenet(
    data_dir = opt.imagenet_location, prepr = imagenet_preprocessing_str)

    in_dataset = get_dataset_with_index(SimpleDataset((im_data[0]-0.5)/0.5, im_data[1]), 0)
    
    return return_dataloaders(lambda: in_dataset, opt, split = mode)