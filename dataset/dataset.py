from torch.utils.data import Dataset
from utils import util
import glob
from PIL import Image

class UTKFace(Dataset):
    """
    UTKFace dataset class
    """
    def __init__(self, data_dir=None, download=True, transform=None, url=None):
        # Set Inputs and Labels
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []

        print(f'data_dir = {data_dir}')
        if download:
            util.download_utkfaces_dataset(url=url, destination=self.data_dir, remove_source=True)

        for path in sorted(glob.glob(f"{self.data_dir}*.jpg.chip.jpg")):
            filename = path.split("\\")[-1].split("_")
            if len(filename)==4:
                self.images.append(path)
                self.ages.append(util.categorize_age(int(filename[0])))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))

    def __len__(self):
         return len(self.images)

    def __getitem__(self, index):
        # Load an Image
        img = Image.open(self.images[index]).convert('RGB')        
        # Transform it
        img = self.transform(img)

        return img, {'age': self.ages[index], 'gender':self.genders[index], 'race':self.races[index]}