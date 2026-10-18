from PIL import Image

from src.data_loaders.lego_dataset import LegoDataset


dataset = LegoDataset()

print(f"Total images loaded: {len(dataset)}")

img, pose = dataset[0]

img_uint8 = (img * 255).byte().cpu().numpy()
image_from_tensor = Image.fromarray(img_uint8)
image_from_tensor.show()
