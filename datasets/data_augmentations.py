from torchvision import transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_transform(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((args.resize, args.resize), scale=(0.8, 1.0)),  # Crop with a conservative scale to retain lesions
            transforms.RandomHorizontalFlip(),  # Horizontal flip
            transforms.RandomVerticalFlip(),  # Vertical flip
            transforms.RandomRotation(degrees=15),  # Rotate by a small degree
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Slight color adjustments
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),  # Gentle Gaussian Blur
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.resize,args.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )



    return train_transform, test_transform 


# train_transform = transforms.Compose(
#     [
#         transforms.RandomResizedCrop((resize,resize), scale=(0.2, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ]
# )



# train_transform = transforms.Compose(
#     [
#         transforms.RandomResizedCrop((resize, resize), scale=(0.8, 1.0)),  # Crop with a conservative scale to retain lesions
#         transforms.RandomHorizontalFlip(),  # Horizontal flip
#         transforms.RandomVerticalFlip(),  # Vertical flip
#         transforms.RandomRotation(degrees=15),  # Rotate by a small degree
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Slight color adjustments
#         transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),  # Gentle Gaussian Blur
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ]
# )


# test_transform = transforms.Compose(
#     [
#         transforms.Resize((resize,resize)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ]
# )