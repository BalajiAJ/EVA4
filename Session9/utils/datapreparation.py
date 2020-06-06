class album_Compose:
    def __init__(self, train=True):
        if train:
            self.albumentations_transform = Compose([
                                                    HorizontalFlip(p=0.5),
                                                    Rotate(limit=(-20, 20)),
                                                    VerticalFlip(p=0.5),
                                                    Cutout(
                                                        num_holes=3, max_h_size=8, max_w_size=8, p=0.5),
                                                    Normalize(
                                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                    ToTensor()
                                                    ])
        else:
            self.albumentations_transform = Compose([
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensor()
            ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img


def load(albumentations=True):
    # Define the transformations for both Test & Train, we may use Data Augmentation, hence better to keep 2 functions for test & train
    if albumentations:
        train_transform = album_Compose(train=True)
        test_transform = album_Compose(train=False)
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Get the Train and Test Set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4,
                           pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes, trainloader, testloader