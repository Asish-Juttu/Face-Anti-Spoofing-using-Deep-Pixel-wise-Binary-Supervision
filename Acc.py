import os
import torch
from torchvision import transforms, datasets
from trainer.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from models.loss import PixWiseBCELoss
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device
from utils.eval import predict, calc_acc, add_images_tb

cfg = read_cfg(cfg_file='config/densenet_161_adam_lr1e-3.yaml')

device = get_device(cfg)

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

loss = PixWiseBCELoss(beta=cfg['train']['loss']['beta'])

writer = SummaryWriter(cfg['log_dir'])

test_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])
testset = PixWiseDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['test_set'],
    map_size=cfg['model']['map_size'],
    transform=test_transform,
    smoothing=cfg['model']['smoothing']
)
testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=cfg['test']['batch_size'],
    shuffle=True,
    num_workers=0
)


saved_name = os.path.join(cfg['output_dir'], '{}_{}.pth'.format(cfg['model']['base'], cfg['dataset']['name']))
state = torch.load(saved_name)

optimizer.load_state_dict(state['optimizer'])
network.load_state_dict(state['state_dict'])

network.eval()
val = 0
avg = 0
sum = 0
count = 0
for i, (img, mask, label) in enumerate(testloader):
    img, mask, label = img.to(device), mask.to(device), label.to(device)
    net_mask, net_label = network(img)
    #loss = loss(net_mask, net_label, mask, label)

    # Calculate predictions
    preds, score = predict(net_mask, net_label, score_type=cfg['test']['score_type'])
    targets, _ = predict(mask, label, score_type=cfg['test']['score_type'])
    acc = calc_acc(preds, targets)
    # Update metrics
    val = acc
    sum += val
    count += 1
    avg = sum / count if count != 0 else 0
    print('Iter: {}, Acc: {}'.format(i,avg))

print('Testing Accuracy is : {}'.format(avg))