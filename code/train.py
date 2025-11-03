import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"
import torch, math, time, argparse, os
import random, dataset, pa_utils, losses, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb
import timm

parser = argparse.ArgumentParser(description=
    'DMA'  
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
    default='../logs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--nb_c', default = None, type = int,
    dest = 'nb_c',
    help = 'number of proxies'
)
parser.add_argument('--tau', default = -1, type = float,
    dest = 'tau',
    help = 'regularization term'
)
parser.add_argument('--batch-size', default = 120, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 60, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'resnet50',
    help = 'Model for training'
)
parser.add_argument('--loss', default = 'dma',
    help = 'Criterion for training'
)
parser.add_argument('--optimizer', default = 'adamw',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 1e-4, type =float,
    help = 'Learning rate setting'
)
parser.add_argument('--weight-decay', default = 1e-4, type =float,
    help = 'Weight decay setting'
)
parser.add_argument('--lr-decay-step', default = 10, type =int,
    help = 'Learning decay step setting'
)
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
    help = 'Learning decay gamma setting'
)
parser.add_argument('--alpha', default = 32, type = float,
    help = 'Scaling Parameter setting'
)
parser.add_argument('--mrg', default = 0.1, type = float,
    help = 'Margin parameter setting'
)
parser.add_argument('--IPC', type = int,
    help = 'Balanced sampling, images per class'
)
parser.add_argument('--warm', default = 5, type = int,
    help = 'Warmup training epochs'
)
parser.add_argument('--bn-freeze', default = 1, type = int,
    help = 'Batch normalization parameter freeze'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)
parser.add_argument('--seed', default = 1, type = int,
    help = 'seed'
)

args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # set random seed for all gpus


if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model, args.loss, args.sz_embedding, args.alpha, 
                                                                                            args.mrg, args.optimizer, args.lr, args.sz_batch, args.remark)
# Wandb Initialization
wandb.init(project=args.dataset + '_DMA', notes=LOG_DIR)
wandb.config.update(args)

os.chdir('./data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    trn_dataset = dataset.load(
            name = args.dataset,
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = (args.model == 'bn_inception')
            ))
else:
    trn_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True, 
                is_inception = (args.model == 'bn_inception')
            ))

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )
    print('Balanced Sampling')
    
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
            name = args.dataset,
            root = data_root,
            mode = 'eval',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
            ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )
    
else:
    query_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
    ))
    
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

    gallery_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'gallery',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
    ))
    
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

nb_classes = trn_dataset.nb_classes()

# Backbone Model
model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

# DML Losses
criterion = losses.dma(nb_classes = nb_classes, sz_embed = args.sz_embedding, nb_c= args.nb_c, tau=args.tau, mrg = args.mrg, alpha = args.alpha).cuda()
 
    
# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
    {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args.lr) * 1},
]
# param_groups = [{'params': model.parameters(), 'lr':float(args.lr) * 1}]

param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr) * 100})

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

recall_total = []
nmi_total = []
for epoch in range(0, args.nb_epochs):
    model.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
            # unfreeze_model_param =  list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())
        if epoch == 0:

            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:                         
        m = model(x.squeeze().cuda())
        loss = criterion(m, y.squeeze().cuda())
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
         
    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()
    
    if(epoch >= 0):
        with torch.no_grad():
            print("**Evaluating...**")
            if args.dataset == 'Inshop':
                Recalls = pa_utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
                recall_total.append(Recalls)  
                np_recall_total = np.array(recall_total)*100
                max_index = np.argmax(np_recall_total[:, 0])
                max_row = np_recall_total[max_index]
                print('################## Best Results ##################')
                print("R@1: {:.3f}".format(max_row[0]), "R@10: {:.3f}".format(max_row[1]), "R@20: {:.3f}".format(max_row[2]), "R@40: {:.3f}".format(max_row[3]))
            elif args.dataset != 'SOP':
                Recalls, nmi = pa_utils.evaluate_cos(model, dl_ev)
                recall_total.append(Recalls)  
                nmi_total.append(nmi)
                np_recall_total = np.array(recall_total)*100
                max_index = np.argmax(np_recall_total[:, 0])
                max_row = np_recall_total[max_index]
                print('####################### Best Results ######################')
                print("R@1: {:.3f}".format(max_row[0]), "R@2: {:.3f}".format(max_row[1]), "R@4: {:.3f}".format(max_row[2]), "R@8: {:.3f}".format(max_row[3]), "NMI: {:.3f}".format(np.max(np.array(nmi_total)*100)))
            else:
                Recalls, nmi = pa_utils.evaluate_cos_SOP(model, dl_ev)
                recall_total.append(Recalls)  
                nmi_total.append(nmi)
                np_recall_total = np.array(recall_total)*100
                max_index = np.argmax(np_recall_total[:, 0])
                max_row = np_recall_total[max_index]
                print('########################## Best Results #########################')
                print("R@1: {:.3f}".format(max_row[0]), "R@10: {:.3f}".format(max_row[1]), "R@100: {:.3f}".format(max_row[2]), "R@1000: {:.3f}".format(max_row[3]), "NMI: {:.3f}".format(np.max(np.array(nmi_total)*100)))

        # Logging Evaluation Score
        if args.dataset == 'Inshop':
            # for i, K in enumerate([1,10,20,30,40,50]):    
            for i, K in enumerate([1,10,20,40]):    
                wandb.log({"R@{}".format(K): Recalls[i]}, step=epoch)
        elif args.dataset != 'SOP':
            for i in range(6):
                wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)
        else:
            for i in range(4):
                wandb.log({"R@{}".format(10**i): Recalls[i]}, step=epoch)
        
