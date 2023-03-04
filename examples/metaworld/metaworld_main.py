import os
import time
import argparse

import numpy as np
import torchvision
import torch
import torch.nn as nn

import plotly.graph_objects as go
import pickle as pkl

import gym
# import d4rl  # Import required to register environments, you may need to also import the submodule

from util import AverageMeter, AugDataset
from encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss


def parse_option():
    parser = argparse.ArgumentParser('Representation Learning with CPC Losses')

    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=20, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=50, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = os.path.join(
        opt.result_folder,
        f"align{opt.align_w:g}alpha{opt.align_alpha:g}_unif{opt.unif_w:g}t{opt.unif_t:g}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def get_data_loader(opt, gamma=0.9):
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.RandomResizedCrop(48, scale=(0.8, 1)),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    # dataset = TwoAugUnsupervisedDataset(
    #     torchvision.datasets.STL10(opt.data_folder, 'train+unlabeled', download=True), transform=transform)
    # dataset = TwoAugUnsupervisedDataset(
    #     torchvision.datasets.CIFAR10(opt.data_folder, train=False, download=True), transform=transform)
    dataset_path = os.path.abspath("data/metaworld_door_open_v2_mixed_img.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)
    print("Load dataset from: {}".format(dataset_path))
    print("Number of transitions in the dataset: {}".format(sum([len(traj) for traj in dataset])))

    relabeled_dataset = []
    for traj in dataset:
        for t in range(len(traj) - 1):
            obs, a = traj[t]
            next_s, next_a = traj[t + 1]

            w_sum = (1 - gamma ** (len(traj) - t - 1)) / (1 - gamma)
            w = gamma ** (np.arange(t + 1, len(traj)) - t - 1) / w_sum

            future_idxs = np.arange(len(traj[t + 1:])) + 1  # check this
            future_idx = np.random.choice(future_idxs, p=w)
            g, _ = traj[t + future_idx]

            # s = np.random.normal(loc=s, scale=0.2)
            # g = np.random.normal(loc=s, scale=0.2)
            # (B, C, H, W)
            # obs = obs.reshape([48, 48, 3]).transpose(2, 0, 1)
            # g = g.reshape([48, 48, 3]).transpose(2, 0, 1)
            # obs = obs.reshape([48, 48, 3]).transpose(2, 0, 1)
            # g = g.reshape([48, 48, 3]).transpose(2, 0, 1)

            obs = obs.reshape([48, 48, 3])
            g = g.reshape([48, 48, 3])

            # relabeled_dataset.append((obs, a, g, next_s, next_a))
            relabeled_dataset.append((obs, g))
    relabeled_dataset = np.array(relabeled_dataset)
    # dataset_dir = os.path.abspath("./data")
    # dataset_path = os.path.join(dataset_dir, "metaworld_door_open_v2_mixed_img_relabeled.pkl")
    # os.makedirs(dataset_dir, exist_ok=True)
    # with open(dataset_path, "wb+") as f:
    #     pkl.dump(relabeled_dataset, f)
    # print("Dataset saved to: {}".format(dataset_path))
    # exit()

    # dataset_path = os.path.abspath("data/metaworld_door_open_v2_mixed_img_relabeled.pkl")
    # with open(dataset_path, "rb") as f:
    #     relabeled_dataset = pkl.load(f)
    # relabeled_dataset = torch.Tensor(relabeled_dataset)
    print("Number of transitions in the relabeled dataset: {}".format(len(relabeled_dataset)))

    dataset = AugDataset(relabeled_dataset, transform)

    # env = gym.make('kitchen-complete-v0')
    # dataset = env.get_dataset()
    # s = env.reset()
    #
    # relabeled_s = []
    # relabeled_g = []
    # for traj in list(d4rl.sequence_dataset(env))[:100]:
    #     for t in range(len(traj['observations']) - 1):
    #         s, a = traj['observations'][t], traj['actions'][t]
    #         next_s, next_a = traj['observations'][t + 1], traj['actions'][t + 1]
    #         # next_s, next_a = traj[t + 1]
    #
    #         w_sum = (1 - gamma ** (len(traj['observations']) - t - 1)) / (1 - gamma)
    #         w = gamma ** (np.arange(t + 1, len(traj['observations'])) - t - 1) / w_sum
    #
    #         future_idxs = np.arange(len(traj['observations'][t + 1:])) + 1  # check this
    #         future_idx = np.random.choice(future_idxs, p=w)
    #         g = traj['observations'][t + future_idx]
    #
    #         # g = np.random.normal(loc=s, scale=0.2)
    #
    #         # relabeled_dataset.append((s, a, g, next_s, next_a))
    #         relabeled_s.append(s)
    #         relabeled_g.append(g)
    # relabeled_s = torch.Tensor(relabeled_s)
    # relabeled_g = torch.Tensor(relabeled_g)
    # print("Number of transitions in the relabeled dataset: {}".format(len(relabeled_s)))
    #
    # dataset = torch.utils.data.TensorDataset(relabeled_s, relabeled_g)

    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)


def visualize(opt, encoder, dataloader):
    # data for sphere
    r = 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    # phi = phi.reshape([-1, repr_dim])
    # sa_repr = phi.apply(phi_params, s, a)
    # s_repr = phi.apply(phi_params, s)
    # dataloader.
    reprs = []
    for transition in dataloader:
        s = transition[:, 0]
        g = transition[:, 1]

        repr = encoder(torch.cat([s.to(opt.gpus[0]), g.to(opt.gpus[0])]))

        reprs.append(repr.cpu().detach().numpy())
    reprs = np.concatenate(reprs)

    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z,
                   colorscale=[[0, 'cornflowerblue'], [1, 'cornflowerblue']],
                   opacity=0.1,
                   showscale=False),
        go.Scatter3d(x=reprs[:, 0], y=reprs[:, 1], z=reprs[:, 2],
                     name=r'$repr$',
                     mode='markers',
                     marker={'size': 8,
                             'color': 'limegreen'},
                     showlegend=True),
    ])

    return fig


def main():
    opt = parse_option()

    # print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = nn.DataParallel(SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0]), opt.gpus)
    # encoder = SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0])

    optim = torch.optim.SGD(encoder.parameters(), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
    #                                                  milestones=opt.lr_decay_epochs)
    # optim = torch.optim.Adam(encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    loader = get_data_loader(opt)
    cpc_loss = nn.CrossEntropyLoss(reduction='none')

    # DEBUG
    # fig = visualize(opt, encoder, loader)
    # fig_path = "figures/d4rl_3d_repr_vis.html"
    # fig.write_html(fig_path, include_mathjax='cdn')
    # print("Figure save to: {}".format(fig_path))
    # exit()

    align_meter = AverageMeter('align_loss')
    unif_meter = AverageMeter('uniform_loss')
    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        align_meter.reset()
        unif_meter.reset()
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, transition in enumerate(loader):
            s = transition[:, 0]
            g = transition[:, 1]

            optim.zero_grad()
            s_repr, g_repr = encoder(torch.cat([s.to(opt.gpus[0]), g.to(opt.gpus[0])])).chunk(2)
            # align_loss_val = align_loss(s_repr, g_repr, alpha=opt.align_alpha)
            # unif_loss_val = (uniform_loss(s_repr, t=opt.unif_t) + uniform_loss(g_repr, t=opt.unif_t)) / 2
            # loss = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w
            # align_meter.update(align_loss_val, x.shape[0])
            # unif_meter.update(unif_loss_val)

            logits = s_repr @ g_repr.T
            labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
            loss = torch.mean(cpc_loss(logits, labels))

            loss_meter.update(loss, s_repr.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                # print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                #       f"{align_meter}\t{unif_meter}\t{loss_meter}\t{it_time_meter}")
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"\t{loss_meter}\t{it_time_meter}")
            t0 = time.time()
        # scheduler.step()
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.module.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')

    fig = visualize(opt, encoder, loader)
    fig_path = "figures/metaworld_door_open_v2_img_repr_vis.html"
    fig.write_html(fig_path, include_mathjax='cdn')
    print("Figure save to: {}".format(fig_path))


if __name__ == '__main__':
    main()
