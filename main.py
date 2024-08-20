"""Functions for training and running EF prediction."""

import math
import os
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.echo import Echo
import utils as utils
from config import dataset_config
from model.convlstm import ConvLSTM


def get_parser():
    parser = argparse.ArgumentParser(description="Parser for training video models")

    parser.add_argument("--exp_name", type=str, default="default", help="Task type")
    parser.add_argument("--output", type=str, required=False, default=None, help="Path to the output directory (must be a directory)")
    parser.add_argument("--task", type=str, default="EF", help="Task type")
    parser.add_argument("--training_data", type=str, default="pediatric", help="The name of trained dataset. [pediatric, dynamic]")
    
    parser.add_argument("--model_name", type=str, default="r2plus1_18", help="Name of the model") # mvit_v2_s, r2plus1_18
    
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--random", dest="pretrained", action="store_false", help="Use randomly initialized model")
    parser.set_defaults(pretrained=True)

    parser.add_argument("--weights", type=str, required=False, default=None, help="Path to the weights file (must exist and be a file)")
    parser.add_argument("--run_test", dest="run_test", action="store_true", help="Run test after training")
    parser.add_argument("--skip_test", dest="run_test", action="store_false", help="Skip test after training")
    parser.set_defaults(run_test=True)

    parser.add_argument("--num_epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr_step_period", type=int, default=15, help="Period of learning rate step decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cpu', 'cuda')")
    parser.add_argument("--seed", type=int, default=113, help="Random seed")

    return parser


def main(args):
    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """
    # incorperate attribute
    for key, value in dataset_config.items():
        setattr(args, key, value)

    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set default output directory
    if args.output is None:
        args.output = os.path.join("output", "video", "{}_{}_{}_{}_{}_{}".format(args.training_data, args.model_name, args.frames, args.period, "pretrained" if args.pretrained else "random", args.num_epochs))
    os.makedirs(args.output, exist_ok=True)

    # save params into output directory
    args_file = os.path.join(args.output, "args.txt")
    with open(args_file, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.output)

    # Set device for computations
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    if args.model_name in ['mvit_v2_s', 'r2plus1_18']:
        model = torchvision.models.video.__dict__[args.model_name](pretrained=args.pretrained)
        # for r2plus1d
        if args.model_name == 'r2plus1_18':
            model.fc = torch.nn.Linear(model.fc.in_features, 1) 
            model.fc.bias.data[0] = 55.6
        elif args.model_name == 'mvit_v2_s':
            # for mvit_v2_s
            model.head[1] = torch.nn.Linear(model.head[1].in_features, 1) 
            model.head[1].bias.data[0] = 55.6
    elif args.model_name in ['convlstm']:
        hidden_dims = [3, 45, 64, 128, 256, 512]
        kernel_size = (3, 3)
        model = ConvLSTM(input_dim=3,
                            hidden_dims=hidden_dims,
                            kernel_size=kernel_size,
                            batch_first=True)
        model.fc.bias.data[0] = 55.6
    # print(model)


    if args.device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(args.device)

    if args.weights is not None:
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    if 'mvit' in args.model_name:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs // 2)
    elif 'r2plus1' in args.model_name:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.lr_step_period is None:
            args.lr_step_period = math.inf
        scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_step_period)
    elif 'convlstm' in args.model_name:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_step_period)

    

    args.data_dir = getattr(args, args.training_data + '_dir')
    # Compute mean and std
    mean, std = utils.get_mean_and_std(Echo(root=args.data_dir, split="train"))
    print(mean, std)
    # [32.618484 32.754513 33.059017] [49.997513 50.009624 50.257397] for dynamics
    # [26.16616  24.224289 27.648663] [44.82573  41.950523 46.429   ] for pediatric
    if 'mvit' in args.model_name:
        kwargs = {"target_type": args.task,
                "mean": mean,
                "std": std,
                "length": args.frames,
                "period": args.period,
                "resize": (224,)
                }
    else:
        kwargs = {"target_type": args.task,
                "mean": mean,
                "std": std,
                "length": args.frames,
                "period": args.period,
                }
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = Echo(root=args.data_dir, split="train", **kwargs, pad=12)
    if args.num_train_patients is not None and len(dataset["train"]) > args.num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), args.num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["test"] = Echo(root='/home/jliu288/data/echocardiogram/pediatric_echo_avi/A4C', split="test", **kwargs)

    # Run training and testing loops
    with open(os.path.join(args.output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(args.output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total number of parameters: {total_params}\n")
        for epoch in range(epoch_resume, args.num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'test']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device.type == "cuda"), drop_last=(phase == "train"))

                loss, yhat, y = run_epoch(model, dataloader, phase == "train", optim, args.device, args=args)

                r2 = sklearn.metrics.r2_score(y, yhat)
                mae = sklearn.metrics.mean_absolute_error(y, yhat)
                rmse = math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))

                # Log metrics to TensorBoard
                writer.add_scalar(f'{phase}/loss', loss, epoch)
                writer.add_scalar(f'{phase}/r2_score', r2, epoch)
                writer.add_scalar(f'{phase}/mae', mae, epoch)
                writer.add_scalar(f'{phase}/rmse', rmse, epoch)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

                f.write("epoch:{}, phase: {}, loss: {},r2_score: {}, MAE: {}, RMSE: {}, time: {}, y_size: {}, BATCH_SIZE: {}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              sklearn.metrics.mean_absolute_error(y, yhat),
                                                              sklearn.metrics.mean_squared_error(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              args.batch_size))
                f.write("(one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(*utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("(one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(*utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("(one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(*tuple(map(math.sqrt, utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': args.period,
                'frames': args.frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(args.output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(args.output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if args.num_epochs != 0:
            checkpoint = torch.load(os.path.join(args.output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if args.run_test:
            for split in ["test"]:
                # Performance without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    Echo(root=args.data_dir, split=split, **kwargs),
                    batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device.type == "cuda"))
                loss, yhat, y = run_epoch(model, dataloader, False, None, args.device, args=args)

                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Performance with test-time augmentation
                ds = Echo(root=args.data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=(args.device.type == "cuda"))
                loss, yhat, y = run_epoch(model, dataloader, False, None, args.device, save_all=True, block_size=args.batch_size, args=args)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Write full performance to file
                with open(os.path.join(args.output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output, "{}_roc.pdf".format(split)))
                plt.close(fig)


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None, args=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    model.train(train)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                if 'convlstm' in args.model_name:
                    X = X.transpose(1, 2)

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                ## loss average, current loss, variance
                pbar.set_postfix_str("loss ave: {:.2f} (loss: {:.2f}), variance: {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)