import open3d as o3d
from datasets import get_datasets
from args import get_args
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn


def main(args):
    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    _, te_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    ds_mean = torch.from_numpy(te_dataset.all_points_mean).cuda()
    ds_std = torch.from_numpy(te_dataset.all_points_std).cuda()

    all_sample = []
    with torch.no_grad():
        for i in range(0, args.num_sample_shapes, args.batch_size):
            B = len(range(i, min(i + args.batch_size, args.num_sample_shapes)))
            N = args.num_sample_points
            _, out_pc = model.sample(B, N)
            out_pc = out_pc * ds_std + ds_mean
            all_sample.append(out_pc)

    sample_pcs = torch.cat(all_sample, dim=0).cpu().detach().numpy()
    print("Generation sample size:(%s, %s, %s)" % sample_pcs.shape)

    # Save the generative output
    os.makedirs("demo", exist_ok=True)
    np.save(os.path.join("demo", "model_out_smp.npy"), sample_pcs)

    # Visualize the demo
    pcl = o3d.geometry.PointCloud()
    for i in range(int(sample_pcs.shape[0])):
        print("Visualizing: %03d/%03d" % (i, sample_pcs.shape[0]))
        pts = sample_pcs[i].reshape(-1, 3)
        pcl.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw_geometries([pcl])


if __name__ == '__main__':
    args = get_args()
    main(args)
