# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.

from timeit import default_timer as timer

try:
    import cupy as cp
    _GPU_AVAILABLE = cp.cuda.is_available()
except ModuleNotFoundError:
    print('CuPy not available, running on CPU.')
    _GPU_AVAILABLE = False

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .utils import find_nearest_neighbors


class Diffuser(object):

    def __init__(self,
                 num_pt_neighbors=20,
                 distance_mu=0.05,
                 normals_mu=0.2,
                 px_to_pt_weight=1e-4):
        self.num_pt_neighbors = num_pt_neighbors
        self.distance_mu = distance_mu
        self.normals_mu = normals_mu
        self.px_to_pt_weight = px_to_pt_weight

    def build_subgraph_2d3d(self, points, frame):
        num_points = points.shape[0]
        num_pixels = frame.rgb_height * frame.rgb_width

        pixel_indices_matrix = np.arange(
            num_pixels, dtype=int).reshape((frame.rgb_height, frame.rgb_width))

        projected_points, visible = frame.project(points)

        pp_rows = np.arange(num_points, dtype=int)[visible]
        pp_cols = pixel_indices_matrix[projected_points[visible, 1],
                                       projected_points[visible, 0]]
        pp_d = np.full((visible[visible].shape[0]), self.px_to_pt_weight)

        return coo_matrix((pp_d, (pp_rows, pp_cols)),
                          shape=(num_points, num_pixels))

    def build_subgraph_3d3d(self, points, normals):
        num_points = points.shape[0]
        # Create KNN graph of 3D points
        distances, neighbors = find_nearest_neighbors(points,
                                                      self.num_pt_neighbors)

        # Weight based on distances
        w_d = np.exp(-(distances**2) / (2 * self.distance_mu**2)).ravel()

        # Weight based on surface normals
        diff_n = normals[neighbors, :] - normals[:, np.newaxis, :]
        diff_sq = diff_n[:, :, 0]**2 + diff_n[:, :, 1]**2 + diff_n[:, :, 2]**2
        w_s = np.exp(-diff_sq / (2 * self.normals_mu**2)).ravel()

        # COO matrix initialization
        weights = w_d * w_s
        row_indices = np.indices(distances.shape)[0].ravel()
        col_indices = neighbors.ravel()

        return coo_matrix((weights, (row_indices, col_indices)),
                          shape=(num_points, num_points))

    def run(self,
            points,
            normals,
            frames,
            num_classes,
            max_iters=200,
            device=0):
        if not _GPU_AVAILABLE or device == 'cpu':
            return self.run_cpu(
                points, normals, frames, num_classes, max_iters=max_iters)
        else:
            return self.run_gpu(
                points,
                normals,
                frames,
                num_classes,
                max_iters=max_iters,
                device=device)

    def run_gpu(self,
                points,
                normals,
                frames,
                num_classes,
                max_iters=200,
                device=0):

        with cp.cuda.Device(device):
            num_points = points.shape[0]
            num_classes += 1
            Z_gpu = cp.zeros((num_points, num_classes))

            for frame, idx in frames:

                # Build matrix Zi
                num_pixels = frame.rgb_height * frame.rgb_width
                pixel_labels = cp.array(frame.labels.ravel()).astype(int)
                pixel_ids = cp.arange(num_pixels).astype(int)
                Zi_gpu = cp.zeros((num_pixels, num_classes))
                Zi_gpu[pixel_ids, pixel_labels] = 1

                # Build subgraph Gi
                Gi_gpu = cp.sparse.csr_matrix(
                    self.build_subgraph_2d3d(points, frame))

                # Accumulate
                Z_gpu += Gi_gpu.dot(Zi_gpu)

                print("Processed frame %d" % idx)

            # Build 3D-3D graph on GPU
            G_gpu = cp.sparse.csr_matrix(
                self.build_subgraph_3d3d(points, normals))

            # Perform row normalization
            row_sums = G_gpu.sum(axis=1).ravel() + Z_gpu.sum(axis=1).ravel()
            Z_gpu /= row_sums[:, cp.newaxis]
            repeats = cp.diff(G_gpu.indptr)
            G_gpu.data /= cp.array(row_sums.get().repeat(repeats.get()))

            # Move initial zero labels to GPU
            Y_gpu = cp.zeros(Z_gpu.shape)

            print("Starting label diffusion")
            start = timer()

            for i in range(max_iters):
                Y_new = G_gpu.dot(Y_gpu) + Z_gpu
                Y_gpu = Y_new

            end = timer()

            print("Done!")
            print("Label diffusion took %f s" %(end - start))

            label_likelihoods = cp.asnumpy(Y_gpu)
            labels = cp.asnumpy(cp.argmax(Y_gpu, axis=1))

            return labels, label_likelihoods

    def run_cpu(self, points, normals, frames, num_classes, max_iters=200):
        num_points = points.shape[0]
        num_classes += 1
        Z_cpu = np.zeros((num_points, num_classes))

        for frame, idx in frames:

            # Build matrix Zi
            num_pixels = frame.rgb_height * frame.rgb_width
            pixel_labels = np.array(frame.labels.ravel()).astype(int)
            pixel_ids = np.arange(num_pixels).astype(int)
            Zi_cpu = np.zeros((num_pixels, num_classes))
            Zi_cpu[pixel_ids, pixel_labels] = 1

            # Build subgraph Gi
            Gi_cpu = csr_matrix(self.build_subgraph_2d3d(points, frame))

            # Accumulate
            Z_cpu += Gi_cpu.dot(Zi_cpu)

            print("Processed frame %d" % idx)

        # Build 3D-3D graph on GPU
        G_cpu = csr_matrix(self.build_subgraph_3d3d(points, normals))

        # Perform row normalization
        row_sums = G_cpu.sum(axis=1).ravel() + Z_cpu.sum(axis=1).ravel()
        row_sums = np.squeeze(np.asarray(row_sums))
        Z_cpu /= row_sums[:, np.newaxis]
        G_cpu.data /= row_sums.repeat(np.diff(G_cpu.indptr))

        # Move initial zero labels to GPU
        Y_cpu = np.zeros(Z_cpu.shape)

        print("Starting label diffusion")
        start = timer()

        for i in range(max_iters):
            Y_new = G_cpu.dot(Y_cpu) + Z_cpu
            Y_cpu = Y_new

        end = timer()

        print("Done!")
        print("Label diffusion took %f s" %(end - start))

        label_likelihoods = Y_cpu
        labels = np.argmax(Y_cpu, axis=1)

        return labels, label_likelihoods
