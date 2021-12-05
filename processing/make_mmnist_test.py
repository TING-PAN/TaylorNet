import argparse
import numpy as np
from tqdm import trange
import os
from data.moving_mnist import MovingMNIST, load_mnist

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type = str, default = '../dataset/mmnist')
	parser.add_argument('--target_len', type = int, metavar = 'LEN', default = 390)
	parser.add_argument('--seed', type = int, metavar = 'SEED', default = 35,
	                    help = 'Fixed NumPy seed to produce the same dataset at each run.')
	parser.add_argument('--frame_size', type = int, metavar = 'SIZE', default = 64)
	parser.add_argument('--size', type = int, metavar = 'SIZE', default = 5000)
	parser.add_argument('--digits', type = int, metavar = 'NUM', default = 2)
	args = parser.parse_args()

	# Fix random seed
	np.random.seed(args.seed)
	# Load digits and shuffle them
	digits = load_mnist(args.data_dir)
	digits_idx = np.random.permutation(len(digits))

	trajectory_sampler = MovingMNIST(root = args.data_dir, is_train = True, n_frames_input = 10, n_frames_output = args.target_len, num_objects = [2])
	total_len = 10 + args.target_len

	test_videos = []

	for i in trange(args.size):
		x = np.zeros((total_len, args.frame_size, args.frame_size), dtype = np.float32)

		# Pick the digits randomly chosen for sequence i and generate their trajectories
		for n in range(args.digits):
			img = digits[digits_idx[i * args.digits + n]]
			img = np.array(img, dtype = np.uint8)
			start_y, start_x = trajectory_sampler.get_random_trajectory(total_len)

			for t in range(total_len):
				top = start_y[t]
				left = start_x[t]
				bottom = top + img.shape[0]
				right = left + img.shape[1]
				# Draw digit
				x[t, top:bottom, left:right] += img

		x[x > 255] = 255
		test_videos.append(x.astype(np.uint8))

	test_videos = np.array(test_videos, dtype = np.uint8).transpose(1, 0, 2, 3)

	# Save results at the given path
	fname = f'mmnist_test_{total_len}.npy'
	path = os.path.join(args.data_dir, fname)
	# print(f'Saving testset at {path}')
	np.save(path, test_videos)
