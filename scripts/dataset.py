import os
import numpy as np
from urllib import request
import errno
import struct
import tarfile
import glob
import scipy.io as sio
from sklearn.utils.extmath import cartesian
from scipy.stats import laplace
import joblib
from spriteworld import factor_distributions as distribs
from spriteworld import renderers as spriteworld_renderers
from spriteworld import sprite
import csv
from collections import defaultdict
import ast
from scripts.data_analysis_utils import load_csv
import pandas as pd
from sklearn import preprocessing
from sklearn import utils
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
from PIL import Image
import pickle
import h5py
from matplotlib import pyplot as plt


class TupleLoader(Dataset):
	def __init__(self, k=-1, rate=1, prior='uniform', transform=None,
				 target_transform=None):
		# k=-1 gives random number of changing factors as in Locatello k=rand
		# rate=-1 gives random rate for each sample in Uniform(1,10)
		self.index_manager = None   # set in child class
		self.factor_sizes = None  # set in child class
		self.categorical = None  # set in child class
		self.data = None  # set in child class

		self.prior = prior
		self.transform = transform
		self.target_transform = target_transform
		self.rate = rate
		self.k = k

		if prior == 'laplace' and k != -1:
			print('warning setting k has no effect on prior=laplace. '
				  'Set k=-1 or leave to default to get rid of this warning.')

		if prior == 'uniform' and rate != -1:
			print('warning setting rate has no effect on prior=uniform. '
				  'Set rate=-1 or leave to default to get rid of this warning.')

	def __len__(self):
		return len(self.data)

	def sample_factors(self, num, random_state):
		"""Sample a batch of observations X. Needed in dis. lib."""
		assert not(num % 2)
		batch_size = int(num / 2)
		indices = random_state.choice(self.__len__(), 2 * batch_size, replace=False)
		batch, latents = [], []
		for ind in indices:
			_, _, l1, _ = self.__getitem__(ind)
			latents.append(l1)
		return np.stack(latents)
	
	def sample_observations_from_factors(self, factors, random_state):
		batch = []
		for factor in factors:
			sample_ind = self.index_manager.features_to_index(factor)
			sample = self.data[sample_ind]
			if self.transform:
				sample = self.transform(sample)
			if len(sample.shape) == 2:  # set channel dim to 1
				sample = sample[None]
			if np.issubdtype(sample.dtype, np.uint8):
				sample = sample.astype(np.float32) / 255.
			batch.append(sample)
		return np.stack(batch)

	def sample(self, num, random_state):
		#Sample a batch of factors Y and observations X
		factors = self.sample_factors(num, random_state)
		return factors, self.sample_observations_from_factors(factors, random_state)

	def sample_observations(self, num, random_state):
		#Sample a batch of observations X
		return self.sample(num, random_state)[1]

	def __getitem__(self, idx):
		n_factors = len(self.factor_sizes)
		first_sample = self.data[idx]

		first_sample_feat = self.index_manager.index_to_features(idx)
		if self.prior == 'uniform':
			# only change up to k factors
			if self.k == -1:
				k = np.random.randint(1, n_factors)  # number of factors which can change
			else:
				k = self.k

			second_sample_feat = first_sample_feat.copy()
			indices = np.random.choice(n_factors, k, replace=False)
			for ind in indices:
				x = np.arange(self.factor_sizes[ind])
				p = np.ones_like(x) / (x.shape[0] - 1)
				p[x == first_sample_feat[ind]] = 0  # dont pick same

				second_sample_feat[ind] = np.random.choice(x, 1, p=p)
			assert np.equal(first_sample_feat - second_sample_feat, 0).sum() == n_factors - k

		elif self.prior == 'laplace':
			second_sample_feat = self.truncated_laplace(first_sample_feat)
		else:
			raise NotImplementedError
		second_sample_ind = self.index_manager.features_to_index(second_sample_feat)
		second_sample = self.data[second_sample_ind]

		if self.transform:
			first_sample = self.transform(first_sample)
			second_sample = self.transform(second_sample)

		if len(first_sample.shape) == 2:  # set channel dim to 1
			first_sample = first_sample[None]
			second_sample = second_sample[None]

		if np.issubdtype(first_sample.dtype, np.uint8) or np.issubdtype(second_sample.dtype, np.uint8):
			first_sample = first_sample.astype(np.float32) / 255.
			second_sample = second_sample.astype(np.float32) / 255.

		if self.target_transform:
			first_sample_feat = self.target_transform(first_sample_feat)
			second_sample_feat = self.target_transform(second_sample_feat)

		return first_sample, second_sample, first_sample_feat, second_sample_feat

	def truncated_laplace(self, start):
		if self.rate == -1:
			rate = np.random.uniform(1, 10, 1)[0]
		else:
			rate = self.rate
		end = []
		n_factors = len(self.factor_sizes)
		for mean, upper in zip(start, np.array(self.factor_sizes)):  # sample each feature individually
			x = np.arange(upper)
			p = laplace.pdf(x, loc=mean, scale=np.log(upper) / rate)
			p /= np.sum(p)
			end.append(np.random.choice(x, 1, p=p)[0])
		end = np.array(end).astype(np.int)
		end[self.categorical] = start[self.categorical]   # don't change categorical factors s.a. shape
		# make sure there is at least one change
		if np.sum(abs(start - end)) == 0:
			ind = np.random.choice(np.arange(n_factors)[~self.categorical], 1)[0]  # don't change categorical factors
			x = np.arange(self.factor_sizes[ind])
			p = laplace.pdf(x, loc=start[ind],
							scale=np.log(self.factor_sizes[ind]) / rate)
			p[x == start[ind]] = 0
			p /= np.sum(p)
			end[ind] = np.random.choice(x, 1, p=p)
		assert np.sum(abs(start - end)) > 0
		return end


class IndexManger(object):
	"""Index mapping from features to positions of state space atoms."""

	def __init__(self, factor_sizes):
		"""Index to latent (= features) space and vice versa.
		Args:
		  factor_sizes: List of integers with the number of distinct values for each
			of the factors.
		"""
		self.factor_sizes = np.array(factor_sizes)
		self.num_total = np.prod(self.factor_sizes)
		self.factor_bases = self.num_total / np.cumprod(self.factor_sizes)

		self.index_to_feat = cartesian([np.array(list(range(i))) for i in self.factor_sizes])

	def features_to_index(self, features):
		"""Returns the indices in the input space for given factor configurations.
		Args:
		  features: Numpy matrix where each row contains a different factor
			configuration for which the indices in the input space should be
			returned.
		"""
		assert np.all((0 <= features) & (features <= self.factor_sizes))
		index = np.array(np.dot(features, self.factor_bases), dtype=np.int64)
		assert np.all((0 <= index) & (index < self.num_total))
		return index

	def index_to_features(self, index):
		assert np.all((0 <= index) & (index < self.num_total))
		features = self.index_to_feat[index]
		assert np.all((0 <= features) & (features <= self.factor_sizes))
		return features


class Cars3D(TupleLoader):
	fname = 'nips2015-analogy-data.tar.gz'
	url = 'http://www.scottreed.info/files/nips2015-analogy-data.tar.gz'
	"""
	[4, 24, 183]
	0. phi altitude viewpoint
	1. theta azimuth viewpoint
	2. car type
	"""

	def __init__(self, path='./data/cars/', data=None, **tupel_loader_kwargs):
		super().__init__(**tupel_loader_kwargs)
		self.factor_sizes = [4, 24, 183]
		self.num_factors = len(self.factor_sizes)
		self.categorical = np.array([False, False, True])
		self.data_shape = [64, 64, 3]
		self.index_manager = IndexManger(self.factor_sizes)

		# download automatically if not exists
		if not os.path.exists(path):
			self.download_data(path)
		if data is None:
			all_files = glob.glob(path + '/*.mat')
			self.data = np.moveaxis(self._load_data(all_files).astype(np.float32), 3, 1)
		else:   # speedup for debugging
			self.data = data

	def _load_data(self, all_files):

		def _load_mesh(filename):
			"""Parses a single source file and rescales contained images."""
			with open(os.path.join(filename), "rb") as f:
				mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
			flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
			rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
			for i in range(flattened_mesh.shape[0]):
				pic = Image.fromarray(flattened_mesh[i, :, :, :])
				pic.thumbnail((64, 64), Image.ANTIALIAS)
				rescaled_mesh[i, :, :, :] = np.array(pic)
			return rescaled_mesh * 1. / 255

		dataset = np.zeros((24 * 4 * 183, 64, 64, 3))

		for i, filename in enumerate(all_files):
			data_mesh = _load_mesh(filename)
			factor1 = np.array(list(range(4)))
			factor2 = np.array(list(range(24)))
			all_factors = np.transpose([np.tile(factor1, len(factor2)),
										np.repeat(factor2, len(factor1)),
										np.tile(i, len(factor1) * len(factor2))])
			indexes = self.index_manager.features_to_index(all_factors)
			dataset[indexes] = data_mesh
		return dataset

	def download_data(self, load_path='./data/cars/'):
		os.makedirs(load_path, exist_ok=True)
		print('downlading data may take a couple of seconds, total ~ 300MB')
		request.urlretrieve(self.url, os.path.join(load_path, self.fname))
		print('extracting data, do NOT interrupt')
		tar = tarfile.open(os.path.join(load_path, self.fname), "r:gz")
		tar.extractall()
		tar.close()
		print('saved data at', load_path)


class SmallNORB(TupleLoader):
	"""`MNIST <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small//>`_ Dataset.

	factors:
	[5, 10, 9, 18, 6]
	- 0. (0 to 4) 0 for animal, 1 for human, 2 for plane, 3 for truck, 4 for car).
	- 1. the instance in the category (0 to 9)
	- 2. the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
	- 3. the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
	- 4. the lighting condition (0 to 5)

	"""

	dataset_root = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
	data_files = {
		'train': {
			'dat': {
				"name": 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
				"md5_gz": "66054832f9accfe74a0f4c36a75bc0a2",
				"md5": "8138a0902307b32dfa0025a36dfa45ec"
			},
			'info': {
				"name": 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
				"md5_gz": "51dee1210a742582ff607dfd94e332e3",
				"md5": "19faee774120001fc7e17980d6960451"
			},
			'cat': {
				"name": 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
				"md5_gz": "23c8b86101fbf0904a000b43d3ed2fd9",
				"md5": "fd5120d3f770ad57ebe620eb61a0b633"
			},
		},
		'test': {
			'dat': {
				"name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
				"md5_gz": "e4ad715691ed5a3a5f138751a4ceb071",
				"md5": "e9920b7f7b2869a8f1a12e945b2c166c"
			},
			'info': {
				"name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
				"md5_gz": "a9454f3864d7fd4bb3ea7fc3eb84924e",
				"md5": "7c5b871cc69dcadec1bf6a18141f5edc"
			},
			'cat': {
				"name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
				"md5_gz": "5aa791cd7e6016cf957ce9bdb93b8603",
				"md5": "fd5120d3f770ad57ebe620eb61a0b633"
			},
		},
	}

	raw_folder = 'raw'
	processed_folder = 'processed'
	train_image_file = 'train_img'
	train_label_file = 'train_label'
	train_info_file = 'train_info'
	test_image_file = 'test_img'
	test_label_file = 'test_label'
	test_info_file = 'test_info'
	extension = '.pt'

	def __init__(self, path='./data/smallNORB/', download=True,
				 mode="all",
				 transform=None,
		     		 evaluate=False,
				 **tupel_loader_kwargs):
		super().__init__(**tupel_loader_kwargs)

		self.root = os.path.expanduser(path)
		self.mode = mode
		self.evaluate = evaluate
		self.factor_sizes = [5, 10, 9, 18, 6]
		self.latent_factor_indices = [0, 2, 3, 4]
		self.num_factors = len(self.latent_factor_indices)
		self.categorical = np.array([True, True, False, False, False])
		self.index_manager = IndexManger(self.factor_sizes)

		if transform:
			self.transform = transform
		else:
			self.transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize((64, 64), interpolation=2),
				transforms.ToTensor(),
				lambda x: x.numpy()])
		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')

		# load labels
		labels_train = self._load(self.train_label_file)
		labels_test = self._load(self.test_label_file)

		# load info files
		infos_train = self._load(self.train_info_file)
		infos_test = self._load(self.test_info_file)

		# load right set
		data_train = self._load("{}_left".format(self.train_image_file))
		data_test = self._load("{}_left".format(self.test_image_file))

		info_train = torch.cat([labels_train[:, None], infos_train], dim=1)
		info_test = torch.cat([labels_test[:, None], infos_test], dim=1)

		infos = torch.cat([info_train, info_test])
		data = torch.cat([data_train, data_test])
		sorted_inds = np.lexsort([infos[:, i] for i in range(4, -1, -1)])

		self.infos = infos[sorted_inds]
		self.data = data[sorted_inds].numpy()  # is uint8
	
	def sample_factors(self, num, random_state):
		# override super to ignore instance (see https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/norb.py#L52)
		factors = super().sample_factors(num, random_state)
		if self.evaluate:
			factors = np.concatenate([factors[:, :1], factors[:, 2:]], 1)
		return factors
	
	def sample_observations_from_factors(self, factors, random_state):
		# override super to ignore instance (see https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/norb.py#L52)
		if self.evaluate:
			instances = random_state.randint(0, self.factor_sizes[1], factors[:, :1].shape)
			factors = np.concatenate([factors[:, :1], instances, factors[:, 1:]], 1)
		return super().sample_observations_from_factors(factors, random_state)

	def __len__(self):
		return len(self.data)

	def _transform(self, img):
		# doing this so that it is consistent with all other data sets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			img = self.transform(img)
		return img

	def _load(self, file_name):
		return torch.load(os.path.join(self.root, self.processed_folder, file_name + self.extension))

	def _save(self, file, file_name):
		with open(os.path.join(self.root, self.processed_folder, file_name + self.extension), 'wb') as f:
			torch.save(file, f)

	def _check_exists(self):
		""" Check if processed files exists."""
		files = (
			"{}_left".format(self.train_image_file),
			"{}_right".format(self.train_image_file),
			"{}_left".format(self.test_image_file),
			"{}_right".format(self.test_image_file),
			self.test_label_file,
			self.train_label_file
		)
		fpaths = [os.path.exists(os.path.join(self.root, self.processed_folder, f + self.extension)) for f in files]
		return False not in fpaths

	def _flat_data_files(self):
		return [j for i in self.data_files.values() for j in list(i.values())]

	def _check_integrity(self):
		"""Check if unpacked files have correct md5 sum."""
		root = self.root
		for file_dict in self._flat_data_files():
			filename = file_dict["name"]
			md5 = file_dict["md5"]
			fpath = os.path.join(root, self.raw_folder, filename)
			if not check_integrity(fpath, md5):
				return False
		return True

	def download(self):
		"""Download the SmallNORB data if it doesn't exist in processed_folder already."""
		import gzip

		if self._check_exists():
			return

		# check if already extracted and verified
		if self._check_integrity():
			print('Files already downloaded and verified')
		else:
			# download and extract
			for file_dict in self._flat_data_files():
				url = self.dataset_root + file_dict["name"] + '.gz'
				filename = file_dict["name"]
				gz_filename = filename + '.gz'
				md5 = file_dict["md5_gz"]
				fpath = os.path.join(self.root, self.raw_folder, filename)
				gz_fpath = fpath + '.gz'

				# download if compressed file not exists and verified
				download_url(url, os.path.join(self.root, self.raw_folder), gz_filename, md5)

				print('# Extracting data {}\n'.format(filename))

				with open(fpath, 'wb') as out_f, \
						gzip.GzipFile(gz_fpath) as zip_f:
					out_f.write(zip_f.read())

				os.unlink(gz_fpath)

		# process and save as torch files
		print('Processing...')

		# create processed folder
		try:
			os.makedirs(os.path.join(self.root, self.processed_folder))
		except OSError as e:
			if e.errno == errno.EEXIST:
				pass
			else:
				raise

		# read train files
		left_train_img, right_train_img = self._read_image_file(self.data_files["train"]["dat"]["name"])
		train_info = self._read_info_file(self.data_files["train"]["info"]["name"])
		train_label = self._read_label_file(self.data_files["train"]["cat"]["name"])

		# read test files
		left_test_img, right_test_img = self._read_image_file(self.data_files["test"]["dat"]["name"])
		test_info = self._read_info_file(self.data_files["test"]["info"]["name"])
		test_label = self._read_label_file(self.data_files["test"]["cat"]["name"])

		# save training files
		self._save(left_train_img, "{}_left".format(self.train_image_file))
		self._save(right_train_img, "{}_right".format(self.train_image_file))
		self._save(train_label, self.train_label_file)
		self._save(train_info, self.train_info_file)

		# save test files
		self._save(left_test_img, "{}_left".format(self.test_image_file))
		self._save(right_test_img, "{}_right".format(self.test_image_file))
		self._save(test_label, self.test_label_file)
		self._save(test_info, self.test_info_file)

		print('Done!')

	@staticmethod
	def _parse_header(file_pointer):
		# Read magic number and ignore
		struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

		# Read dimensions
		dimensions = []
		num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
		for _ in range(num_dims):
			dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

		return dimensions

	def _read_image_file(self, file_name):
		fpath = os.path.join(self.root, self.raw_folder, file_name)
		with open(fpath, mode='rb') as f:
			dimensions = self._parse_header(f)
			assert dimensions == [24300, 2, 96, 96]
			num_samples, _, height, width = dimensions

			left_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)
			right_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

			for i in range(num_samples):
				# left and right images stored in pairs, left first
				left_samples[i, :, :] = self._read_image(f, height, width)
				right_samples[i, :, :] = self._read_image(f, height, width)

		return torch.ByteTensor(left_samples), torch.ByteTensor(right_samples)

	@staticmethod
	def _read_image(file_pointer, height, width):
		"""Read raw image data and restore shape as appropriate. """
		image = struct.unpack('<' + height * width * 'B', file_pointer.read(height * width))
		image = np.uint8(np.reshape(image, newshape=(height, width)))
		return image

	def _read_label_file(self, file_name):
		fpath = os.path.join(self.root, self.raw_folder, file_name)
		with open(fpath, mode='rb') as f:
			dimensions = self._parse_header(f)
			assert dimensions == [24300]
			num_samples = dimensions[0]

			struct.unpack('<BBBB', f.read(4))  # ignore this integer
			struct.unpack('<BBBB', f.read(4))  # ignore this integer

			labels = np.zeros(shape=num_samples, dtype=np.int32)
			for i in range(num_samples):
				category, = struct.unpack('<i', f.read(4))
				labels[i] = category
			return torch.LongTensor(labels)

	def _read_info_file(self, file_name):
		fpath = os.path.join(self.root, self.raw_folder, file_name)
		with open(fpath, mode='rb') as f:

			dimensions = self._parse_header(f)
			assert dimensions == [24300, 4]
			num_samples, num_info = dimensions

			struct.unpack('<BBBB', f.read(4))  # ignore this integer

			infos = np.zeros(shape=(num_samples, num_info), dtype=np.int32)

			for r in range(num_samples):
				for c in range(num_info):
					info, = struct.unpack('<i', f.read(4))
					infos[r, c] = info

		return torch.LongTensor(infos)


class Shapes3D(TupleLoader):
	"""Shapes3D dataset.
	self.factor_sizes = [10, 10, 10, 8, 4, 15]

	The data set was originally introduced in "Disentangling by Factorising".
	The ground-truth factors of variation are:
	0 - floor color (10 different values)
	1 - wall color (10 different values)
	2 - object color (10 different values)
	3 - object size (8 different values)
	4 - object type (4 different values)
	5 - azimuth (15 different values)
	"""
	#url = 'https://liquidtelecom.dl.sourceforge.net/project/shapes3d/Shapes3D.zip'
	#fname = 'shapes3d.pkl'
	url = 'https://storage.googleapis.com/3d-shapes/3dshapes.h5'
	fname = '3dshapes.h5'

	def __init__(self, path='./data/shapes3d/', data=None, **tupel_loader_kwargs):
		super().__init__(**tupel_loader_kwargs)

		self.factor_sizes = [10, 10, 10, 8, 4, 15]
		self.num_factors = len(self.factor_sizes)
		self.categorical = np.array([False, False, False, False, True, False])
		self.index_manager = IndexManger(self.factor_sizes)

		self.path = path

		if not os.path.exists(self.path):
			self.download()

		# read dataset
		print('init of shapes dataset (takes a couple of seconds) (large data array)')
		if data is None:
			with h5py.File(os.path.join(self.path, self.fname), 'r') as dataset:
				images = dataset['images'][()]
			self.data = np.transpose(images, (0, 3, 1, 2))   # np.uint8
		else:
			self.data = data

	def download(self):
		print('downloading shapes3d')
		os.makedirs(self.path, exist_ok=True)
		request.urlretrieve(self.url, os.path.join(self.path, self.fname))


class SpriteDataset(TupleLoader):
	"""
	A PyTorch wrapper for the dSprites dataset by
	Matthey et al. 2017. The dataset provides a 2D scene
	with a sprite under different transformations:
	# dim, type,     #values  avail.-range
	* 0, color       |  1 |     1-1
	* 1, shape       |  3 |     1-3
	* 2, scale       |  6 |     0.5-1.
	* 3, orientation | 40 |     0-2pi
	* 4, x-position  | 32 |     0-1
	* 5, y-position  | 32 |     0-1
	for details see https://github.com/deepmind/dsprites-dataset
	"""

	def __init__(self, path='./data/dsprites/', **tupel_loader_kwargs):
		super().__init__(**tupel_loader_kwargs)

		url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

		self.path = path
		self.factor_sizes = [3, 6, 40, 32, 32]
		self.num_factors = len(self.factor_sizes)
		self.categorical = np.array([True, False, False, False, False])
		self.index_manager = IndexManger(self.factor_sizes)

		try:
			self.data = self.load_data()
		except FileNotFoundError:
			if not os.path.exists(path):
				os.makedirs(path, exist_ok=True)
			print(
				f'downloading dataset ... saving to {os.path.join(path, "dsprites.npz")}')
			request.urlretrieve(url, os.path.join(path, 'dsprites.npz'))
			self.data = self.load_data()

	def __len__(self):
		return len(self.data)

	def load_data(self):
		dataset_zip = np.load(os.path.join(self.path, 'dsprites.npz'),
							  encoding="latin1", allow_pickle=True)
		return dataset_zip["imgs"].squeeze().astype(np.float32)

class MPI3DReal(TupleLoader):
	"""
	object_color	white=0, green=1, red=2, blue=3, brown=4, olive=5
	object_shape	cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
	object_size	small=0, large=1
	camera_height	top=0, center=1, bottom=2
	background_color	purple=0, sea green=1, salmon=2
	horizontal_axis	0,...,39
	vertical_axis	0,...,39
	"""
	url = 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz'
	fname = 'mpi3d_real.npz'

	def __init__(self, path='./data/mpi3d_real/', **tupel_loader_kwargs):
		super().__init__(**tupel_loader_kwargs)

		self.factor_sizes = [6, 6, 2, 3, 3, 40, 40]
		self.num_factors = len(self.factor_sizes)
		self.categorical = np.array([False, True, False, False, False, False, False])
		self.index_manager = IndexManger(self.factor_sizes)
		if not os.path.exists(path):
			self.download(path)

		load_path = os.path.join(path, self.fname)
		data = np.load(load_path)['images']
		self.data = np.transpose(data.reshape([-1, 64, 64, 3]), (0, 3, 1, 2))  # np.uint8

	def download(self, path):
		os.makedirs(path, exist_ok=True)
		print('downloading')
		request.urlretrieve(self.url, os.path.join(path, self.fname))
		print('download complete')

def value_to_key(x, val):
	for k in x.keys():
		if x[k] == val:
			return k
				
def rgb(c):
	return tuple((255 * np.array(c)).astype(np.uint8))

class NaturalSprites(Dataset):
	def __init__(self, natural_discrete=False, path='./data/natural_sprites/'):
		self.natural_discrete = natural_discrete
		self.sequence_len = 2 #only consider pairs
		self.area_filter = 0.1 #filter out 10% of outliers
		self.path = path
		self.fname = 'downscale_keepaspect.csv'
		self.url = 'https://zenodo.org/record/3948069/files/downscale_keepaspect.csv?download=1'
		self.load_data()
				
	def load_data(self):
		# download if not avaiable
		file_path = os.path.join(self.path, self.fname)
		if not os.path.exists(file_path):
			os.makedirs(self.path, exist_ok=True)
			print(f'file not found, downloading from {self.url} ...')
			from urllib import request
			url = self.url
			request.urlretrieve(url, file_path)
		with open(file_path) as data:
			self.csv_dict = load_csv(data, sequence=self.sequence_len)
		self.orig_num = [32, 32, 6, 40, 4, 1, 1, 1]
		self.dsprites = {'x': np.linspace(0.2,0.8,self.orig_num[0]), 
					 'y': np.linspace(0.2,0.8,self.orig_num[1]), 
					 'scale': np.linspace(0,0.5,self.orig_num[2]+1)[1:], 
					 'angle': np.linspace(0,360,self.orig_num[3],dtype=np.int,endpoint=False), 
					 'shape': ['square', 'triangle', 'star_4', 'spoke_4'], 
					 'c0': [1.], 'c1': [1.], 'c2': [1.]}
		distributions = []
		for key in self.dsprites.keys():
			distributions.append(distribs.Discrete(key, self.dsprites[key]))
		self.factor_dist = distribs.Product(distributions)
		self.renderer = spriteworld_renderers.PILRenderer(image_size=(64, 64), anti_aliasing=5,
														  color_to_rgb=rgb)
		if self.area_filter:
			keep_idxes = []
			print(len(self.csv_dict['x']))
			for i in range(self.sequence_len):
				x = pd.Series(np.array(self.csv_dict['area'])[:,i])
				keep_idxes.append(x.between(x.quantile(self.area_filter/2), x.quantile(1-(self.area_filter/2))))
			for k in self.csv_dict.keys():
				y = pd.Series(self.csv_dict[k])
				self.csv_dict[k] = np.array([x for x in y[np.logical_and(*keep_idxes)]])
			print(len(self.csv_dict['x']))
		if self.natural_discrete:
			num_bins = self.orig_num[:3]
			self.lab_encs = {}
			print('num_bins', num_bins)
			for i,key in enumerate(['x','y','area']):
				count, bin_edges = np.histogram(np.array(self.csv_dict[key]).flatten().tolist(), bins=num_bins[i])
				bin_left, bin_right = bin_edges[:-1], bin_edges[1:]
				bin_centers = bin_left + (bin_right - bin_left)/2
				new_data = []
				old_shape = np.array(self.csv_dict[key]).shape
				lab_enc = preprocessing.LabelEncoder()
				if key == 'area':
					self.lab_encs['scale'] = lab_enc.fit(np.sqrt(bin_centers/(64**2)))
				else:
					self.lab_encs[key] = lab_enc.fit(bin_centers/64)
				for j in range(self.sequence_len):
					differences = (np.array(self.csv_dict[key])[:,j].reshape(1,-1) - bin_centers.reshape(-1,1))
					new_data.append([bin_centers[x] for x in np.abs(differences).argmin(axis=0)])
				self.csv_dict[key] = np.swapaxes(new_data, 0, 1)
				assert old_shape == np.array(self.csv_dict[key]).shape
				assert len(np.unique(np.array(self.csv_dict[key]).flatten())) == num_bins[i]
			for i,key in enumerate(['angle', 'shape', 'c0', 'c1', 'c2']):
				lab_enc = preprocessing.LabelEncoder()
				self.lab_encs[key] = lab_enc.fit(self.dsprites[key])
			assert self.lab_encs.keys() == self.dsprites.keys()
		self.factor_sizes = [len(np.unique(np.array(self.csv_dict['x']).flatten())),
							 len(np.unique(np.array(self.csv_dict['y']).flatten())),
							 len(np.unique(np.array(self.csv_dict['area']).flatten())),
							 40,4,1,1,1]
		print(self.factor_sizes)
		self.latent_factor_indices = list(range(5))
		self.num_factors = len(self.latent_factor_indices)
		self.observation_factor_indices = [i for i in range(self.num_factors) if i not in self.latent_factor_indices]
		self.mapping = {'square': 0, 
						'triangle': 1, 
						'star_4': 2, 
						'spoke_4': 3}	
		
	def __getitem__(self, index):
		sampled_latents = self.factor_dist.sample()
		idx = np.random.choice(len(self.csv_dict['id']), p=None)
		sprites = []
		latents = []
		for i in range(self.sequence_len):
			curr_latents = sampled_latents.copy()
			csv_vals = [self.csv_dict['x'][idx][i], self.csv_dict['y'][idx][i], self.csv_dict['area'][idx][i]]
			curr_latents['x'] = csv_vals[0]/64
			curr_latents['y'] = csv_vals[1]/64
			curr_latents['scale'] = np.sqrt(csv_vals[2]/(64**2))
			sprites.append(sprite.Sprite(**curr_latents))
			latents.append(curr_latents)
		first_sample = np.transpose(self.renderer.render(sprites=[sprites[0]]).astype(np.float32) / 255., (2,0,1))
		second_sample = np.transpose(self.renderer.render(sprites=[sprites[1]]).astype(np.float32) / 255., (2,0,1))
		latents1 = np.array([self.convert_cat(item) for item in latents[0].values()])
		latents2 = np.array([self.convert_cat(item) for item in latents[1].values()])
		return first_sample, second_sample, latents1, latents2

	def __len__(self):
		return len(self.csv_dict['id'])

	def sample_factors(self, num, random_state):
		#Sample a batch of factors Y
		if self.natural_discrete:
			factors = np.zeros(shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
			for pos, i in enumerate(self.latent_factor_indices):
				factors[:, pos] = random_state.randint(self.factor_sizes[i], size=num)
			return factors
		else:
			factors = []
			for n in range(num):
				sampled_latents = self.factor_dist.sample()
				idx = random_state.choice(len(self.csv_dict['id']), p=None)
				sampled_latents['x'] = self.csv_dict['x'][idx][0]/64
				sampled_latents['y'] = self.csv_dict['y'][idx][0]/64
				sampled_latents['scale'] = np.sqrt(self.csv_dict['area'][idx][0]/(64**2))
				factors.append(np.array([self.convert_cat(item) for item in sampled_latents.values()]))
			return np.array(factors)[:,:5]

	def sample_observations_from_factors(self, factors, random_state):
		#Sample a batch of observations X given a batch of factors Y.
		images = []
		for f in factors:
			if self.natural_discrete:
				f_convert = []
				for i in self.latent_factor_indices:
					f_convert.append(list(self.lab_encs[list(self.dsprites.keys())[i]].inverse_transform([f[i]]))[0])
				rendering = self.renderer.render(sprites=[sprite.Sprite(**self.back_to_dict(f_convert, False))])
			else:
				rendering = self.renderer.render(sprites=[sprite.Sprite(**self.back_to_dict(f, True))])
			images.append(np.transpose(rendering.astype(np.float32) / 255., (2,0,1)))
		return np.array(images)
	def sample(self, num, random_state):
		#Sample a batch of factors Y and observations X
		factors = self.sample_factors(num, random_state)
		return factors, self.sample_observations_from_factors(factors, random_state)

	def sample_observations(self, num, random_state):
		#Sample a batch of observations X
		return self.sample(num, random_state)[1]

	def convert_cat(self, item):
		if type(item) is str:
			return self.mapping[item]
		else:
			return item

	def back_to_dict(self, x, continuous=True):
		res = {}
		for i,k in enumerate(list(self.dsprites.keys())[:5]):
			if continuous and k == 'shape':
				res[k] = value_to_key(self.mapping, x[i])
			else:
				res[k] = x[i]
		res['c0'] = 1.
		res['c1'] = 1.
		res['c2'] = 1.
		return res


class KittiMasks(Dataset):
	'''
	latents encode:
	0: center of mass vertical position
	1: center of mass horizontal position
	2: area
	'''
	def __init__(self, path='./data/kitti/', transform=None,
				 max_delta_t=5):
		self.path = path
		self.data = None
		self.latents = None
		self.lens = None
		self.cumlens = None
		self.max_delta_t = max_delta_t
		self.fname = 'kitti_peds_v2.pickle'
		self.url = 'https://zenodo.org/record/3931823/files/kitti_peds_v2.pickle?download=1'

		if transform == 'default':
			self.transform = transforms.Compose(
				[
					transforms.ToPILImage(),
					transforms.RandomAffine(degrees=(2., 2.), translate=(5 / 64., 5 / 64.)),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					lambda x: x.numpy()
				])
		else:
			self.transform = None

		self.load_data()

	def load_data(self):
		# download if not avaiable
		file_path = os.path.join(self.path, self.fname)
		if not os.path.exists(file_path):
			os.makedirs(self.path, exist_ok=True)
			print(f'file not found, downloading from {self.url} ...')
			from urllib import request
			url = self.url
			request.urlretrieve(url, file_path)

		with open(file_path, 'rb') as data:
			data = pickle.load(data)
		self.data = data['pedestrians']
		self.latents = data['pedestrians_latents']

		self.lens = [len(seq) - 1 for seq in self.data]  # start image in sequence can never be starting point
		self.cumlens = np.cumsum(self.lens)

	def sample_observations(self, num, random_state, return_latents=False):
		"""Sample a batch of observations X. Needed in dis. lib."""
		assert not (num % 2)
		batch_size = int(num / 2)
		indices = random_state.choice(self.__len__(), 2 * batch_size, replace=False)
		batch, latents = [], []
		for ind in indices:
			first_sample, second_sample, l1, l2 = self.__getitem__(ind)
			batch.append(first_sample)
			latents.append(l1)
		batch = np.stack(batch)
		if not return_latents:
			return batch
		else:
			return batch, np.stack(latents)

	def sample(self, num, random_state):
		# Sample a batch of factors Y and observations X
		x, y = self.sample_observations(num, random_state, return_latents=True)
		return y, x

	def __getitem__(self, index):
		sequence_ind = np.searchsorted(self.cumlens, index, side='right')
		if sequence_ind == 0:
			start_ind = index
		else:
			start_ind = index - self.cumlens[sequence_ind - 1]
		seq_len = len(self.data[sequence_ind])
		t_steps_forward = np.random.randint(1, self.max_delta_t + 1)
		end_ind = min(start_ind + t_steps_forward, seq_len - 1)

		first_sample = self.data[sequence_ind][start_ind].astype(np.uint8) * 255
		second_sample = self.data[sequence_ind][end_ind].astype(np.uint8) * 255

		latents1 = self.latents[sequence_ind][start_ind]  # center of mass vertical, com hor, area
		latents2 = self.latents[sequence_ind][end_ind]  # center of mass vertical, com hor, area

		if self.transform:
			stack = np.concatenate([first_sample[:, :, None],
									second_sample[:, :, None],
									np.ones_like(second_sample[:, :, None]) * 255],  # add ones to treat like RGB image
								   axis=2)
			samples = self.transform(stack)  # do same transforms to start and ending
			first_sample, second_sample = samples[0], samples[1]

		if len(first_sample.shape) == 2:  # set channel dim to 1
			first_sample = first_sample[None]
			second_sample = second_sample[None]

		if np.issubdtype(first_sample.dtype, np.uint8) or np.issubdtype(second_sample.dtype, np.uint8):
			first_sample = first_sample.astype(np.float32) / 255.
			second_sample = second_sample.astype(np.float32) / 255.

		return first_sample, second_sample, latents1, latents2

	def __len__(self):
		return self.cumlens[-1]


def custom_collate(sample):
	inputs, labels = [], []
	for s in sample:
		inputs.append(s[0])
		inputs.append(s[1])
		labels.append(s[2])
		labels.append(s[3])
	return torch.tensor(np.stack(inputs)), torch.tensor(np.stack(labels))

def return_data(args):
	name = args.dataset
	batch_size = args.batch_size
	num_workers = args.num_workers
	image_size = args.image_size
	assert image_size == 64, 'currently only image size of 64 is supported'
	# half batch_size for video couples
	assert not (batch_size % 2)
	batch_size = batch_size // 2
	num_channel = 1

	if name.lower() == 'dsprites':
		train_data = SpriteDataset(
			prior=args.data_distribution,
			rate=args.rate_data,
			k=args.data_k)

	elif name.lower() == 'cars3d':
		train_data = Cars3D(
			prior=args.data_distribution,
			rate=args.rate_data,
			k=args.data_k)
		num_channel = 3

	elif name.lower() == 'smallnorb':
		train_data = SmallNORB(
			prior=args.data_distribution,
			rate=args.rate_data,
			k=args.data_k,
			evaluate=args.evaluate)

	elif name.lower() == 'shapes3d':
		train_data = Shapes3D(
			prior=args.data_distribution,
			rate=args.rate_data,
			k=args.data_k)
		num_channel = 3

	elif name.lower() == 'mpi3d':
		train_data = MPI3DReal(
			prior=args.data_distribution,
			rate=args.rate_data,
			k=args.data_k)
		num_channel = 3
				
	elif name.lower() == 'natural':
		train_data = NaturalSprites(natural_discrete=args.natural_discrete)
		num_channel = 3

	elif name.lower() == 'kittimasks':
		if args.evaluate:
			train_data = KittiMasks(max_delta_t=args.kitti_max_delta_t, transform=None)
		else:
			train_data = KittiMasks(max_delta_t=args.kitti_max_delta_t)
		num_channel = 1

	else:
		raise NotImplementedError

	return DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=True,
		collate_fn=custom_collate,
	), num_channel


def test_data(dset, plot=False):
	print(
		f'dataset {dset.data.shape}, min {np.min(dset.data)}, max {np.max(dset.data)} '
		f'type {type(dset.data)} {dset.data.dtype}, factors {dset.factor_sizes}')

	print('Laplace')
	dset.prior = 'laplace'
	dl = DataLoader(dset, shuffle=True, batch_size=32,
					collate_fn=custom_collate)
	for b, l in dl:
		print(b.shape, type(b), b.min().item(), b.max().item(), l.shape)
		if plot:
			plt.figure(figsize=(12, 12))
			for i in range(32):
				plt.subplot(8, 4, i + 1)
				if b.shape[1] == 1:
					plt.imshow(b[i, 0])
				elif b.shape[1] == 3:
					plt.imshow(np.transpose(b[i], (1, 2, 0)))
				plt.title(str(l[i]))
				plt.axis('off')
			plt.tight_layout()
			plt.show()
		break

	print('Uniform')
	dset.prior = 'uniform'
	dl = DataLoader(dset, shuffle=True, batch_size=32,
					collate_fn=custom_collate)
	for b, l in dl:
		print(b.shape, type(b), b.min().item(), b.max().item(), l.shape)
		if plot:
			plt.figure(figsize=(12, 12))
			for i in range(32):
				plt.subplot(8, 4, i + 1)
				if b.shape[1] == 1:
					plt.imshow(b[i, 0])
				elif b.shape[1] == 3:
					plt.imshow(np.transpose(b[i], (1, 2, 0)))
				plt.title(str(l[i]))
				plt.axis('off')
			plt.tight_layout()
			plt.show()
		break


if __name__ == '__main__':
	# cars
	print('cars3D')
	dset = Cars3D(prior='uniform', rate=1, k=-1)
	test_data(dset, False)

	# norb
	print('SmallNORB')
	dset = SmallNORB(prior='laplace', rate=1, k=-1)
	test_data(dset, False)

	# dsprites example
	print('DSprites')
	dset = SpriteDataset(prior='laplace', rate=1, k=-1)
	test_data(dset, False)

	# mpi real example
	print('MPI3dReal')
	dset = MPI3DReal(prior='laplace', rate=1, k=-1)
	test_data(dset, False)

	# shapes dataset
	print('Shapes3D... takes 5min')
	dset = Shapes3D(prior='laplace', rate=1, k=-1)
	test_data(dset, False)
