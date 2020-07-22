import glob
import numpy as np
import os
import pickle
import scipy.ndimage as ndi 
import torch
import PIL.Image as Image
from scipy.ndimage.measurements import center_of_mass

# kitti mots
def get_data(path='./data/kitti/instances/', 
             folder=range(0, 21),
             n_hor_windows=6,
             img_size=64):
    all_imgs = []
    # get each sequence
    for j in folder:  # 21 sequences
        img_folder = f'{j:04d}'
        files = sorted(glob.glob(path + f'{img_folder}/*.png'))
        imgs = np.zeros((len(files), n_hor_windows, img_size, img_size), dtype=np.uint32)
        # load sequence_i
        for i, f in enumerate(files):
            img = np.array(Image.open(f))
            if i == 0:
                print('folder', img_folder, 'first image shape ', img.shape)
            shape = img.shape
            x_size = int(shape[1] // (shape[0] / img_size))
            img_t = torch.tensor(img.astype(np.float32))
            img = torch.nn.functional.interpolate(img_t[None, None], size=(img_size, x_size)).numpy().astype(np.uint32)[0, 0]
            # tile into windows
            hor_stride = (x_size - img_size) // (n_hor_windows - 1)
            for k in range(n_hor_windows):
                offset = k*hor_stride
                imgs[i, k, :, :] = img[:, offset:offset+img_size]
        all_imgs.append(imgs)
    return all_imgs

def get_individual_sequences(all_imgs, n_hor_windows=6, mask_threshold=30):
    sequences = []
    for f_id, imgs_windows in enumerate(all_imgs):  # all videos
        print('sequence orig', f_id)
        for window_i in range(n_hor_windows):   # one window
            imgs = imgs_windows[:, window_i]
            ids = np.where(np.bincount(imgs.ravel()) != 0)[0][1:-1]  # first is bg, last is non-category
            for id_i in ids:   # indiv
                if id_i // 1000 == 1:
                    continue
                imgs_id_i = np.zeros(imgs.shape, dtype=np.bool)
                imgs_id_i[imgs == id_i] = 1
                t_inds = np.where(imgs_id_i != 0)[0]
                t_inds = np.arange(t_inds[0], t_inds[-1]+1) # make dense
                sequence_id_i = []
                for t_ind in t_inds:  # augean stables
                    frame = imgs_id_i[t_ind]
                    if np.sum(frame) < mask_threshold:   # mask too small 
                        if len(sequence_id_i) > 2:     # min sequ len 2
                            sequences.append(np.stack(sequence_id_i))  # add to sequences
                        sequence_id_i = [] # hole in sequence, start new one
                        continue
                    else:
                        sequence_id_i.append(frame)  # add to sequence
                if len(sequence_id_i) > 1:
                    sequences.append(np.stack(sequence_id_i))  # add to sequences
    return sequences

# get center of mass, and area
def get_latents(sequence):
    all_latents = []
    for seq in sequence:
        latents = np.zeros((len(seq), 3), dtype=np.float32)
        for i, img in enumerate(seq):
            com = center_of_mass(img)
            latents[i] = np.array([com[0], com[1], np.sum(img)])  # y pos, x pos, area
        all_latents.append(latents)
    return all_latents

def main(args):
    # raw data from https://www.vision.rwth-aachen.de/page/mots
    all_imgs_c = get_data(path='./data/kitti/instances/', folder=range(0, 21),
                         n_hor_windows=args.n_hor_windows, img_size=args.img_size)  # mostly cars
    all_imgs_p = get_data(path='./data/kitti/mots/instances/', folder=[2, 5, 9, 11],
                         n_hor_windows=args.n_hor_windows, img_size=args.img_size)  # pedestrians
    print('number folders mostly cars', len(all_imgs_c))
    print('number folders pedestrians', len(all_imgs_p))
    sequences_p = get_individual_sequences(all_imgs_c, n_hor_windows=args.n_hor_windows)
    print()
    print('pedestrians')
    sequences_c = get_individual_sequences(all_imgs_p, n_hor_windows=args.n_hor_windows)
    all_sequences = sequences_p + sequences_c
    all_latents = get_latents(all_sequences)
    # save data
    with open(os.path.join('./data/kitti_peds_v2.pickle'), 'wb') as f:   # v0, v1 are only internal and were not released
        pickle.dump({'pedestrians':all_sequences, 'pedestrians_latents': all_latents}, f)
    # this is to do the data analysis
    dd = {}
    dd['id'] = [] 
    dd['category_id'] = []
    dd['category'] = [] 
    dd['x'] = []
    dd['x_diff'] = []
    dd['y'] = []
    dd['y_diff'] = []
    dd['area'] = []
    dd['area_diff'] = []
    dd['masks'] = []
    rotate = args.rotate
    for id_i, (seq, lat) in enumerate(zip(all_sequences, all_latents)):
        for (start_img, next_img), (start_latent, next_latent) in zip(zip(seq[:-1], seq[1:]), zip(lat[:-1], lat[1:])):
            if rotate:
                start_img  = ndi.rotate(start_img, 45)
                next_img  = ndi.rotate(next_img, 45)
                start_latent, next_latent

                start_com = center_of_mass(start_img)
                next_com = center_of_mass(next_img)
                start_latent = [start_com[0], start_com[1], np.sum(start_img)]  # y pos, x pos, area
                next_latent = [next_com[0], next_com[1], np.sum(next_img)]  # y pos, x pos, area

            dd['id'].append(id_i)
            dd['category_id'].append(1)
            dd['category'].append('pedestrian')
            dd['x'].append([start_latent[1], next_latent[1]])
            dd['x_diff'].append(next_latent[1] - start_latent[1])

            dd['y'].append([start_latent[0], next_latent[0]])
            dd['y_diff'].append(next_latent[0] - start_latent[0])

            dd['area'].append([np.sum(start_img), np.sum(next_img)])
            dd['area_diff'].append(np.sum(next_img) - np.sum(start_img))

            dd['masks'].append([start_img.astype(np.uint8), next_img.astype(np.uint8)])
    prefix = ''
    if rotate:
        prefix += '_rotate'
    with open(f'./data/kitti_dict_p_v2{prefix}.pkl', 'wb') as f:
        pickle.dump(dd, f)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--n-hor-windows', type=int, default=6)
    parser.add_argument('--rotate', action='store_true')
    args = parser.parse_args()
    main(args)


