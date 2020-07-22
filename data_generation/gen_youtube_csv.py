import csv
import os
from coco.coco import COCO
import coco.mask as maskUtils
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from tqdm import tqdm

def myRange(start,end,step):
    i = start
    while i < end:
        yield i
        i += step
    yield end

def main(args, data_dir):
    downscale = args.downscale
    keepaspectratio = args.keepaspect
    stride = args.stride
    annIds = coco.getAnnIds()
    anns = coco.loadAnns(annIds)
    max_len = np.max([len(x['segmentations']) for x in anns])
    file_name = '{}{}{}'.format('downscale_' if args.downscale else '',
                                     'keepaspect_' if args.keepaspect else '',
                                    'stride_{}_'.format(args.stride) if args.stride != 32 else '')
    with open('{}.csv'.format(file_name.rstrip('_')), mode='w') as movie_file:
        movie_writer = csv.writer(movie_file, delimiter=',')
        movie_writer.writerow(['id','cat_id'] + ['t_{}'.format(i) for i in range(max_len)])
        for ann in tqdm(anns) if args.verbose else anns:
            vals = []
            for seg in ann['segmentations']:
                if seg:
                    rle = maskUtils.frPyObjects([seg], ann['height'], ann['width'])
                    mask = np.squeeze(maskUtils.decode(rle))
                    if mask.any():
                        if downscale:
                            if keepaspectratio:
                                tr_mask = imresize(mask, (64,128))
                                tr_masks = []
                                window_idxes = list(myRange(0,64,stride))
                                for i in range(len(window_idxes)):
                                    tr_masks.append(tr_mask[:,window_idxes[i]:window_idxes[i]+64])
                            else:
                                tr_masks=[imresize(mask, (64,64))]
                        else:
                            tr_masks=[mask]
                        temp = []
                        for tr_mask in tr_masks:
                            if tr_mask.any():
                                com_val = np.array(ndimage.measurements.center_of_mass(tr_mask)).astype(np.float).tolist()
                                y_val = com_val[0]
                                x_val = com_val[1]
                                rle = maskUtils.encode(np.asfortranarray(tr_mask))
                                area_val = maskUtils.area(rle).astype(np.int)
                                temp.append((y_val,x_val,area_val))
                            else:
                                temp.append(None)
                        vals.append(temp)
                    else:
                        vals.append(None)
                else:
                    vals.append(None)
            movie_writer.writerow([ann['id'],ann['category_id']] + vals)
                

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--downscale', action='store_true')
    parser.add_argument('--keepaspect', action='store_true')
    args = parser.parse_args()
    #download data from https://competitions.codalab.org/competitions/20127#participate-get-data
    data_dir = './data/youtube_voc'
    annFile= os.path.join(data_dir, 'train.json')
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    main(args, data_dir)