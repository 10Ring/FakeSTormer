#-*- coding: utf-8 -*-
import os
import random

import numpy as np


def _extract_data_based_dist(data_type: str, image_paths: list, labels: list, dist: list, **params):
    def sampling_frames(image_paths: list, labels: list, dist: list, **params):
        '''
        Extracting image paths and labels based on given distribution
        '''
        total_f = sum(labels)
        total_r = len(labels) - total_f

        assert total_f > 0, "Number of fake images must be greater than 0 for distribution sampling!"
        assert total_r > 0, "Number of real images must be greater than 0 for distribution sampling!"

        r_dist, f_dist = dist[0], dist[1]

        idxes = sorted(range(0, len(labels)), key=lambda k: labels[k])
        r_idxes = idxes[:total_r]
        f_idxes = idxes[total_r:]

        print(f'Original Number of Fake images --- {len(f_idxes)}')
        print(f'Original Number of Real images --- {len(r_idxes)}')

        if int((total_f/f_dist)*r_dist) > total_r:
            total_f = int((total_r/r_dist)*f_dist)
            f_idxes = random.sample(f_idxes, total_f)
        else:
            total_r = int((total_f/f_dist)*r_dist)
            r_idxes = random.sample(r_idxes, total_r)
        
        print(f'Number of Fake images --- {len(f_idxes)} given Fake distribution --- {f_dist}')
        print(f'Number of Real images --- {len(r_idxes)} given Real distribution --- {r_dist}')

        new_idxes = r_idxes + f_idxes

        image_paths = [image_paths[i] for i in new_idxes]
        labels = np.array(labels)[new_idxes]

        for k, v in params.items():
            if v is not None and len(v):
                params[k] = [v[i] for i in new_idxes]

        return image_paths, labels, params
    
    def sampling_videos(image_paths: list, labels: list, dist: list, **params):
        '''
        Extracting image paths and labels based on given distribution for video data
        '''
        f_vid_ids = []
        r_vid_ids = []
        for ip in image_paths:
            faketype = ip.split('/')[8]
            vid_id = os.path.dirname(ip)
            if faketype == 'real_videos' or 'real' in faketype or 'original' in faketype:
                r_vid_ids.append(vid_id)
            else:
                f_vid_ids.append(vid_id)

        f_vid_ids = list(set(f_vid_ids))
        r_vid_ids = list(set(r_vid_ids))
        total_f = len(f_vid_ids)
        total_r = len(r_vid_ids)

        assert total_f > 0, "Number of fake videos must be greater than 0 for distribution sampling!"
        assert total_r > 0, "Number of real videos must be greater than 0 for distribution sampling!"
        print(f'Original Number of Fake videos --- {total_f}')
        print(f'Original Number of Real videos --- {total_r}')

        r_dist, f_dist = dist[0], dist[1]

        if int((total_f/f_dist)*r_dist) > total_r:
            total_f = int((total_r/r_dist)*f_dist)
            f_vid_ids = random.sample(f_vid_ids, total_f)
        else:
            total_r = int((total_f/f_dist)*r_dist)
            r_vid_ids = random.sample(r_vid_ids, total_r)

        print(f'Number of Fake videos --- {len(f_vid_ids)} given Fake distribution --- {f_dist}')
        print(f'Number of Real videos --- {len(r_vid_ids)} given Real distribution --- {r_dist}')

        vid_ids = r_vid_ids + f_vid_ids
        new_idxes = []

        for i in range(len(labels)):
            ip = image_paths[i]
            vid_id = '/'.join([ip.split('/')[-3], ip.split('/')[-2]])
            if vid_id in vid_ids:
                new_idxes.append(i)

        image_paths = [image_paths[i] for i in new_idxes]
        labels = np.array(labels)[new_idxes]

        for k, v in params.items():
            if v is not None and len(v):
                params[k] = [v[i] for i in new_idxes]

        return image_paths, labels, params

    if data_type == 'image':
        return sampling_frames(image_paths, labels, dist, **params)
    else:
        return sampling_videos(image_paths, labels, dist, **params)
