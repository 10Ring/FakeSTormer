#-*- coding: utf-8 -*-
import os
import sys
import random
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import albumentations as alb
import cv2
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from skimage import transform as sktransform

from package_utils.deepfake_mask import dynamic_blend, random_get_hull
from package_utils.transform import randaffine
from package_utils.image_utils import load_image
from package_utils.bi_online_generation import random_erode_dilate, blendImages
from ..pipelines.geo_transform import get_source_transforms
from ..common import ParameterStore


def gen_SBI(img, landmark, **kwargs):
    '''
    This function is adapted to process SBI generation for both image-level and video-level
    '''
    index = kwargs.get('index')
    assert index is not None
    debug = kwargs.get('debug') or False

    param_store_ins = ParameterStore.get_instance()

    use_lms68 = param_store_ins.get_parameters('use_lms68') or False
    data_type = param_store_ins.get_parameters('data_type')
    assert data_type is not None

    if (data_type=='image' and np.random.rand()<0.25) or \
        (data_type=='video' and index==0 and np.random.rand()<0.25) or use_lms68:
        landmark = landmark[:68]

        if not param_store_ins.has_key('use_lms68') and data_type=='video':
            param_store_ins.add_parameters('use_lms68', True)
    
    # Getting ConvexHull
    mask, hull_type = random_get_hull(landmark, img, hull_type=param_store_ins.get_parameters('hull_type'))
    if index==0 and data_type=='video':
        param_store_ins.add_parameters('hull_type', hull_type)

    # For debugging
    if index is not None and debug:
        # Image.fromarray(img).save(f'samples/debugs/BG_{index}.jpg')
        Image.fromarray((mask*255).astype(np.uint8)).save(f'samples/debugs/ConvexHull_{index}.jpg')

    source = img.copy()
    rand_value = param_store_ins.get_parameters('rand_value') or np.random.rand()
    if index==0 and data_type=='video':
        param_store_ins.add_parameters('rand_value', rand_value)

    if rand_value < 0.5:
        if data_type == 'video':
            if index == 0:
                transform = get_source_transforms(data_type=data_type)
                data = transform(image=source.astype(np.uint8))
                source = data['image']
                replay_params = data['replay']
                param_store_ins.add_parameters('s_replay_params', replay_params)
            else:
                replay_params = param_store_ins.get_parameters('s_replay_params')
                data = alb.ReplayCompose.replay(replay_params, image=source.astype(np.uint8))
                source = data['image']
        else:
            source = get_source_transforms()(image=source.astype(np.uint8))['image']
    else:
        if data_type == 'video':
            if index == 0:
                transform = get_source_transforms(data_type=data_type)
                data = transform(image=img.astype(np.uint8))
                img = data['image']
                replay_params = data['replay']
                param_store_ins.add_parameters('i_replay_params', replay_params)
            else:
                replay_params = param_store_ins.get_parameters('i_replay_params')
                data = alb.ReplayCompose.replay(replay_params, image=img.astype(np.uint8))
                img = data['image']
        else:
            img = get_source_transforms()(image=img.astype(np.uint8))['image']

    # if index is not None and debug:
    #     Image.fromarray(source.astype(np.uint8)).save(f'samples/debugs/FG_{index}.jpg')

    if data_type == 'image':
        source, mask, _ = randaffine(source, mask[:,:,0])
    else:
        # if use_lms68:
        #     seq_distortion = kwargs.get('distortion')
        #     img_h, img_w, img_c = mask.shape
        #     aug_size = param_store_ins.get_parameters('aug_size') or random.randint(int(img_h*0.8), int(img_h/0.8))
        #     mask = sktransform.resize(mask,(aug_size,aug_size),preserve_range=True) # resize mask before deformation
        #     mask = seq_distortion.augment_image(mask)
        #     mask, ksize, rand_erode = random_erode_dilate(mask,
        #                                                   ksize=param_store_ins.get_parameters('ksize'),
        #                                                   rand_erode=param_store_ins.get_parameters('rand_erode')) # mask of shape (H,W,3)
        #     mask = sktransform.resize(mask,(img_h,img_w),preserve_range=True) # getting back mask
        #     mask = mask[:,:,0]

        #     # filte empty mask after deformation
        #     if np.sum(mask) == 0 :
        #         raise ValueError('Deformed mask has no facial region for blending!!!')

        #     if not param_store_ins.has_key('ksize'):
        #         param_store_ins.add_parameters('ksize', ksize)
        #     if not param_store_ins.has_key('rand_erode'):
        #         param_store_ins.add_parameters('rand_erode', rand_erode)
        #     if not param_store_ins.has_key('aug_size'):
        #         param_store_ins.add_parameters('aug_size', aug_size)
        # else:
        if index == 0:
            source, mask, fg_replay_params = randaffine(source, 
                                                        mask[:,:,0], 
                                                        index=index, 
                                                        data_type=data_type)
            param_store_ins.add_parameters('f_replay_params', fg_replay_params['f_replay_params'])
            param_store_ins.add_parameters('g_replay_params', fg_replay_params['g_replay_params'])
        else:
            f_replay_params = param_store_ins.get_parameters('f_replay_params')
            g_replay_params = param_store_ins.get_parameters('g_replay_params')
            source, mask, _ = randaffine(source, 
                                        mask[:,:,0], 
                                        index=index, 
                                        data_type=data_type, 
                                        f_replay_params=f_replay_params, 
                                        g_replay_params=g_replay_params) # mask of shape (H, W)

    # Getting Deformed ConvexHull
    if index is not None and debug:
        Image.fromarray((mask*255).astype(np.uint8)).save(f'samples/debugs/Deformed_ConvexHull_{index}.jpg')

    if data_type == 'image':
        img_blended, mask, _ = dynamic_blend(source, img, mask)
    else:
        # use_BI = param_store_ins.get_parameters('use_BI') or np.random.rand() > 0.5
        # if not param_store_ins.has_key('use_BI'):
        #     param_store_ins.add_parameters('use_BI', use_BI)
        # if use_lms68:
        #     if index == 0:
        #         img_blended, mask, blending_params = blendImages(source, 
        #                                                         img, 
        #                                                         mask*255)
        #         param_store_ins.add_parameters('blending_params', blending_params)
        #     else:
        #         img_blended, mask, _ = blendImages(source, 
        #                                            img, 
        #                                            mask*255, 
        #                                            **param_store_ins.get_parameters('blending_params'))
        #     mask = mask[:,:,0:1]
        # else:
        if index == 0:
            img_blended, mask, blending_params = dynamic_blend(source, img, mask)
            param_store_ins.add_parameters('blending_params', blending_params)
        else:
            blending_params = param_store_ins.get_parameters('blending_params')
            img_blended, mask, _ = dynamic_blend(source, img, mask, **blending_params)
    img_blended = img_blended.astype(np.uint8)
    img = img.astype(np.uint8)

    return img, img_blended, mask


def gen_target(background_face, background_landmark, margin=[20, 20], **kwargs):
    index = kwargs.get('index')
    assert index is not None

    if isinstance(background_face, str):
        background_face = load_image(background_face)

    background_face, face_img, mask_f = gen_SBI(background_face, background_landmark, **kwargs)
    mask_f = (1 - mask_f) * mask_f * 4
    mask_r = np.zeros((mask_f.shape[0], mask_f.shape[1], 1))
    
    margin_x, margin_y = margin
    H, W = len(face_img), len(face_img[0])
    face_img = face_img[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    background_face = background_face[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    
    mask_f = mask_f[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    mask_r = mask_r[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    
    mask_f, mask_r = np.repeat(mask_f,3,2), np.repeat(mask_r,3,2)
    mask_f, mask_r = (mask_f*255).astype(np.uint8), (mask_r*255).astype(np.uint8)

    # lower_bound = [0.5,0.75,1,1]
    # fake_intensity = np.random.uniform(lower_bound[np.random.randint(len(lower_bound))], 1.)
    fake_intensity = np.random.uniform(0.5, 1.)
    return face_img, mask_f, background_face, mask_r, fake_intensity


def reorder_landmark(landmark):
    landmark_add=np.zeros((13,2))
    for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx]=landmark[idx_l]
    landmark[68:]=landmark_add
    return landmark


def sbi_hflip(img, mask=None, landmark=None, bbox=None):
    H, W = img.shape[:2]
    if landmark is not None:
        landmark = landmark.copy()

    if bbox is not None:
        bbox = bbox.copy()

    if landmark is not None:
        landmark_new=np.zeros_like(landmark)

        landmark_new[:17]=landmark[:17][::-1]
        landmark_new[17:27]=landmark[17:27][::-1]

        landmark_new[27:31]=landmark[27:31]
        landmark_new[31:36]=landmark[31:36][::-1]

        landmark_new[36:40]=landmark[42:46][::-1]
        landmark_new[40:42]=landmark[46:48][::-1]

        landmark_new[42:46]=landmark[36:40][::-1]
        landmark_new[46:48]=landmark[40:42][::-1]

        landmark_new[48:55]=landmark[48:55][::-1]
        landmark_new[55:60]=landmark[55:60][::-1]

        landmark_new[60:65]=landmark[60:65][::-1]
        landmark_new[65:68]=landmark[65:68][::-1]
        if len(landmark)==68:
            pass
        elif len(landmark)==81:
            landmark_new[68:81]=landmark[68:81][::-1]
        else:
            raise NotImplementedError
        landmark_new[:,0]=W-landmark_new[:,0]
    else:
        landmark_new=None

    if bbox is not None:
        bbox_new=np.zeros_like(bbox)
        bbox_new[0,0]=bbox[1,0]
        bbox_new[1,0]=bbox[0,0]
        bbox_new[:,0]=W-bbox_new[:,0]
        bbox_new[:,1]=bbox[:,1].copy()
        if len(bbox)>2:
            bbox_new[2,0]=W-bbox[3,0]
            bbox_new[2,1]=bbox[3,1]
            bbox_new[3,0]=W-bbox[2,0]
            bbox_new[3,1]=bbox[2,1]
            bbox_new[4,0]=W-bbox[4,0]
            bbox_new[4,1]=bbox[4,1]
            bbox_new[5,0]=W-bbox[6,0]
            bbox_new[5,1]=bbox[6,1]
            bbox_new[6,0]=W-bbox[5,0]
            bbox_new[6,1]=bbox[5,1]
    else:
        bbox_new=None

    if mask is not None:
        mask=mask[:,::-1]
    else:
        mask=None
    img=img[:,::-1].copy()
    return img,mask,landmark_new,bbox_new


def BI_postprocessing(img, face_img, mask):
    param_store_ins = ParameterStore.get_instance()
    face_img = Image.fromarray(face_img)
    img = Image.fromarray(img)

    # randomly downsample after BI pipeline
    rand_val_post = param_store_ins.get_parameters('rand_val_post') or random.randint(0,1)
    if rand_val_post:
        aug_size = param_store_ins.get_parameters('post_aug_size') or random.randint(64, 317)
        rand_resize =  param_store_ins.get_parameters('rand_resize') or random.randint(0,1)

        if rand_resize:
            face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            img = img.resize((aug_size, aug_size), Image.BILINEAR)
        else:
            face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
            img = img.resize((aug_size, aug_size), Image.NEAREST)

        if not param_store_ins.has_key('post_aug_size'):
            param_store_ins.add_parameters('post_aug_size', aug_size)

        if not param_store_ins.has_key('rand_resize'):
            param_store_ins.add_parameters('rand_resize', rand_resize)

    if not param_store_ins.has_key('rand_val_post'):
        param_store_ins.add_parameters('rand_val_post', rand_val_post)

    face_img = face_img.resize((317, 317),Image.BILINEAR)
    img = img.resize((317, 317),Image.BILINEAR)
    face_img = np.array(face_img)
    img = np.array(img)

    soft_margin = param_store_ins.get_parameters('soft_margin') or np.random.randint(-30, 30)
    if not param_store_ins.has_key('soft_margin'):
        param_store_ins.add_parameters('soft_margin', soft_margin)

    face_img = face_img[30+soft_margin:(287+soft_margin), 30+soft_margin:(287+soft_margin), :]
    img = img[30+soft_margin:(287+soft_margin), 30+soft_margin:(287+soft_margin), :]
    mask = mask[30+soft_margin:(287+soft_margin), 30+soft_margin:(287+soft_margin), :]

    return img, face_img, mask
