# Code adapted from https://github.com/orhir/STG-NF/blob/main/dataset.py
import json
import math
import os
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image



def get_ab_labels(global_data_np_ab, segs_meta_ab, path_to_vid_dir='', segs_root=''):
    pose_segs_root = segs_root
    clip_list = os.listdir(pose_segs_root)
    clip_list = sorted(
        fn.replace("alphapose_tracked_person.json", "annotations") for fn in clip_list if fn.endswith('.json'))
    labels = np.ones_like(global_data_np_ab)
    for clip in tqdm(clip_list):
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_annotations.*', clip)[0]
        if type == "normal":
            continue
        clip_id = type + "_" + clip_id
        clip_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                      (segs_meta_ab[:, 0] == scene_id))[0]
        clip_metadata = segs_meta_ab[clip_metadata_inds]
        clip_res_fn = os.path.join(path_to_vid_dir, "Scene{}".format(scene_id), clip)
        filelist = sorted(os.listdir(clip_res_fn))
        clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist]
        # FIX shape bug
        clip_shapes = set([clip_gt.shape for clip_gt in clip_gt_lst])
        min_width = min([clip_shape[0] for clip_shape in clip_shapes])
        min_height = min([clip_shape[1] for clip_shape in clip_shapes])
        clip_labels = np.array([clip_gt[:min_width, :min_height] for clip_gt in clip_gt_lst])
        gt_file = os.path.join("data/UBnormal/gt", clip.replace("annotations", "tracks.txt"))
        clip_gt = np.zeros_like(clip_labels)
        with open(gt_file) as f:
            abnormality = f.readlines()
            for ab in abnormality:
                i, start, end = ab.strip("\n").split(",")
                for t in range(int(float(start)), int(float(end))):
                    clip_gt[t][clip_labels[t] == int(float(i))] = 1
        for t in range(clip_gt.shape[0]):
            if (clip_gt[t] != 0).any():  # Has abnormal event
                ab_metadata_inds = np.where(clip_metadata[:, 3].astype(int) == t)[0]
                # seg = clip_segs[ab_metadata_inds][:, :2, 0]
                clip_fig_idxs = set([arr[2] for arr in segs_meta_ab[ab_metadata_inds]])
                for person_id in clip_fig_idxs:
                    person_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                                    (segs_meta_ab[:, 0] == scene_id) &
                                                    (segs_meta_ab[:, 2] == person_id) &
                                                    (segs_meta_ab[:, 3].astype(int) == t))[0]
                    data = np.floor(global_data_np_ab[person_metadata_inds].T).astype(int)
                    if data.shape[-1] != 0:
                        if clip_gt[t][
                            np.clip(data[:, 0, 1], 0, clip_gt.shape[1] - 1),
                            np.clip(data[:, 0, 0], 0, clip_gt.shape[2] - 1)
                        ].sum() > data.shape[0] / 2:
                            # This pose is abnormal
                            labels[person_metadata_inds] = -1
    return labels[:, 0, 0, 0]


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False,
                         global_pose_data=[], dataset="ShanghaiTech"):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = single_pose_dict2np(clip_dict, idx)
        if dataset == "UBnormal":
            key = ('{:02d}_{}_{:02d}'.format(int(scene_id), clip_id, int(idx)))
        else:
            key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np = split_pose_to_segments(sing_pose_np,
                                                                                            sing_pose_meta,
                                                                                            sing_pose_keys,
                                                                                            start_ofst, seg_stride,
                                                                                            seg_len,
                                                                                            scene_id=scene_id,
                                                                                            clip_id=clip_id,
                                                                                            single_score_np=sing_scores_np,
                                                                                            dataset=dataset)
        pose_segs_data.append(curr_pose_segs_np)
        score_segs_data.append(curr_pose_score_np)
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_pose_segs_meta
    if len(pose_segs_data) == 0:
        pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3)
        score_segs_data_np = np.empty(0).reshape(0, seg_len)
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    global_pose_data_np = np.concatenate(global_pose_data, axis=0)
    del pose_segs_data
    # del global_pose_data
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys, global_pose_data_np, global_pose_data, score_segs_data_np
    else:
        return pose_segs_data_np, pose_segs_meta, global_pose_data_np, global_pose_data, score_segs_data_np


def single_pose_dict2np(person_dict, idx):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    sing_scores_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(single_person[key]['scores'])
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    sing_scores_np = np.stack(sing_scores_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np


def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id='', single_score_np=None, dataset="ShanghaiTech"):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np = np.empty([0, seg_len])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(int)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            curr_score = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)
            if dataset == "UBnormal":
                pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
            else:
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    return pose_segs_np, pose_segs_meta, pose_score_np


def normalize_pose(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', False)
    # sub_mean = kwargs.get('sub_mean', True)
    # scale = kwargs.get('scale', False)
    # scale_proportional = kwargs.get('scale_proportional', True)

    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    pose_data_zero_mean = pose_data_centered
    # return pose_data_zero_mean

    pose_data_zero_mean[..., :2] = (pose_data_centered[..., :2] - pose_data_centered[..., :2].mean(axis=(1, 2))[:, None, None, :]) / pose_data_centered[..., 1].std(axis=(1, 2))[:, None, None, None]
    return pose_data_zero_mean



SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]

class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    def __init__(self, path_to_json_dir, path_to_vid_dir=None, normalize_pose_segs=True, return_indices=False,
                 return_metadata=False, debug=False, return_global=True, evaluate=False, abnormal_train_path=None,
                 **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.patches_db = None
        self.use_patches = False
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = dataset_args.get('headless', False)
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.debug = debug
        num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.return_global = return_global
        self.transform_list = dataset_args.get('trans_list', None)
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        self.segs_data_np, self.segs_meta, self.person_keys, self.global_data_np, \
        self.global_data, self.segs_score_np = \
            gen_dataset(path_to_json_dir, num_clips=num_clips, ret_keys=True,
                        ret_global_data=return_global, **dataset_args)
        self.segs_meta = np.array(self.segs_meta)
        if abnormal_train_path is not None:
            self.segs_data_np_ab, self.segs_meta_ab, self.person_keys_ab, self.global_data_np_ab, \
            self.global_data_ab, self.segs_score_np_ab = \
                gen_dataset(abnormal_train_path, num_clips=num_clips, ret_keys=True,
                            ret_global_data=return_global, **dataset_args)
            self.segs_meta_ab = np.array(self.segs_meta_ab)
            ab_labels = get_ab_labels(self.segs_data_np_ab, self.segs_meta_ab, path_to_vid_dir, abnormal_train_path)
            num_normal_samp = self.segs_data_np.shape[0]
            num_abnormal_samp = (ab_labels == -1).sum()
            total_num_normal_samp = num_normal_samp + (ab_labels == 1).sum()
            print("Num of abnormal sapmles: {}  | Num of normal samples: {}  |  Precent: {}".format(
                num_abnormal_samp, total_num_normal_samp, num_abnormal_samp / total_num_normal_samp))
            self.labels = np.concatenate((np.ones(num_normal_samp), ab_labels),
                                         axis=0).astype(int)
            self.segs_data_np = np.concatenate((self.segs_data_np, self.segs_data_np_ab), axis=0)
            self.segs_meta = np.concatenate((self.segs_meta, self.segs_meta_ab), axis=0)
            self.global_data_np = np.concatenate((self.global_data_np, self.global_data_np_ab), axis=0)
            self.segs_score_np = np.concatenate(
                (self.segs_score_np, self.segs_score_np_ab), axis=0)
            self.global_data += self.global_data_ab
            self.person_keys.update(self.person_keys_ab)
        else:
            self.labels = np.ones(self.segs_data_np.shape[0])
        # Convert person keys to ints
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations

        if self.normalize_pose_segs:
            data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
                                              **self.args).squeeze(axis=0).transpose(2, 0, 1)

        ret_arr = [data_transformed, trans_index]

        ret_arr += [self.segs_score_np[sample_index]]
        ret_arr += [self.labels[sample_index]]
        return ret_arr

    def get_all_data(self, normalize_pose_segs=True):
        if normalize_pose_segs:
            segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
                (0, 3, 1, 2))
        else:
            segs_data_np = self.segs_data_np
        if self.num_transform == 1 or self.eval:
            return list(segs_data_np)
        return segs_data_np

    def __len__(self):
        return self.num_transform * self.num_samples


def get_dataset_and_loader(args, trans_list, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'headless': args.headless, 'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale,
                    'seg_len': args.seg_len, 'return_indices': True, 'return_metadata': True, "dataset": args.dataset,
                    'train_seg_conf_th': args.train_seg_conf_th, 'specific_clip': args.specific_clip}
    dataset, loader = dict(), dict()
    splits = ['train', 'test'] if not only_test else ['test']
    for split in splits:
        evaluate = split == 'test'
        abnormal_train_path = args.pose_path_train_abnormal if split == 'train' else None
        normalize_pose_segs = args.global_pose_segs
        dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
        dataset_args['vid_path'] = args.vid_path[split]
        dataset[split] = PoseSegDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                        normalize_pose_segs=normalize_pose_segs,
                                        evaluate=evaluate,
                                        abnormal_train_path=abnormal_train_path,
                                        **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader


def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False


def gen_dataset(person_json_root, num_clips=None, kp18_format=True, ret_keys=False, ret_global_data=True,
                **dataset_args):
    segs_data_np = []
    segs_score_np = []
    segs_meta = []
    global_data = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'ShanghaiTech')

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
    if num_clips is not None:
        json_list = [json_list[num_clips]]  # For debugging purposes
    for person_dict_fn in tqdm(json_list):
        if dataset == "UBnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        else:
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            if shanghaitech_hr_skip(dataset=="ShaghaiTech-HR", scene_id, clip_id):
                continue
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = gen_clip_seg_data_np(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset=dataset)

        _, _, _, global_data_np, global_data, _ = gen_clip_seg_data_np(clip_dict, start_ofst, 1, 1, scene_id=scene_id,
                                                                       clip_id=clip_id,
                                                                       ret_keys=ret_keys,
                                                                       global_pose_data=global_data,
                                                                       dataset=dataset)
        segs_data_np.append(clip_segs_data_np)
        segs_score_np.append(score_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

    # Global data
    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)

    # if normalize_pose_segs:
    #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
    #     global_data_np = normalize_pose(global_data_np, vid_res=vid_res, **dataset_args)
    #     global_data = [normalize_pose(np.expand_dims(data, axis=0), **dataset_args).squeeze() for data
    #                    in global_data]
    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
        global_data_np = keypoints17_to_coco18(global_data_np)
        global_data = [keypoints17_to_coco18(data) for data in global_data]
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
        global_data_np = global_data_np[:, :, 5:]
        global_data = [data[:, 5:, :] for data in global_data]

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = \
            seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)
    if ret_global_data:
        if ret_keys:
            return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np
        else:
            return segs_data_np, segs_meta, global_data_np, global_data, segs_score_np
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_score_np
    else:
        return segs_data_np, segs_meta, segs_score_np


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=int)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np