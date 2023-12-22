import logging
import numpy as np
import torch.utils.data

import itertools
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


import re

from .coco import CocoDataset


@DATASETS.register_module()
class CocoWithCaptionDataset(CocoDataset):
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 max_words=128,):
        
        ann_file = ann_file.split('?')
        
        super(CocoWithCaptionDataset, self).__init__(
            ann_file[0],
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
        )
        self.cap_ann_file = ann_file[1]
        self.max_words = max_words
        
        self.cap_data_infos = self.load_cap_annotations(self.cap_ann_file)
        
        
        
        # 
        # if not test_mode:
        #     valid_inds = self._cap_filter_imgs()
        #     self.cap_data_infos = [self.cap_data_infos[i] for i in valid_inds]
        #     # set group flag for the sampler
        #     self._cap_set_group_flag()

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        ann = self._parse_ann_info(self.data_infos[idx], ann_info)
        
        
        cap_ann_ids = self.coco_cap.get_ann_ids(img_ids=[img_id])
        cap_ann_info = self.coco_cap.load_anns(cap_ann_ids)
        caption = self._cap_parse_ann_info(cap_ann_info)
        ann['caption'] = caption
        return ann

    def _cap_parse_ann_info(self, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_cap = []
        
        for i, ann in enumerate(ann_info):
            gt_cap.append(self.pre_caption(ann['caption'], self.max_words))

        return gt_cap


    def pre_caption(self, caption, max_words=None):
        caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if max_words is not None and len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    # def _cap_filter_imgs(self, min_size=32):
    #     """Filter images too small."""
    #     if self.filter_empty_gt:
    #         warnings.warn(
    #             'CustomDataset does not support filtering empty gt images.')
    #     valid_inds = []
    #     for i, img_info in enumerate(self.cap_data_infos):
    #         if min(img_info['width'], img_info['height']) >= min_size:
    #             valid_inds.append(i)
    #     return valid_inds
    # 
    # def _cap_set_group_flag(self):
    #     """Set flag according to image aspect ratio.
    #     Images with aspect ratio greater than 1 will be set as group 1,
    #     otherwise group 0.
    #     """
    #     self.cap_flag = np.zeros(len(self), dtype=np.uint8)
    #     for i in range(len(self)):
    #         img_info = self.cap_data_infos[i]
    #         if img_info['width'] / img_info['height'] > 1:
    #             self.cap_flag[i] = 1
    
    def load_cap_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco_cap = COCO(self.cap_ann_file)
        self.img_ids = self.coco_cap.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco_cap.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco_cap.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    




# class EpochListening:
#     """Mixin for receiving updates whenever the epoch increments."""
# 
#     @property
#     def can_reuse_epoch_itr_across_epochs(self):
#         """
#         Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
#         this dataset across epochs.
#         This needs to return ``False`` if the sample sizes can change across
#         epochs, in which case we may need to regenerate batches at each epoch.
#         If your dataset relies in ``set_epoch`` then you should consider setting
#         this to ``False``.
#         """
#         return True
# 
#     def set_epoch(self, epoch):
#         """Will receive the updated epoch number at the beginning of the epoch."""
#         pass
# 
# 
# 
# class FairseqDataset(torch.utils.data.Dataset, EpochListening):
#     """A dataset that provides helpers for batching."""
# 
#     def __getitem__(self, index):
#         raise NotImplementedError
# 
#     def __len__(self):
#         raise NotImplementedError
# 
#     def collater(self, samples):
#         """Merge a list of samples to form a mini-batch.
#         Args:
#             samples (List[dict]): samples to collate
#         Returns:
#             dict: a mini-batch suitable for forwarding with a Model
#         """
#         raise NotImplementedError
# 
#     def num_tokens(self, index):
#         """Return the number of tokens in a sample. This value is used to
#         enforce ``--max-tokens`` during batching."""
#         raise NotImplementedError
# 
#     def num_tokens_vec(self, indices):
#         """Return the number of tokens for a set of positions defined by indices.
#         This value is used to enforce ``--max-tokens`` during batching."""
#         raise NotImplementedError
# 
#     def size(self, index):
#         """Return an example's size as a float or tuple. This value is used when
#         filtering a dataset with ``--max-positions``."""
#         raise NotImplementedError
# 
#     def ordered_indices(self):
#         """Return an ordered list of indices. Batches will be constructed based
#         on this order."""
#         return np.arange(len(self), dtype=np.int64)
# 
#     @property
#     def supports_prefetch(self):
#         """Whether this dataset supports prefetching."""
#         return False
# 
#     def attr(self, attr: str, index: int):
#         return getattr(self, attr, None)
# 
#     def prefetch(self, indices):
#         """Prefetch the data required for this epoch."""
#         raise NotImplementedError
# 
#     def get_batch_shapes(self):
#         """
#         Return a list of valid batch shapes, for example::
#             [(8, 512), (16, 256), (32, 128)]
#         The first dimension of each tuple is the batch size and can be ``None``
#         to automatically infer the max batch size based on ``--max-tokens``.
#         The second dimension of each tuple is the max supported length as given
#         by :func:`fairseq.data.FairseqDataset.num_tokens`.
#         This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
#         to restrict batch shapes. This is useful on TPUs to avoid too many
#         dynamic shapes (and recompilations).
#         """
#         return None
# 
#     def batch_by_size(
#         self,
#         indices,
#         max_tokens=None,
#         max_sentences=None,
#         required_batch_size_multiple=1,
#     ):
#         """
#         Given an ordered set of indices, return batches according to
#         *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
#         """
#         from fairseq.data import data_utils
# 
#         fixed_shapes = self.get_batch_shapes()
#         if fixed_shapes is not None:
# 
#             def adjust_bsz(bsz, num_tokens):
#                 if bsz is None:
#                     assert max_tokens is not None, "Must specify --max-tokens"
#                     bsz = max_tokens // num_tokens
#                 if max_sentences is not None:
#                     bsz = min(bsz, max_sentences)
#                 elif (
#                     bsz >= required_batch_size_multiple
#                     and bsz % required_batch_size_multiple != 0
#                 ):
#                     bsz -= bsz % required_batch_size_multiple
#                 return bsz
# 
#             fixed_shapes = np.array(
#                 [
#                     [adjust_bsz(bsz, num_tokens), num_tokens]
#                     for (bsz, num_tokens) in fixed_shapes
#                 ]
#             )
# 
#         try:
#             num_tokens_vec = self.num_tokens_vec(indices).astype('int64')
#         except NotImplementedError:
#             num_tokens_vec = None
# 
#         return data_utils.batch_by_size(
#             indices,
#             num_tokens_fn=self.num_tokens,
#             num_tokens_vec=num_tokens_vec,
#             max_tokens=max_tokens,
#             max_sentences=max_sentences,
#             required_batch_size_multiple=required_batch_size_multiple,
#             fixed_shapes=fixed_shapes,
#         )
# 
#     def filter_indices_by_size(self, indices, max_sizes):
#         """
#         Filter a list of sample indices. Remove those that are longer than
#         specified in *max_sizes*.
#         WARNING: don't update, override method in child classes
#         Args:
#             indices (np.array): original array of sample indices
#             max_sizes (int or list[int] or tuple[int]): max sample size,
#                 can be defined separately for src and tgt (then list or tuple)
#         Returns:
#             np.array: filtered sample array
#             list: list of removed indices
#         """
#         if isinstance(max_sizes, float) or isinstance(max_sizes, int):
#             if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
#                 ignored = indices[self.sizes[indices] > max_sizes].tolist()
#                 indices = indices[self.sizes[indices] <= max_sizes]
#             elif (
#                 hasattr(self, "sizes")
#                 and isinstance(self.sizes, list)
#                 and len(self.sizes) == 1
#             ):
#                 ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
#                 indices = indices[self.sizes[0][indices] <= max_sizes]
#             else:
#                 indices, ignored = data_utils._filter_by_size_dynamic(
#                     indices, self.size, max_sizes
#                 )
#         else:
#             indices, ignored = data_utils._filter_by_size_dynamic(
#                 indices, self.size, max_sizes
#             )
#         return indices, ignored
# 
#     @property
#     def supports_fetch_outside_dataloader(self):
#         """Whether this dataset supports fetching outside the workers of the dataloader."""
#         return True
# 
# 
# 
# class OFADataset(FairseqDataset):
#     def __init__(self, split, dataset, bpe, src_dict, tgt_dict):
#         self.split = split
#         self.dataset = dataset
#         self.bpe = bpe
#         self.src_dict = src_dict
#         self.tgt_dict = tgt_dict
# 
#         self.bos = src_dict.bos()
#         self.eos = src_dict.eos()
#         self.pad = src_dict.pad()
#         self.bos_item = torch.LongTensor([self.bos])
#         self.eos_item = torch.LongTensor([self.eos])
# 
#     def __len__(self):
#         return len(self.dataset)
# 
#     def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
#         s = self.tgt_dict.encode_line(
#             line=self.bpe.encode(text) if use_bpe else text,
#             add_if_not_exist=False,
#             append_eos=False
#         ).long()
#         if length is not None:
#             s = s[:length]
#         if append_bos:
#             s = torch.cat([self.bos_item, s])
#         if append_eos:
#             s = torch.cat([s, self.eos_item])
#         return s
# 
#     def pre_question(self, question, max_ques_words=None):
#         question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
# 
#         question = re.sub(
#             r"\s{2,}",
#             ' ',
#             question,
#         )
#         question = question.rstrip('\n')
#         question = question.strip(' ')
# 
#         # truncate question
#         question_words = question.split(' ')
#         if max_ques_words is not None and len(question_words) > max_ques_words:
#             question = ' '.join(question_words[:max_ques_words])
# 
#         return question
# 
#     def pre_caption(self, caption, max_words=None):
#         caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
# 
#         caption = re.sub(
#             r"\s{2,}",
#             ' ',
#             caption,
#         )
#         caption = caption.rstrip('\n')
#         caption = caption.strip(' ')
# 
#         # truncate caption
#         caption_words = caption.split(' ')
#         if max_words is not None and len(caption_words) > max_words:
#             caption = ' '.join(caption_words[:max_words])
# 
#         return caption
# 
# 
# def collate(samples, pad_idx, eos_idx):
#     if len(samples) == 0:
#         return {}
# 
#     def merge(key):
#         return data_utils.collate_tokens(
#             [s[key] for s in samples],
#             pad_idx,
#             eos_idx=eos_idx,
#         )
# 
#     id = np.array([s["id"] for s in samples])
#     src_tokens = merge("source")
#     src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
# 
#     patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
#     patch_masks = torch.cat([sample['patch_mask'] for sample in samples])
# 
#     prev_output_tokens = None
#     target = None
#     if samples[0].get("target", None) is not None:
#         target = merge("target")
#         tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
#         ntokens = tgt_lengths.sum().item()
# 
#         if samples[0].get("prev_output_tokens", None) is not None:
#             prev_output_tokens = merge("prev_output_tokens")
#     else:
#         ntokens = src_lengths.sum().item()
# 
#     batch = {
#         "id": id,
#         "nsentences": len(samples),
#         "ntokens": ntokens,
#         "net_input": {
#             "src_tokens": src_tokens,
#             "src_lengths": src_lengths,
#             "patch_images": patch_images,
#             "patch_masks": patch_masks,
#             "prev_output_tokens": prev_output_tokens
#         },
#         "target": target,
#     }
# 
#     return batch
# 
# class CaptionDataset(OFADataset):
#     def __init__(
#         self,
#         split,
#         dataset,
#         bpe,
#         src_dict,
#         tgt_dict=None,
#         max_src_length=128,
#         max_tgt_length=30,
#         patch_image_size=224,
#         imagenet_default_mean_and_std=False,
#         scst=False
#     ):
#         super().__init__(split, dataset, bpe, src_dict, tgt_dict)
#         self.max_src_length = max_src_length
#         self.max_tgt_length = max_tgt_length
#         self.patch_image_size = patch_image_size
#         self.scst = scst
# 
#         self.transtab = str.maketrans({key: None for key in string.punctuation})
# 
#         if imagenet_default_mean_and_std:
#             mean = IMAGENET_DEFAULT_MEAN
#             std = IMAGENET_DEFAULT_STD
#         else:
#             mean = [0.5, 0.5, 0.5]
#             std = [0.5, 0.5, 0.5]
# 
#         self.patch_resize_transform = transforms.Compose([
#             lambda image: image.convert("RGB"),
#             transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std),
#         ])
# 
#         if type(bpe).__name__ == 'GPT2BPE':
#             self.prompt = " what does the image describe?"
#         elif type(bpe).__name__ == 'BertBPE':
#             self.prompt = "图片描述了什么内容?"
# 
#     def __getitem__(self, index):
#         uniq_id, image, caption = self.dataset[index]
# 
#         image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
#         patch_image = self.patch_resize_transform(image)
#         patch_mask = torch.tensor([True])
# 
#         if self.split == 'train' and not self.scst:
#             caption = caption.translate(self.transtab).strip()
#             caption_token_list = caption.strip().split()
#             tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
#         else:
#             caption = ' '.join(caption.strip().split())
#             caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
#             tgt_caption = '&&'.join(caption_list)
#         src_item = self.encode_text(self.prompt)
#         tgt_item = self.encode_text(" {}".format(tgt_caption))
# 
#         src_item = torch.cat([self.bos_item, src_item, self.eos_item])
#         target_item = torch.cat([tgt_item, self.eos_item])
#         prev_output_item = torch.cat([self.bos_item, tgt_item])
# 
#         example = {
#             "id": uniq_id,
#             "source": src_item,
#             "patch_image": patch_image,
#             "patch_mask": patch_mask,
#             "target": target_item,
#             "prev_output_tokens": prev_output_item
#         }
#         return example
# 
#     def collater(self, samples, pad_to_length=None):
#         """Merge a list of samples to form a mini-batch.
#         Args:
#             samples (List[dict]): samples to collate
#         Returns:
#             dict: a mini-batch containing the data of the task
#         """
#         return collate(samples, pad_idx=self.pad, eos_idx=self.eos)