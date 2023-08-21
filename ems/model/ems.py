from importlib.metadata import distribution
from typing import List, Optional

import torch

from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from ems.model.utils.tools import remove_padding
import torch.nn as nn

from ems.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
from ems.model.base import BaseModel
from torch.distributions.distribution import Distribution
from ems.data.tools import collate_tensor_with_padding,mask_to_lengths
import torch.nn.functional as F


class EMS(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 temporal_window: int = 3,
                 text_emb_size: int = 768,
                 if_humor: bool = False,
                 if_weighted: bool = False,
                 if_contrast: bool = False,
                 **kwargs):
        super().__init__()
        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder, nfeats=nfeats)
        self.if_humor = if_humor
        if self.if_humor:
            self.humor_encoder = instantiate(motionencoder, nfeats=nfeats)
            self.humor_decoder = instantiate(motiondecoder, nfeats=nfeats)
        self.feat_cache = {}
        self.mse = torch.nn.MSELoss()

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct

        self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)
        self.optimizer = instantiate(optim, params=self.parameters())

        self._losses = torch.nn.ModuleDict({split: instantiate(losses, vae=vae,
                                                               _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.temporal_window = temporal_window
        self.attn = torch.nn.Linear(nfeats,64)
        self.temporal_pool = nn.AdaptiveAvgPool1d(temporal_window)
        # self.std_attn = torch.nn.Linear(latent_dim*2,latent_dim)

        self.metrics = ComputeMetrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.prev_thresh = 0.3
        self.if_weighted = if_weighted
        self.if_contrast = if_contrast
        
        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict) -> List[Tensor]:
        if "cur_act" not in batch:
            text_mu,text_var,_,_ = self.textencoder(batch["text"])
            text_std = text_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(text_mu, text_std)
            latent_vector = self.sample_from_distribution(distribution)
            features = self.motiondecoder(latent_vector, batch["length"])
            return features
        if "prev_act" in batch and "next_act" in batch:
            connect_features = torch.cat([batch["prev_act"],batch["cur_act"],batch["next_act"]],dim=1)
            connect_lengths = [batch["cur_act"].size(1)+batch["prev_act"].size(1)+batch["next_act"].size(1)]
        elif "prev_act" in batch:
            connect_features = torch.cat([batch["prev_act"],batch["cur_act"]],dim=1)
            connect_lengths = [batch["cur_act"].size(1)+batch["prev_act"].size(1)]
        elif "next_act" in batch:
            connect_features = torch.cat([batch["cur_act"],batch["next_act"]],dim=1)
            connect_lengths = [batch["cur_act"].size(1)+batch["next_act"].size(1)]   
        connect_mu,connect_var = self.motionencoder(connect_features, connect_lengths)
        connect_std = connect_var.exp().pow(0.5)
        connect_distribution = torch.distributions.Normal(connect_mu, connect_std)
        connect_latent_vector = self.sample_from_distribution(connect_distribution)
        connect_features = self.motiondecoder(connect_latent_vector, connect_lengths)
        return connect_features
    
    def eval_forward(self, batch: dict) -> List[Tensor]:
        text_mu,text_var,_,_ = self.textencoder(batch["text"])
        text_std = text_var.exp().pow(0.5)
        distribution = torch.distributions.Normal(text_mu, text_std)
        latent_vector = self.sample_from_distribution(distribution)
        features = self.motiondecoder(latent_vector, batch["length"])
        
        prev_features = None
        if batch["prev_text"][0] != -1:
            prev_mu, prev_var, _, _ = self.textencoder(batch["prev_text"])
            prev_std = prev_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(prev_mu, prev_std)
            latent_vector = self.sample_from_distribution(distribution)
            prev_features = self.motiondecoder(latent_vector, batch["prev_length"])
            if prev_features.size(1) > self.temporal_window:
                prev_features = prev_features[:,-self.temporal_window:]
            else:
                prev_features = self.temporal_pool(prev_features.permute(0,2,1)).permute(0,2,1)
        
        next_features = None
        if batch["next_text"][0] != -1:
            next_mu, next_var, _, _ = self.textencoder(batch["next_text"])
            next_std = next_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(next_mu, next_std)
            latent_vector = self.sample_from_distribution(distribution)
            next_features = self.motiondecoder(latent_vector, batch["next_length"])
            if next_features.size(1) > self.temporal_window:
                next_features = next_features[:,:self.temporal_window]
            else:
                next_features = self.temporal_pool(next_features.permute(0,2,1)).permute(0,2,1)
        
        if prev_features is not None and next_features is not None:
            connect_features = torch.cat([prev_features,features,next_features],dim=1)
            connect_lengths = [features.size(1)+prev_features.size(1)+next_features.size(1)]
        elif prev_features is not None:
            connect_features = torch.cat([prev_features,features],dim=1)
            connect_lengths = [features.size(1)+prev_features.size(1)]
        elif next_features is not None:
            connect_features = torch.cat([features,next_features],dim=1)
            connect_lengths = [features.size(1)+next_features.size(1)] 
        else:
            connect_features = torch.cat([features],dim=1)
            connect_lengths = [features.size(1)]
        
        connect_mu,connect_var = self.motionencoder(connect_features, connect_lengths)
        connect_std = connect_var.exp().pow(0.5)
        connect_distribution = torch.distributions.Normal(connect_mu, connect_std)
        connect_latent_vector = self.sample_from_distribution(connect_distribution)
        connect_features = self.motiondecoder(connect_latent_vector, connect_lengths)
        return features, connect_features

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def contrast_forward(self, contrast_text_sentences, contrast_datastruct, contrast_lengths):
        text_mu,text_var,text_encoded,text_mask = self.textencoder(contrast_text_sentences)
        text_std = text_var.exp().pow(0.5)
        text_distribution = torch.distributions.Normal(text_mu, text_std)
        text_latent_vector = self.sample_from_distribution(text_distribution)
        
        contrast_datastruct.transforms = self.transforms
        motion_mu,motion_var = self.motionencoder(contrast_datastruct.features, contrast_lengths)
        motion_std = motion_var.exp().pow(0.5)
        motion_distribution = torch.distributions.Normal(motion_mu, motion_std)
        motion_latent_vector = self.sample_from_distribution(motion_distribution)
        return text_distribution,text_latent_vector,motion_distribution,motion_latent_vector
    
    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], 
                                *,return_latent: bool = False):
        # Encode the text to the latent space
        recons_text_encoded = None
        text_encoded = None
        text_lengths = None
        
        if self.hparams.vae:  
            text_mu,text_var,text_encoded,text_mask = self.textencoder(text_sentences)
            text_std = text_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(text_mu, text_std)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            raise NotImplementedError("Currently only support vae")
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        return datastruct, latent_vector, distribution, text_encoded, text_lengths, recons_text_encoded

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 text_lengths = None,
                                 return_latent: bool = False
                                 ):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms
        motion_recons_text_encoded = None
        # Encode the motion to the latent space
        if self.hparams.vae:
            motion_mu,motion_var = self.motionencoder(datastruct.features, lengths)
            motion_std = motion_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(motion_mu, motion_std)
            # distribution = self.motionencoder(datastruct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            raise NotImplementedError("Currently only support vae")
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        motion_datastruct = self.Datastruct(features=features)

        if not return_latent:
            return motion_datastruct
        return motion_datastruct, latent_vector, distribution, motion_recons_text_encoded

    def connect_to_connect_forward(self, prev_features, datastruct, next_features, prev_ids, next_ids, lengths, return_latent = True):
        cur_features = datastruct.features
        connect_features  = []
        connect_lengths = []
        for idx in range(len(datastruct)):
            cur_feature = cur_features[idx][:lengths[idx]]
            cur_length = lengths[idx]
            feature_lst = [cur_feature]
            if prev_ids[idx] != -1:
                feature_lst.insert(0,prev_features[idx])
                cur_length += self.temporal_window
            if next_ids[idx] != -1:
                cur_length += self.temporal_window
                feature_lst.append(next_features[idx])
            connect_feature = torch.cat(feature_lst,dim=0)
            # assert cur_length == connect_feature.size(0)
            connect_features.append(connect_feature)
            connect_lengths.append(cur_length)
        connect_features = collate_tensor_with_padding(connect_features)
        # encode the connect features to motion encoder-decoder
        if self.hparams.vae:
            connect_mu,connect_var = self.motionencoder(connect_features, connect_lengths)
            connect_std = connect_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(connect_mu, connect_std)
            # distribution = self.motionencoder(datastruct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            raise NotImplementedError("Currently only support vae")
        # print(connect_lengths)
        features = self.motiondecoder(latent_vector, connect_lengths)
        connect_datastruct = self.Datastruct(features=features)

        if not return_latent:
            return connect_datastruct
        return connect_datastruct, latent_vector, distribution, connect_lengths
    
    def humor_forward(self, datastruct,
                            lengths: Optional[List[int]] = None,
                            connect_lengths: Optional[List[int]] = None,
                            return_latent: bool = False):
        datastruct.transforms = self.transforms
        # Encode the motion to the latent space
        if self.hparams.vae:
            humor_mu,humor_var = self.humor_encoder(datastruct.features, lengths)
            humor_std = humor_var.exp().pow(0.5)
            distribution = torch.distributions.Normal(humor_mu, humor_std)
            # distribution = self.motionencoder(datastruct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            raise NotImplementedError("Currently only support vae")
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        # Decode the latent vector to a motion
        features = self.humor_decoder(latent_vector, connect_lengths)
        humor_datastruct = self.Datastruct(features=features)

        if not return_latent:
            return humor_datastruct
        return humor_datastruct, latent_vector, distribution
    
    # def update_feat_cache(self,ids,text_datastruct,gt_datastruct,lengths):
    #     pred_features = text_datastruct.features.detach()
    #     gt_features = gt_datastruct.features
    #     for id,idx in enumerate(ids):
    #         pred_feature = pred_features[id][:lengths[id]]
    #         gt_feature = gt_features[id][:lengths[id]]
    #         if idx not in self.feat_cache:
    #             if self.mse(pred_feature,gt_feature) < self.prev_thresh:
    #                 self.feat_cache[idx] = {"prev":pred_feature[-self.temporal_window:],"next":pred_feature[:self.temporal_window]}
    #         else:
    #             if self.mse(pred_feature[-self.temporal_window:],gt_feature[-self.temporal_window:]) < self.mse(self.feat_cache[idx]["prev"],gt_feature[-self.temporal_window:]):
    #                 self.feat_cache[idx]["prev"] = pred_feature[-self.temporal_window:]
    #             if self.mse(pred_feature[:self.temporal_window],gt_feature[:self.temporal_window]) < self.mse(self.feat_cache[idx]["next"],gt_feature[:self.temporal_window]):
    #                 self.feat_cache[idx]["next"] = pred_feature[:self.temporal_window]
    
    def update_feat_cache(self,ids,text_datastruct,gt_datastruct,lengths):
        pred_features = text_datastruct.features.detach()
        # gt_features = gt_datastruct.features
        for id,idx in enumerate(ids):
            pred_feature = pred_features[id][:lengths[id]]
            if pred_feature.size(0)< self.temporal_window:
                pred_feature = self.temporal_pool(pred_feature.unsqueeze(0).permute(0,2,1)).permute(0,2,1).squeeze(0)
            if idx not in self.feat_cache:
                self.feat_cache[idx] = {"prev":pred_feature[-self.temporal_window:],"next":pred_feature[:self.temporal_window]}
            else:
                self.feat_cache[idx]["prev"] = pred_feature[-self.temporal_window:]
                self.feat_cache[idx]["next"] = pred_feature[:self.temporal_window]
     
    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        # print(batch["keyids"])
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        datastruct_from_text, latent_from_text, distribution_from_text, text_encoded, text_lengths, recons_text_encoded = ret
        # rfeats: [batch,frames,135]
        # latent: [batch,256]
        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(batch["datastruct"],
                                            batch["length"],
                                            text_lengths,
                                            return_latent=True)
        datastruct_from_motion, latent_from_motion, distribution_from_motion, motion_recons_text_encoded = ret
        
        # Update prev and next features from 
        prev_features = batch["prev_datastruct"].features
        next_features = batch["next_datastruct"].features
        for idx,prev_id in enumerate(batch["prev_ids"]):
            if prev_id in self.feat_cache:
                prev_features[idx] = self.feat_cache[prev_id]["prev"]
        for idx, next_id in enumerate(batch["next_ids"]):
            if next_id in self.feat_cache:
                next_features[idx] = self.feat_cache[next_id]["next"]
        
        ret = self.connect_to_connect_forward(prev_features, datastruct_from_text, next_features, batch["prev_ids"], batch["next_ids"], batch["length"], return_latent = True)
        datastruct_from_connect, latent_from_connect, distribution_from_connect, connect_lengths = ret
        
        if self.if_humor:
            ret = self.humor_forward(batch["datastruct"],
                                     batch["length"],
                                     connect_lengths,
                                     return_latent=True)
            datastruct_from_humor, latent_from_humor, distribution_from_humor = ret
        
        contrast_text_distribution= None
        contrast_text_latent_vector = None
        contrast_motion_distribution = None
        contrast_motion_latent_vector = None
        
        if self.if_contrast:
            ret = self.contrast_forward(batch["contrast_text"], batch["contrast_datastruct"], batch["contrast_length"])
            contrast_text_distribution,contrast_text_latent_vector,contrast_motion_distribution,contrast_motion_latent_vector = ret
        # GT data
        datastruct_ref = batch["datastruct"]
        datastruct_ref_connect = batch["connect_datastruct"]
        ref_probs = None
        pred_probs = None
        
        self.update_feat_cache(batch["keyids"],
                                datastruct_from_text,
                                batch["datastruct"],
                                batch["length"])
        
        # Compare to a Normal distribution
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
            mu_refconnect = torch.zeros_like(distribution_from_connect.loc)
            scale_refconnect = torch.ones_like(distribution_from_connect.scale)
            distribution_ref_connect = torch.distributions.Normal(mu_refconnect,scale_refconnect)
        else:
            distribution_ref = None
        
        # Compute the losses
        if self.if_weighted:
            weights = batch["weights"]
        else:
            weights = None
        if self.if_humor:
            loss = self.losses[split].update(ds_text=datastruct_from_text,
                                         ds_motion=datastruct_from_motion,
                                         ds_connect = datastruct_from_connect,
                                         ds_ref=datastruct_ref,
                                         ds_ref_connect = datastruct_ref_connect,
                                         ds_humor = datastruct_from_humor,
                                         lat_text=latent_from_text,
                                         lat_motion=latent_from_motion,
                                         lat_connect = latent_from_connect,
                                         lat_humor = latent_from_humor,
                                         lat_motion_contrast = contrast_motion_latent_vector,
                                         lat_text_contrast = contrast_text_latent_vector,
                                         dis_text=distribution_from_text,
                                         dis_motion=distribution_from_motion,
                                         dis_connect = distribution_from_connect,
                                         dis_humor = distribution_from_humor,
                                         dis_motion_contrast = contrast_motion_distribution,
                                         dis_text_contrast = contrast_text_distribution,
                                         dis_ref = distribution_ref,
                                         dis_ref_connect = distribution_ref_connect,
                                         ref_probs = ref_probs,
                                         pred_probs = pred_probs,
                                         motion2text = motion_recons_text_encoded,
                                         text2text = recons_text_encoded,
                                         textref = text_encoded,
                                         weights=weights,
                                         lengths = batch["length"])
        else:
            loss = self.losses[split].update(ds_text=datastruct_from_text,
                                         ds_motion=datastruct_from_motion,
                                         ds_connect = datastruct_from_connect,
                                         ds_ref=datastruct_ref,
                                         ds_ref_connect = datastruct_ref_connect,
                                         lat_text=latent_from_text,
                                         lat_motion=latent_from_motion,
                                         lat_connect = latent_from_connect,
                                         lat_motion_contrast = contrast_motion_latent_vector,
                                         lat_text_contrast = contrast_text_latent_vector,
                                         dis_text=distribution_from_text,
                                         dis_motion=distribution_from_motion,
                                         dis_connect = distribution_from_connect,
                                         dis_motion_contrast = contrast_motion_distribution,
                                         dis_text_contrast = contrast_text_distribution,
                                         dis_ref = distribution_ref,
                                         dis_ref_connect = distribution_ref_connect,
                                         ref_probs = ref_probs,
                                         pred_probs = pred_probs,
                                         motion2text = motion_recons_text_encoded,
                                         text2text = recons_text_encoded,
                                         textref = text_encoded,
                                         weights=weights,
                                         lengths = batch["length"])
        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
        
        if split == "val":
            # Compute the metrics
            self.metrics.update(datastruct_from_connect.detach().joints,
                                datastruct_ref_connect.detach().joints,
                                connect_lengths)
        return loss