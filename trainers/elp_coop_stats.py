# save coop w/ DePT inference features, for channel importance statistics
from dassl.engine import TRAINER_REGISTRY

from .elp_coop import ExtrasLinearProbeCoOp


@TRAINER_REGISTRY.register()
class ExtrasLinearProbeCoOpStats(ExtrasLinearProbeCoOp):
    def model_inference(self, input_):
        if self.is_base:
            return self._forward_base(input_)
        else:
            return self._forward_new(input_)

    def _forward_base(self, img, labels=None):
        assert not self.model.prompt_learner.training
        
        text_feats, img_feats = self.model._forward_feats(img)

        text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)

        if self.model.film_cfg.LINEAR_PROBE:
            text_feats_lp = self.model.film_lp_text(text_feats)
            img_feats_lp = self.model.film_lp_img(img_feats)
            text_feats_lp_norm = text_feats_lp / text_feats_lp.norm(dim=-1, keepdim=True)
            img_feats_lp_norm = img_feats_lp / img_feats_lp.norm(dim=-1, keepdim=True)

        lambda_ = 0
        text_feats_norm = text_feats_norm * (1 - lambda_) + text_feats_lp_norm * lambda_
        img_feats_norm = img_feats_norm * (1 - lambda_) + img_feats_lp_norm * lambda_

        return text_feats_norm, img_feats_norm
    
    def _forward_new(self, img):
        assert not self.model.prompt_learner.training
        
        text_feats, img_feats = self.model._forward_feats(img)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return text_feats, img_feats
