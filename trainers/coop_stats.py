# save coop inference features, for channel importance statistics
from dassl.engine import TRAINER_REGISTRY

from .coop import CoOp


@TRAINER_REGISTRY.register()
class CoOpStats(CoOp):
    def model_inference(self, image):
        image_features = self.model.image_encoder(image.type(self.model.dtype))

        prompts = self.model.prompt_learner()
        tokenized_prompts = self.model.tokenized_prompts
        text_features = self.model.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features, image_features
