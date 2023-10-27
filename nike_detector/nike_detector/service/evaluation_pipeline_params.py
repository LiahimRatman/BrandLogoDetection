import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass()
class NikeDetectionPipelineParams:
    brand_name: str
    confidence: float
    output_images_dir: str
    model_path: str


NikeDetectionPipelineParamsSchema = class_schema(NikeDetectionPipelineParams)


def read_nike_detection_pipeline_params(config_path: str) -> NikeDetectionPipelineParams:
    """
    Read Nike Logo detection pipeline parameters.
    :param config_path: config path
    :return: NikeDetectionPipelineParams
    """
    schema = NikeDetectionPipelineParamsSchema()
    with open(config_path, 'r') as classification_config:
        params = yaml.safe_load(classification_config)

    return schema.load(params)
