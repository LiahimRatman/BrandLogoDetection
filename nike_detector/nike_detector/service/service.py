from typing import Optional

import pkg_resources
from ultralytics import YOLO
from PIL import Image as Img
import gdown

import torch

from .evaluation_pipeline_params import read_nike_detection_pipeline_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
data_path = pkg_resources.resource_filename(__name__, 'configs/base_config.yaml')
feature_config = read_nike_detection_pipeline_params(data_path)


def init_nike_model():
    # Load a model
    output = 'nike_model.pt'
    gdown.download(feature_config.model_path, output, quiet=False)

    models['yolo_model'] = YOLO('yolov8x.yaml')  # build a new model from scratch
    models['yolo_model'] = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
    models['yolo_model'] = YOLO(output)

    return models['yolo_model']


def predict(image,
            save=True,
            save_txt=True):
    run_name = 'predict'
    results = models['yolo_model'].predict(image,
                                           conf=feature_config.confidence,
                                           device=device,
                                           save=True if save else False,
                                           project=feature_config.output_images_dir,
                                           name=run_name,
                                           save_txt=True if save_txt else False,
                                           exist_ok=True)

    result = results[0]
    im_array = result.plot()  # plot a BGR numpy array of predictions
    im_array = im_array[..., ::-1]
    im = Img.fromarray(im_array)  # RGB PIL image

    results = {
        "image": im,
        "image_size": {
            "height": im_array.shape[0],
            "width": im_array.shape[1],
            "channels": im_array.shape[2],
        },
        "boxes": result.boxes.xywhn.tolist()[0]
    }

    return results


def track(video_path: Optional[str],
          save=True,
          save_txt=True):  # path to img or url
    run_name = 'track'

    result = models['yolo_model'].track(source=video_path,
                                        conf=feature_config.confidence,
                                        device=device,
                                        save=True,
                                        project=feature_config.output_images_dir,
                                        name=run_name,
                                        save_txt=True,
                                        exist_ok=True)

    results = {
        'video_save_path': feature_config.output_images_dir + '/' + run_name + '/' + video_path,
        'result': result
    }

    return results


