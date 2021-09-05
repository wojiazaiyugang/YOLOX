import os
from datetime import datetime

import cv2
from helper import onnx2tensorrt

from inference import do_inference
from inference import get_engine, allocate_buffers, load_data

if __name__ == '__main__':
    # test()
    # exit()

    onnx_file = r"/home/senseport0/Workspace/HiAlgorithm/mmdetection/checkpoints/ssd_openpai_epoch1.onnx"
    trt_file = r"/home/senseport0/Workspace/HiAlgorithm/mmdetection/checkpoints/ssd_openpai_epoch1.trt"
    # build_engine(onnx_file, trt_file)
    # onnx2tensorrt(onnx_file, trt_file)
    # exit()

    # test_dir = r"/home/senseport0/Workspace/HiAlgorithm/mmclassification/data/goal_classification/test"
    # d = os.path.join(test_dir, "0")
    with get_engine(trt_file) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine=engine)
        for i in range(100):
            s = datetime.now()
            inputs[0].host = load_data(cv2.imread(os.path.join(r"./test_image.jpg")))
            output = do_inference(context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print((datetime.now() - s).total_seconds() * 1000)
        print(output)