import os
from typing import List, Tuple
from dataclasses import dataclass

import cv2
import torch
import torchvision
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 不能删 init了啥玩意
import tensorrt as trt


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


@dataclass
class DetectResult:
    """
    检测结果
    """
    bbox: Tuple[int, int, int, int]  # bbox框 左上角xy和右下角xy
    score: float  # 分数


class BasketballDetector:

    def __init__(self, onnx_file: str, engine_file: str):
        self.input_size = (640, 640)  # 输入图像尺寸
        self.infer_image_shape = ()  # 当前推理的图像尺寸
        self.trt_logger = trt.Logger()
        self.engine = self.get_engine(onnx_file, engine_file)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def get_engine(self, onnx_file: str, engine_file: str):
        """
        获取engine
        :param onnx_file:
        :param engine_file:
        :return:
        """
        if os.path.exists(engine_file):
            print(f"读取trt engine {engine_file}")
            return trt.Runtime(self.trt_logger).deserialize_cuda_engine(open(engine_file, "rb").read())
        else:
            print(f"trt engine {engine_file}不存在，开始在线转换 {onnx_file}")
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with trt.Builder(self.trt_logger) as builder, \
                    builder.create_network(explicit_batch) as network, \
                    trt.OnnxParser(network, self.trt_logger) as parser:

                builder.max_workspace_size = 1 << 30
                builder.max_batch_size = 1
                if builder.platform_has_fast_fp16:
                    builder.fp16_mode = True
                with open(onnx_file, 'rb') as model_file:
                    if not parser.parse(model_file.read()):
                        print('Failed to parse the ONNX file')
                        for err in range(parser.num_errors):
                            print(parser.get_error(err))
                        return None
                network.get_input(0).shape = [1, 3, *self.input_size]
                engine = builder.build_cuda_engine(network)
                if engine is None:
                    print('Failed to build engine')
                    return None
                with open(engine_file, 'wb') as engine_file:
                    engine_file.write(engine.serialize())
                return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def decode_outputs(self, outputs):
        """
        trt输出解码
        :return:
        """
        grids = []
        strides = []
        dtype = torch.FloatTensor
        for (hsize, wsize), stride in zip([torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])],
                                          [8, 16, 32]):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def pre_processing(self, img: np.ndarray):
        """
        图像预处理
        :param img:
        :return:
        """
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        r = min(self.input_size[0] / self.infer_image_shape[0], self.input_size[1] / self.infer_image_shape[1])
        resized_img = cv2.resize(
            img,
            (int(self.infer_image_shape[1] * r), int(self.infer_image_shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(self.infer_image_shape[0] * r), : int(self.infer_image_shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def post_processing(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output

    def infer(self, image) -> List[DetectResult]:
        self.infer_image_shape = image.shape
        image = self.pre_processing(image)
        np.copyto(self.inputs[0].host, image.flatten())
        trt_outputs = self.do_inference()
        trt_outputs = torch.from_numpy(trt_outputs[0])
        trt_outputs.resize_(1, 8400, 6)
        trt_outputs = self.decode_outputs(trt_outputs)
        trt_outputs = self.post_processing(prediction=trt_outputs,
                                           num_classes=1,
                                           conf_thre=0.3,
                                           nms_thre=0.3,
                                           class_agnostic=True)
        if trt_outputs[0] is None:
            return []
        results = trt_outputs[0].numpy()
        # input_image = cv2.resize(input_image, input_size)
        ratio = min(self.input_size[0] / self.infer_image_shape[0], self.input_size[1] / self.infer_image_shape[1])
        detect_results = []
        for result in results:
            bbox = list(map(int, result[:4] / ratio))
            score = float(result[4] * result[5])
            detect_results.append(DetectResult(bbox=bbox, score=score))
        return detect_results


if __name__ == '__main__':
    import datetime

    image = cv2.imread("assets/dog.jpg")
    print(image.shape)
    basketball_detector = BasketballDetector("checkpoints/yolox_l_basketball_detection.onnx",
                                             "checkpoints/yolox_l_basketball_detection.trt")
    print(image.shape)
    detect_results = basketball_detector.infer(image)
    print(detect_results)
    for detect_result in detect_results:
        cv2.rectangle(image, (detect_result.bbox[0], detect_result.bbox[1]),
                      (detect_result.bbox[2], detect_result.bbox[3]), (0, 0, 255), 2)
        cv2.putText(image, str(detect_result.score), (detect_result.bbox[0], detect_result.bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print(bbox, result)
    cv2.imwrite("assets/output.jpg", image)
