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


class BasketballDetector:

    def __init__(self, engine_path: str):
        self.input_size = (640, 640)  # 输入图像尺寸
        self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(open(engine_path, "rb").read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

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
        for (hsize, wsize), stride in zip([torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])], [8, 16, 32]):
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
        :param input_size:
        :param img:
        :return:
        """
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
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

    def infer(self, image):
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
            return
        results = trt_outputs[0].numpy()
        # input_image = cv2.resize(input_image, input_size)
        ratio = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        for result in results:
            bbox = list(map(int, result[:4] / ratio))
            score = float(result[4] * result[5])
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(image, str(score), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # print(bbox, result)
        # cv2.imwrite("/home/senseport0/Workspace/YOLOX/assets/output.jpg", image)


def f():
    basketball_detector = BasketballDetector("/home/senseport0/Workspace/YOLOX/YOLOX_outputs/yolox_l_basketball_detection/model_trt.engine")
    input_image = cv2.imread('/home/senseport0/Workspace/YOLOX/assets/6.jpg')
    for i in range(100):
        # s = datetime.datetime.now()
        print(basketball_detector.infer(input_image))
        # print((datetime.datetime.now() - s).total_seconds() * 1000)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    t1 = multiprocessing.Process(target=f)
    t2 = multiprocessing.Process(target=f)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
