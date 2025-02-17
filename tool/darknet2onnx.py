import sys
import torch
from tool.darknet2pytorch import Darknet


def transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name=None):
    # 创建模型
    model = Darknet(cfgfile)

    # 打印网络
    model.print_network()
    # 加载权重
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    # 如果batch_size设置为<=0，那么设置ONNX模型为动态输出
    dynamic = False
    if batch_size <= 0:
        dynamic = True

    # 输入名
    input_names = ["input"]
    # 输出名
    output_names = ['boxes', 'confs']

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        if not onnx_file_name:
            onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.height, model.width)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weightfile')
    parser.add_argument('--batch_size', type=int,
                        help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--onnx_file_path', help="Output onnx file path")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    transform_to_onnx(args.config, args.weightfile, args.batch_size, args.onnx_file_path)
