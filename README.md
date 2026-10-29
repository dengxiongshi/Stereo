# Stereo

## About

这个项目是基于[lightstereo_cpp](https://github.com/zz990099/lightstereo_cpp)和[banet_cpp](https://github.com/zz990099/banet_cpp)的综合c++实现，除了原先支持的`TensorRT`、`RKNN`、`OnnxRuntime`三种硬件平台，还添加了香橙派AI Pro中的Ascend 3104B(OM)模型的实现。

## Usage

### Download Project

```
git clone https://github.com/dengxiongshi/Stereo.git
```

### Run

```
cd Stereo
./build-starup.sh om|ort|rknn|trt
cd build/bin
./test_stereo_lightstereo | ./test_stereo_banet
```

注意：

1. 对于香橙派AI Pro OM模型，其中OpenCV是经过简化编译，剔除了PNG、JPEG，只支持bmp图片的读写
2. ONNX转成的OM模型，其输出的node名与ONNX模型的输出名不同，需要使用deploy/test.py来确定

   a. 程序依赖acl库，参考 [基于 Caffe ResNet-50 网络实现图片分类](https://gitee.com/ascend/samples/blob/master/python/level2_simple_inference/1_classification/resnet50_imagenet_classification/README_CN.md?svcp_stk=1_t9kS9Iz7z_6ne2efCtV5ZnmHhFnkWNlf_XTmsNLGNKu9f1U5BvzoQjba4ZA9JnPT01Dnga2r2Z_hVOf12RJ6gANL7q1nVIBrOUZ0LRoL2tC62ucrcNuQ1EWSL6zH7vFc3-vqceRqdsRkij2w1X_auT_TekO3eXXBM1P4Vk7xQcf_7iZO5eIkhwXYPg3v9npreyQ7hc8Nv7vewGJWvXk1tg%3D%3D)来搭建环境

   b. 确定好输出名后，有以下几点需要修改：

```c++
     test_stereo_lightstereo.cpp:

      auto engine = CreateOmInferCore(
               "../../weights/LightStereo-S-KITTI.om",
               {{"left", {1, 3, 256, 512}}, {"right", {1, 3, 256, 512}}},
               {{"PartitionedCall_/ReduceSum_1_ReduceSum_217:0:disp_pred", {1, 1, 256, 512}}}
      );
      ​
      lightstereo_model_ = CreateLightStereoModel(
                         engine, preprocess_block, 256, 512,
                         {"left", "right"},
                         {"PartitionedCall_/ReduceSum_1_ReduceSum_217:0:disp_pred"}
      );
      ​

     test_stereo_banet2d.cpp:

      auto engine = CreateOmInferCore(
              "../../weights/BANet-KITTI.om",
              {{"left", {1, 3, 256, 512}}, {"right", {1, 3, 256, 512}}},
              {{"/Mul_9:0:disp_pred", {1, 1, 256, 512}}}
      );

      banet_model_ = CreateBANetModel(
                          engine, preprocess_block, 256, 512,
                          {"left", "right"},
                          {"/Mul_9:0:disp_pred"}
      );
```

## References

[BANet](https://github.com/gangweix/BANet)

[EasyDeploy](https://github.com/zz990099/EasyDeployTool)