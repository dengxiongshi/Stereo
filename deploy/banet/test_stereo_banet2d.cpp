/**
 * @Copyright (C) ShenZhen ShenZhi WeiLai Co., Ltd. 2017-2025. All rights reserved.
 * @Date 2025/10/27 16:39.
 * @author 邓熊狮
 * @FlieName test_stereo_banet2d
 * @description:
 **/
#include <iostream>
#include <fstream>
#include <memory>
#include "image_processing_utils/image_processing_utils.hpp"
#include "stereo/banet.hpp"
#include "test_utils/stereo_matching_test_utils.hpp"
#include "ort_core/ort_core.hpp"
#include <opencv2/opencv.hpp>  // 假设用OpenCV处理图像
//#include <opencv2/highgui/highgui.hpp>

using namespace easy_deploy;

// 移除GTest依赖，改为普通基类
class BasebanetFixture {
protected:
    std::shared_ptr<BaseStereoMatchingModel> banet_model_;  // 立体匹配模型指针
    std::string left_image_path_;                                 // 左图路径
    std::string right_image_path_;                                // 右图路径
    std::string test_banet_result_path_;                    // 测试结果保存路径
    size_t      speed_test_predict_rounds_;                       // 速度测试循环次数
};

// 继承普通基类，去掉GTest依赖
class BANet_OnnxRuntime_Fixture : public BasebanetFixture {
public:
    // 将GTest的SetUp改为普通初始化函数
    bool init() {
        try {
            // 创建ONNX推理引擎（捕获可能的异常）
            std::cout << "[INFO] 创建ONNX推理引擎..." << std::endl;
            auto engine = CreateOrtInferCore(
                    "./../../../BANet/banet-2d/weights/kitti_ori.onnx",
                    {{"left", {1, 3, 256, 512}}, {"right", {1, 3, 256, 512}}},  // 输入形状
                    {{"disp_pred", {1, 1, 256, 512}}}                                   // 输出形状
            );
            if (!engine) {
                std::cerr << "[ERROR] ONNX推理引擎创建失败！" << std::endl;
                return false;
            }

            // 创建预处理模块
            std::cout << "[INFO] 创建图像预处理模块..." << std::endl;
            auto preprocess_block = CreateCpuImageProcessingResizePad(
                    ImageProcessingPadMode::TOP_RIGHT,
                    ImageProcessingPadValue::EDGE,
                    true, true, {0, 0, 0}, {1, 1, 1}
            );
            if (!preprocess_block) {
                std::cerr << "[ERROR] 预处理模块创建失败！" << std::endl;
                return false;
            }

            // 初始化立体匹配模型
            std::cout << "[INFO] 初始化立体匹配模型..." << std::endl;
            banet_model_ = CreateBANetModel(
                    engine, preprocess_block, 256, 512,
                    {"left", "right"},  // 输入名称
                    {"disp_pred"}               // 输出名称
            );
            if (!banet_model_) {
                std::cerr << "[ERROR] 立体匹配模型初始化失败！" << std::endl;
                return false;
            }

            // 初始化测试参数
            speed_test_predict_rounds_    = 100;
            left_image_path_              = "./../../../lightstereo_cpp/test_data/left.png";
            right_image_path_             = "./../../../lightstereo_cpp/test_data/right.png";
            test_banet_result_path_       = "./banet_ort_test_result.png";

            std::cout << "[INFO] 初始化完成！" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] 初始化过程抛出异常：" << e.what() << std::endl;
            return false;
        }
    }

    // 验证图像路径是否有效
    bool checkImagePaths() const {
        std::cout << "[INFO] 检查图像路径有效性..." << std::endl;
        // 检查左图
        std::ifstream left_file(left_image_path_);
        if (!left_file.good()) {
            std::cerr << "[ERROR] 左图文件不存在：" << left_image_path_ << std::endl;
            return false;
        }
        // 检查右图
        std::ifstream right_file(right_image_path_);
        if (!right_file.good()) {
            std::cerr << "[ERROR] 右图文件不存在：" << right_image_path_ << std::endl;
            return false;
        }
        std::cout << "[INFO] 图像路径验证通过！" << std::endl;
        return true;
    }

    // 执行推理测试
    bool runInferenceTest() const {
        if (!banet_model_) {
            std::cerr << "[ERROR] 模型未初始化，无法执行推理！" << std::endl;
            return false;
        }

        std::cout << "[INFO] 加载测试图像..." << std::endl;
        cv::Mat left_img = cv::imread(left_image_path_);
        cv::Mat right_img = cv::imread(right_image_path_);
        if (left_img.empty()) {
            std::cerr << "[ERROR] 左图加载失败：" << left_image_path_ << std::endl;
            return false;
        }
        if (right_img.empty()) {
            std::cerr << "[ERROR] 右图加载失败：" << right_image_path_ << std::endl;
            return false;
        }

        std::cout << "[INFO] 执行模型推理..." << std::endl;
        cv::Mat disp;  // 视差图输出
        bool infer_success = banet_model_->ComputeDisp(left_img, right_img, disp);
        if (!infer_success || disp.empty()) {
            std::cerr << "[ERROR] 模型推理失败或视差图为空！" << std::endl;
            return false;
        }
        if (!test_banet_result_path_.empty()) {
            double minVal, maxVal;
            cv::minMaxLoc(disp, &minVal, &maxVal);
            cv::Mat normalized_disp_pred;
            disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                           -minVal * 255.0 / (maxVal - minVal));

            cv::Mat color_normalized_disp_pred;
            cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
            std::cout << "[INFO] 推理成功，保存结果到：" << test_banet_result_path_ << std::endl;
            if (!cv::imwrite(test_banet_result_path_, color_normalized_disp_pred)) {
                std::cerr << "[WARNING] 结果图像保存失败！" << std::endl;
                // 保存失败不影响整体流程，仅警告
            }
        }

        return true;
    }

    // 执行速度测试（循环推理多次）
    bool runSpeedTest() const {
        if (!banet_model_) {
            std::cerr << "[ERROR] 模型未初始化，无法执行速度测试！" << std::endl;
            return false;
        }

        std::cout << "[INFO] 加载测试图像（速度测试）..." << std::endl;
        cv::Mat left_img = cv::imread(left_image_path_);
        cv::Mat right_img = cv::imread(right_image_path_);
        if (left_img.empty() || right_img.empty()) {
            std::cerr << "[ERROR] 速度测试图像加载失败！" << std::endl;
            return false;
        }

        std::cout << "[INFO] 开始速度测试（循环" << speed_test_predict_rounds_ << "次）..." << std::endl;
        auto start = cv::getTickCount();  // 计时开始

        for (size_t i = 0; i < speed_test_predict_rounds_; ++i) {
            cv::Mat disp;
            if (!banet_model_->ComputeDisp(left_img, right_img, disp) || disp.empty()) {
                std::cerr << "[ERROR] 第" << i+1 << "次推理失败！" << std::endl;
                return false;
            }
        }

        auto end = cv::getTickCount();    // 计时结束
        double elapsed = (end - start) / cv::getTickFrequency();  // 总耗时（秒）
        double avg_time = elapsed / speed_test_predict_rounds_;   // 平均耗时（秒）

        std::cout << "[INFO] 速度测试完成！" << std::endl;
        std::cout << "[RESULT] 总耗时：" << elapsed << "秒，平均每次：" << avg_time << "秒，FPS：" << 1/avg_time << std::endl;

        return true;
    }
};

// 主函数：调用Fixture类执行所有测试流程
int main() {
    std::cout << "===== Lightstereo ONNX Runtime 测试程序 =====" << std::endl;

    // 实例化测试类
    BANet_OnnxRuntime_Fixture test_fixture;

    // 步骤1：初始化（模型、参数）
    if (!test_fixture.init()) {
        std::cerr << "===== 测试失败：初始化阶段出错 =====" << std::endl;
        return 1;  // 初始化失败，返回错误码
    }

    // 步骤2：验证图像路径
    if (!test_fixture.checkImagePaths()) {
        std::cerr << "===== 测试失败：图像路径无效 =====" << std::endl;
        return 1;
    }

    // 步骤3：执行推理测试
    if (!test_fixture.runInferenceTest()) {
        std::cerr << "===== 测试失败：推理阶段出错 =====" << std::endl;
        return 1;
    }

    // 步骤4：执行速度测试（可选，根据需求启用）
//    if (!test_fixture.runSpeedTest()) {
//        std::cerr << "===== 测试失败：速度测试阶段出错 =====" << std::endl;
//        return 1;
//    }

    // 所有测试通过
    std::cout << "===== 所有测试通过！ =====" << std::endl;
    return 0;
}