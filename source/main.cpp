#include<opencv2/opencv.hpp>

float confThreshold = 0.5f;
float nmsThreshold = 0.4f;
int inpWidth = 416;
int inpHeight = 416;
std::vector<std::string> classes;

std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net) {
    static std::vector<cv::String> names;
    if (names.empty()) {
        std::vector<int> outLayers =
            net.getUnconnectedOutLayers();
        std::vector<cv::String> layersNames =
            net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i]-1];
    }
    return names;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat & frame) {
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
}


void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
               box.x + box.width, box.y + box.height, frame);
    }
    return;
}

int detect(cv::Mat& image) {

    // Load names of classes
    std::string classesFile = "coco.names";
    
    std::string line;
    std::ifstream ifs(classesFile.c_str());
    while (std::getline(ifs, line))
        classes.push_back(line);

    std::string modelConfiguration = "yolov3.cfg";
    std::string modelWeights = "yolov3.weights";

    // Load the network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(
        modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    if (image.empty()) {
        std::cout << "input image empty !!!" << std::endl;
        return 1;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0,
        cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(image, outs);

    // Profile
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time for a frame: %.2f ms", t);
    cv::putText(image, label, cv::Point(0, 15),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    cv::Mat detectedFrame;
    image.convertTo(detectedFrame, CV_8U);
    cv::imwrite("predict.png",detectedFrame);

    return 0;
}

int main(int argc, char * argv[]) {
    cv::Mat image = cv::imread("D:/workspace/cvyolo/data/dark.jpg");
    cv::imshow("image", image);
    detect(image);
    cv::waitKey();
    return 0;
}
