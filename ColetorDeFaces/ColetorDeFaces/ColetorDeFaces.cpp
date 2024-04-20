#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/serialize.h>
#include <dlib/matrix.h>
#include <dlib/opencv/cv_image.h>
#include <filesystem>

#include <string>
#include <iostream>
#include <filesystem>
#include <sys/stat.h>


using namespace std;
namespace fs = std::filesystem;
using namespace cv;
using namespace dlib;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;


int main()
{
    frontal_face_detector detector_face = get_frontal_face_detector();

    shape_predictor detector_pontos;
    deserialize("modelos_treinados\\shape_predictor_68_face_landmarks.dat") >> detector_pontos;

    CascadeClassifier haarcascade;
    haarcascade.load("C:\\Users\\harri\\Documents\\GitHub\\ColetorDeFaces\\ColetorDeFaces\\ColetorDeFaces\\modelos_treinados\\haarcascade_frontalface_alt.xml");    

    int qtdFramesPorVideo = 20;

    string root = "C:\\Users\\harri\\Documents\\Programacao\\Python\\CelebV-HQ-main\\CelebV-HQ-main\\downloaded_celebvhq_final";


    for (const auto& pasta : fs::directory_iterator(root)) {
        if (fs::is_directory(pasta.path())) {

            for (const auto& entry : fs::directory_iterator(pasta.path().string())) {
                
                string videoPath = entry.path().string();                    
                cout <<"Processando pasta: "<< pasta.path().string() << "\n";

                    
                if (!fs::is_directory(videoPath)) {

                    std::vector<string> videoPathSplit = split(videoPath, "\\");
                    string pastaVideo = split(videoPathSplit[videoPathSplit.size() - 1], ".")[0];
                    string pastaVideoPath = pasta.path().string() + "\\" + pastaVideo;


                    VideoCapture capture(videoPath);
                    Mat frame;

                    if (!capture.isOpened()) {
                        throw "Erro ao acessar o video";
                    }


                    Mat grayscale_image;
                    std::vector<cv::Rect> features;

                    bool continuaAnalisandoVideo = true;
                    int indexFrame = 0;

                    while (continuaAnalisandoVideo) {

                        capture >> frame;
                        if (frame.empty())
                            break;

                        cvtColor(frame, grayscale_image, COLOR_BGR2GRAY);
                        equalizeHist(grayscale_image, grayscale_image);

                        haarcascade.detectMultiScale(grayscale_image, features, 1.1, 4, 0, Size(30, 30));

                        for (auto&& feature : features) {

                            array2d<bgr_pixel> dlibImg;
                            assign_image(dlibImg, cv_image<bgr_pixel>(frame));


                            for (auto face : detector_face(dlibImg)) {

                                full_object_detection pontos = detector_pontos(dlibImg, face);

                                if (pontos.num_parts() > 0) {

                                    if (indexFrame < qtdFramesPorVideo) {

                                        string arquivoNovo = pastaVideoPath + "\\" + "Frame" + to_string(indexFrame) + ".jpeg";

                                        if (fs::exists(pastaVideoPath) && fs::is_directory(pastaVideoPath)) {
                                            imwrite(arquivoNovo, frame);
                                        }
                                        else {
                                            create_directory(pastaVideoPath);
                                            imwrite(arquivoNovo, frame);
                                        }

                                        indexFrame += 1;
                                    }
                                    else {
                                        continuaAnalisandoVideo = false;
                                    }
                                }
                            }
                        }
                    }
                    capture.release();
                }                    
            }                            
        }
    }
}
