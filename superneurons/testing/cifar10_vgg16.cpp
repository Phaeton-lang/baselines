#include <stdlib.h>
#include <superneurons.h>

using namespace SuperNeurons;

int main(int argc, char **argv) {

    char* train_label_bin;
    char* train_image_bin;
    char* test_label_bin;
    char* test_image_bin;
    char* train_mean_file;

    train_mean_file = (char *) "/home/ljs/data/data/cifar_train.mean";
    train_image_bin = (char *) "/home/ljs/data/data/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/home/ljs/data/data/cifar10_train_label_0.bin";
    test_image_bin  = (char *) "/home/ljs/data/data/cifar10_test_image_0.bin";
    test_label_bin  = (char *) "/home/ljs/data/data/cifar10_test_label_0.bin";

    if(argc != 2) {
      printf("ERROR! Please input batch size!!!\n");
      exit(-1);
    }
    int bs = std::stoi(argv[1]);
    const size_t batch_size = bs; //train and test must be same
    const size_t C = 3, H = 32, W = 32;
    const int flag = 1;     // 1 for read from memory, 0 for read from disk


    base_preprocess_t<float>* mean_sub =
            (base_preprocess_t<float>*) new mean_subtraction_t<float>(batch_size, C, H, W, train_mean_file);

    base_preprocess_t<float> *pad = (base_preprocess_t<float> *) new border_padding_t<float>(
            batch_size, C, H, W, 4, 4);
    base_preprocess_t<float> *crop = (base_preprocess_t<float> *) new central_crop_t<float>(
            batch_size, C, H + 8, W + 8, batch_size, C, H, W);
    base_preprocess_t<float> *flip = (base_preprocess_t<float> *) new random_flip_left_right_t<float>(
            batch_size, C, H, W);
    base_preprocess_t<float> *bright = (base_preprocess_t<float> *) new random_brightness_t<float>(
            batch_size, C, H, W, 63);
    base_preprocess_t<float> *contrast = (base_preprocess_t<float> *) new random_contrast_t<float>(
            batch_size, C, H, W, 0.2, 1.8);
    base_preprocess_t<float> *standardization =
            (base_preprocess_t<float> *) new per_image_standardization_t<float>(
                    batch_size, C, H, W);

    preprocessor<float>* processor = new preprocessor<float>();
    processor->add_preprocess(mean_sub)
            ->add_preprocess(pad)
            ->add_preprocess(crop)
            ->add_preprocess(flip)
            ->add_preprocess(bright)
            ->add_preprocess(contrast)
            ->add_preprocess(standardization);
    preprocessor<float>* p2 = new preprocessor<float>();
    p2->add_preprocess(new mean_subtraction_t<float>(batch_size, C, H, W, train_mean_file))
            ->add_preprocess(new per_image_standardization_t<float>(batch_size, C, H, W));

    //test
    parallel_reader_t<float > *reader2 = new parallel_reader_t<float>(test_image_bin, test_label_bin, 2, batch_size, 3, 32, 32, p2, 4, 1,
                                                                      flag);
    base_layer_t<float>* data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, reader2);
    //train
    parallel_reader_t<float > *reader1 = new parallel_reader_t<float>(train_image_bin, train_label_bin, 2, batch_size, 3, 32, 32, processor, 4, 4,
                                                                      flag);
    base_layer_t<float>* data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, reader1);


    /*--------------network configuration--------------*/

    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.01, 0.0, 0.9);
    network_t<float> n(solver);

    base_layer_t<float> *conv1_1 = (base_layer_t<float> *) new conv_layer_t<float>(64, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act1_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv1_2 = (base_layer_t<float> *) new conv_layer_t<float>(64, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act1_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_1 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);


    base_layer_t<float> *conv2_1 = (base_layer_t<float> *) new conv_layer_t<float>(128, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act2_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv2_2 = (base_layer_t<float> *) new conv_layer_t<float>(128, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01),
                                                                                   true);
    base_layer_t<float> *act2_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_2 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);



    base_layer_t<float> *conv3_1 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act3_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv3_2 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act3_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv3_3 = (base_layer_t<float> *) new conv_layer_t<float>(256, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act3_3 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_3 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);



    base_layer_t<float> *conv4_1 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act4_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv4_2 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act4_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv4_3 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act4_3 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float> *pool_4 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);



    base_layer_t<float> *conv5_1 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act5_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv5_2 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act5_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *conv5_3 = (base_layer_t<float> *) new conv_layer_t<float>(512, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                   true);
    base_layer_t<float> *act5_3 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                 CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *pool_5 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);




    base_layer_t<float> *full_conn_1 = (base_layer_t<float> *) new fully_connected_layer_t<float>(4096, new gaussian_initializer_t<float>(0, 0.01), true);
    base_layer_t<float> *relu6       = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *drop6       = (base_layer_t<float> *) new dropout_layer_t<float>(0.5);

    base_layer_t<float> *full_conn_2 = (base_layer_t<float> *) new fully_connected_layer_t<float>(4096, new gaussian_initializer_t<float>(0, 0.01),
                                                                                                  true);
    base_layer_t<float> *relu7       = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float> *drop7       = (base_layer_t<float> *) new dropout_layer_t<float>(0.5);

    base_layer_t<float> *full_conn_3 = (base_layer_t<float> *) new fully_connected_layer_t<float>(10, new gaussian_initializer_t<float>(0, 0.01),
                                                                                                  true);
    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE,
                                                                                      CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to( conv1_1 );
    //setup network
    data_1->hook( conv1_1 );

    conv1_1->hook(act1_1);
    act1_1->hook(conv1_2);
    conv1_2->hook(act1_2);
    act1_2->hook(pool_1);
    pool_1->hook(conv2_1);

    conv2_1->hook(act2_1);
    act2_1->hook(conv2_2);
    conv2_2->hook(act2_2);
    act2_2->hook(pool_2);
    pool_2->hook(conv3_1);

    conv3_1->hook(act3_1);
    act3_1->hook(conv3_2);
    conv3_2->hook(act3_2);
    act3_2->hook(conv3_3);
    conv3_3->hook(act3_3);
    act3_3->hook(pool_3);
    pool_3->hook(conv4_1);

    conv4_1->hook(act4_1);
    act4_1->hook(conv4_2);
    conv4_2->hook(act4_2);
    act4_2->hook(conv4_3);
    conv4_3->hook(act4_3);
    act4_3->hook(pool_4);
    pool_4->hook(conv5_1);

    conv5_1->hook(act5_1);
    act5_1->hook(conv5_2);
    conv5_2->hook(act5_2);
    act5_2->hook(conv5_3);
    conv5_3->hook(act5_3);
    act5_3->hook(pool_5);
    pool_5->hook(full_conn_1);

    full_conn_1->hook(relu6);
    relu6->hook(drop6);

    drop6->hook(full_conn_2);
    full_conn_2->hook(relu7);
    relu7->hook(drop7);
    drop7->hook(full_conn_3);

    full_conn_3->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);

    n.setup_test( data_2, 100 );
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs/batch_size;
    n.train(20000, tracking_window, 1000);

    delete reader1;
    delete reader2;
}
