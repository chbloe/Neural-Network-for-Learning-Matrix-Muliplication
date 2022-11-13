#include <string>
#include <limits>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <experimental/filesystem>
#include "Strassen_NN.h"

using namespace std;
using namespace arma;
namespace fs = std::experimental::filesystem;




/**
    Constructor
*/
Strassen_NN::Strassen_NN(vector<int>& matrix_dimensions,
                            int rank_estimate,
                            size_t training_size,
                            size_t test_size,
                            int seed_num,
                            size_t epochs,
                            double learning_rate,
                            double regularization_parameter,
                            double range_scale_factor,
                            int exp_id,
                            double threshold_error_out,
                            string data_series_path,
                            string comment)

:   matrix_dimensions(matrix_dimensions),
    rank_estimate(rank_estimate),
    seed_num(seed_num),
    epochs(epochs),
    training_size(training_size),
    test_size(test_size),
    learning_rate(learning_rate),
    regularization_parameter(regularization_parameter),
    weight_decay_factor(learning_rate*regularization_parameter/training_size),
    threshold_error_out(threshold_error_out),
    range_scale_factor(range_scale_factor),
    data_series_path(data_series_path),
    comment(comment),

    /**
        initialize vectors and matrices
    */

    /// layers
    x_0A(vec(matrix_dimensions[0]*matrix_dimensions[1], fill::zeros)),
    x_0B(vec(matrix_dimensions[1]*matrix_dimensions[2], fill::zeros)),
    x_1(vec(rank_estimate, fill::zeros)),
    x_2(vec(matrix_dimensions[0]*matrix_dimensions[2], fill::zeros)),

    /// signals
    s_1A(vec(rank_estimate, fill::zeros)),
    s_1B(vec(rank_estimate, fill::zeros)),

    /// sensitivities
    temp(vec(rank_estimate, fill::zeros)),
    delta_1A(vec(rank_estimate, fill::zeros)),
    delta_1B(vec(rank_estimate, fill::zeros)),
    delta_2(vec(matrix_dimensions[0]*matrix_dimensions[2], fill::zeros)),

    /// weight matrices
    W_1A(mat(rank_estimate, matrix_dimensions[0]*matrix_dimensions[1], fill::randu)),
    W_1B(mat(rank_estimate, matrix_dimensions[1]*matrix_dimensions[2], fill::randu)),
    W_2(mat(matrix_dimensions[0]*matrix_dimensions[2], rank_estimate, fill::randu)),

    /// matrices required for weight updates
    v_dW_2(mat(matrix_dimensions[0]*matrix_dimensions[2], rank_estimate, fill::zeros)),
    v_dW_1A(mat(rank_estimate, matrix_dimensions[0]*matrix_dimensions[1], fill::zeros)),
    v_dW_1B(mat(rank_estimate, matrix_dimensions[1]*matrix_dimensions[2], fill::zeros)),
    /// ---
    S_dW_2(mat(matrix_dimensions[0]*matrix_dimensions[2], rank_estimate, fill::zeros)),
    S_dW_1A(mat(rank_estimate, matrix_dimensions[0]*matrix_dimensions[1], fill::zeros)),
    S_dW_1B(mat(rank_estimate, matrix_dimensions[1]*matrix_dimensions[2], fill::zeros)),

    /// errors
    in_sample_error(std::numeric_limits<double>::max() * vec(epochs, fill::ones)),
    out_sample_error(std::numeric_limits<double>::max() * vec(epochs, fill::ones))

{
    //
    //set_optimal_weights_2_2_2();
    //////cout << "!!! optimal weights set" << endl;
    //    set_near_optimal_weights_2_2_2();
    //    cout << "!!! near optimal weights set" << endl;

    /// weights are currently between [0,1]. shift to between [-1,1].
    W_1A *= 2; W_1A -= 1;
    W_1B *= 2; W_1B -= 1;
    W_2 *= 2; W_2 -= 1;


    /// create a directory to save the data for this SNN instance
    std::stringstream path;
    path << data_series_path <<
                        "rsf_" << range_scale_factor << " " <<
                        "lr_" << learning_rate << " " <<
                        "rp_" << regularization_parameter << " " <<
                        "seed_" << seed_num << " " <<
                        "exp_id_" << exp_id  << "/";

    instance_path = path.str();
    fs::create_directory(instance_path);

    /// save experiment parameters
    save_info();
}


void Strassen_NN::set_epochs(int e)
{
    epochs = e;
    in_sample_error.set_size( epochs );
    out_sample_error.set_size( epochs );
}


double Strassen_NN::forward_propagation(const mat& A, const mat& B)
{
    /// vectorized input for network
    x_0A = vectorise(A);
    x_0B = vectorise(B);

    s_1A = W_1A * x_0A;
    s_1B = W_1B * x_0B;

    /// compute hidden layer
    x_1 = s_1A % s_1B;  /// also  = s_1

    /// compute output layer
    x_2 = W_2 * x_1;

    /// compute target for this data & current deviation
    delta_2 = x_2 - vectorise( A * B );

    return dot(delta_2, delta_2);
}


/**
    compute deltas for

   dE/dw = dE/ds * ds/dw
         = delta * x

*/
void Strassen_NN::backward_propagation()
{
    temp = W_2.t() * delta_2;

    /// hidden layer
    delta_1A = s_1B % temp;

    /// hidden layer
    delta_1B = s_1A % temp;
}



/**
    stochastic gradient descent with momentum
*/

void Strassen_NN::momentum(const vec& delta, const vec& x, mat& W, mat& v_dW)
{
    const mat dW = delta * x.t();
    v_dW = beta_1 * v_dW + learning_rate * dW;
    W -= v_dW + weight_decay_factor *  W;
}

void Strassen_NN::update_weight_matrices()
{
    momentum(delta_2, x_1, W_2, v_dW_2);
    momentum(delta_1A, x_0A, W_1A, v_dW_1A);
    momentum(delta_1B, x_0B, W_1B, v_dW_1B);
}



/**

    Adam Optimization (for adaptive gradient algorithm)
    is a modified stochastic gradient descent
    with per-parameter learning rate

*/

void Strassen_NN::adam_optimization(const vec& delta, const vec& x, mat& W, mat& v_dW, mat& S_dW)
{
    mat dW = delta * x.t();

    v_dW = beta_1 * v_dW + (1-beta_1) * dW;
    S_dW = beta_2 * S_dW + (1-beta_2) * arma::square(dW);

    mat v_dW_corr = v_dW / (1-beta_1_t);
    mat S_dW_corr = S_dW / (1-beta_2_t);

    W -= learning_rate * ( v_dW_corr  /  (arma::sqrt(S_dW_corr) + epsilon) ) + weight_decay_factor * W ;
}
//
//
//void Strassen_NN::update_weight_matrices()
//{
//    beta_1_t *= beta_1;
//    beta_2_t *= beta_2;
//
//    adam_optimization(delta_2, x_1, W_2, v_dW_2, S_dW_2);
//    adam_optimization(delta_1A, x_0A, W_1A, v_dW_1A, S_dW_1A);
//    adam_optimization(delta_1B, x_0B, W_1B, v_dW_1B, S_dW_1B);
//}


/**

    Training data is constantly generated "batchwise",
    because there is literally infinite amount of
    training data available.

*/
void Strassen_NN::run()
{
    /// control seed for experiment
    arma_rng::set_seed(seed_num);

    for (size_t i = 0; i < epochs; ++i) {

        double e_in = 0.0;

        /// generate training and test data for current experiment
        cube training_A(matrix_dimensions[0], matrix_dimensions[1], training_size, fill::randu);
        cube training_B(matrix_dimensions[1], matrix_dimensions[2], training_size, fill::randu);
        expand_data_range(training_A, training_B, 2.0);

        /// run through entire training set
        for(size_t j = 0; j < training_size; ++j) {

            e_in += forward_propagation(training_A.slice(j), training_B.slice(j));
            backward_propagation();
            update_weight_matrices();
        }

        /// round all weights to nearest integer
        W_1A = arma::round(W_1A);
        W_1B = arma::round(W_1B);
        W_2 = arma::round(W_2);

        /// in-sample error
        in_sample_error[i] = e_in / training_size;
        /// out-of-sample error
        out_sample_error[i] = test_out_of_sample();

        /// check whether current weight matrices should be saved
       if ( (i > 0) && (out_sample_error[i] < threshold_error_out) && (out_sample_error[i] < out_sample_error[i-1]) ) {
            /// save in-sample errors and weight matrices
            save_weights(i);
        }

    }

    /// save errors and final weights
    save_data(epochs);
}







//
//void Strassen_NN::run_mini_batches_with_adam()
//{
//    for (size_t i = 0; i < epochs; ++i) {
//
//        double e_in = 0.0;
//
//        /// run through entire training set
//        for(size_t j = 0; j < training_size; j += batchsize) {
//
//            /// reset variables required to compute adaptive learning rate
//            v_dW_2.zeros(); S_dW_2.zeros();
//            v_dW_1A.zeros(); S_dW_1A.zeros();
//            v_dW_1B.zeros(); S_dW_1B.zeros();
//
//            beta_1_t = 1.0;
//            beta_2_t = 1.0;
//
//            /// run through a mini-batch
//            for (size_t b = 0; b < batchsize; ++b) {
//
//                const auto item_no = j + b;
//
//                e_in += forward_propagation(training_A.slice(item_no), training_B.slice(item_no));
//
//                backward_propagation();
//
//                update_weight_matrices();
//            }
//        }
//
//        /// in-sample error
//        in_sample_error[i] = e_in / training_size;
//
//        /// out-of-sample error
//        out_sample_error[i] = test_out_of_sample();
//
//
//        /// check whether current weight matrices should be saved
//       if ( (i > 0) && (out_sample_error[i] < threshold_error_out) && (out_sample_error[i] < out_sample_error[i-1]) ) {
//            /// save in-sample errors and weight matrices
//            save_weights(i);
//        }
//
//
//#if 0
//
//        cout << "e_in( epoch " << i << " ) : " << in_sample_error[i] << endl;
//        //cout << "e_in_other( epoch " << i << " ) : " << e_in_other/ training_A.n_slices << endl;
//
//        cout << "e_out( epoch " << i << " ) : " << out_sample_error[i] << endl << endl;
//
//        /// display weights
//        //display_weight_matrices();
//#endif // 0
//    }
//
//    /// save errors and final weights
//    save_data(epochs);
//}


/**

   compute out-of-sample error for current epoch
*/
double Strassen_NN::test_out_of_sample()
{
    /// generate test data
    cube test_A(matrix_dimensions[0], matrix_dimensions[1], test_size, fill::randu);
    cube test_B(matrix_dimensions[1], matrix_dimensions[2], test_size, fill::randu);
    expand_data_range(test_A, test_B, range_scale_factor);

    double e_out = 0.0;

    for(size_t i = 0; i < test_size; ++i) {
        e_out += forward_propagation(test_A.slice(i), test_B.slice(i));
    }

    return e_out / test_size;
}
