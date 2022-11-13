#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <experimental/filesystem>
#include <chrono>

#include "Strassen_NN.h"

using namespace std;
namespace fs = std::experimental::filesystem;
using namespace arma;


///------------------------------------------------------------------------------------------
///------UTILITIES---------------------------------------------------------------------------
///------------------------------------------------------------------------------------------


/**
    meant to be applied to uniformly distributed matrices with elements in [0,1]
*/
void Strassen_NN::expand_data_range(cube& A, cube& B, double scale, double upper_matrix_element_magnitude_boundary)
{
    A *= 2*scale*upper_matrix_element_magnitude_boundary;
    A -= scale*upper_matrix_element_magnitude_boundary;

    B *= 2*scale*upper_matrix_element_magnitude_boundary;
    B -= scale*upper_matrix_element_magnitude_boundary;
}


/**
    meant to be applied to uniformly distributed matrices with elements in [0,1]
*/
void Strassen_NN::expand_data_range(cube& A, cube& B, double scale)
{
    A *= 2*scale;
    A -= scale;

    B *= 2*scale;
    B -= scale;
}

void Strassen_NN::display_weight_matrices() const
{
    cout << "W_1a: " << endl << W_1A << endl;
    cout << "W_1b: " << endl << W_1B << endl;
    cout << "W_2: " << endl << W_2 << endl;
}


void Strassen_NN::save_data(int n)
{
    save_weights(n);
    save_errors();
}


void Strassen_NN::save_info() const
{
    const string file_path = instance_path +  "data_info.txt";

    ofstream data_info;
    data_info.open(file_path);

    data_info  << scientific <<
        "rank estimate: " << rank_estimate  << endl <<
        "initial seed: " << seed_num << endl <<
        "epochs: "<< epochs << endl <<
         "learning rate: "  << learning_rate << endl <<
         "regularization parameter: " << regularization_parameter << endl <<
         "training data: " << training_size << endl <<
         "test data: " << test_size  << endl <<
         "range scale factor: " << range_scale_factor << endl <<
         "comment: " << comment << endl;
    data_info.close();
}

void Strassen_NN::save_errors() const
{
    const string in_error_file_name = instance_path + "in_sample_error.dat";
    in_sample_error.save(in_error_file_name, raw_ascii);

    const string out_error_file_name = instance_path + "out_sample_error.dat";
    out_sample_error.save(out_error_file_name, raw_ascii);
}


void Strassen_NN::save_weights(int n)
{
    const string file_name1A = instance_path + "W1A_epoch" + to_string(n) + ".dat";
    const string file_name1B = instance_path + "W1B_epoch" + to_string(n) + ".dat";
    const string file_name2 = instance_path + "W2_epoch" + to_string(n) + ".dat";

    W_1A.save(file_name1A, raw_ascii);
    W_1B.save(file_name1B, raw_ascii);
    W_2.save(file_name2, raw_ascii);
}


void Strassen_NN::set_optimal_weights_2_2_2()
{
    W_1B =  {
                {1, 0, 0, 0},
                {1, 1, 0, 0},
                {0, 0, 1, 1},
                {0, 0, 0, 1},
                {1, 0, 0, 1},
                {0, 1, 0, -1},
                {1, 0, -1, 0},
            };

    W_1A = {
                {0, 1, 0, -1},
                {0, 0, 0, 1},
                {1, 0, 0, 0},
                {-1, 0, 1, 0},
                {1, 0, 0, 1},
                {0, 0, 1, 1},
                {1, 1, 0, 0},

            };

    W_2 = {
                {0,-1,0,1,1,1,0},
                {1,1,0,0,0,0,0},
                {0,0,1,1,0,0,0},
                {1,0,-1,0,1,0,-1},
          };

}

void Strassen_NN::set_near_optimal_weights_2_2_2()
{
    W_1B =  {
                {1, 0.2, 0, -0.1},
                {1, 1, 0, 0},
                {0.2, 0, 1.3, 1},
                {0, 0, 0, 1},
                {1, 0, 0, 1},
                {0, 1, 0, -1},
                {1.1, 0.3, -1, 0},
            };

    W_1A = {
                {0, 1.1, 0, -1.2},
                {0.3, 0, 0.4, 1},
                {1, 0, 0, 0},
                {-1, 0, 1, 0},
                {1, 0, 0, 1},
                {0, 0.2, 1.3, 1},
                {1.3, 1, 0, -0.2},

            };

    W_2 = {
                {0,-1.3,0,1,1,1,0},
                {1,1,0.2,0.4,0,0,0},
                {0.1,0,1,1,0,0.2,0},
                {1.1,0,-1.2,0,1.3,0,-1},
          };

}
