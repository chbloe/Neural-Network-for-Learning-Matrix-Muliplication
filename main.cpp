#include <iostream>
#include <vector>
#include <cmath>
#include <armadillo>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <experimental/filesystem>
#include <boost/program_options.hpp>
#include "Strassen_NN.h"


using namespace std;
using namespace arma;
using namespace boost::program_options;
namespace fs = std::experimental::filesystem;




class NotImplemented : public std::logic_error
{
public:
    NotImplemented() : std::logic_error("Function not yet implemented") { };
};


void help()
{
    cout << endl << "Try 'snn -h' or 'snn --help' for more information.\n\n" << endl;
}

/**
 *  define command-line options
 *
 */
 void general_define_options(options_description& general)
{
    general.add_options()
    ("help,h", "display options help")
    ("update-method,u", value<string>(), "select method of weight update. E.g. SGD, SGDM (SDG with momentum), ADAM")
    ("path,p", value<string>(), "directory path to write output")
    ("comment,m", value<string>(), "comment about experiment")
    ("rank,k", value<int>(), "rank of matrix product")
    ("matrix_dimensions,d",  value<vector<int>>()->multitoken(), "matrix matrix_dimensionsensions. Eg. 2 2 2")
    ("seed_init,s", value<int>(), "initial seed for random number generation")
    ("epochs,e",  value<int>(), "number of epochs")
    ("train,x",  value<int>(), "number of training samples")
    ("test,y",  value<int>(), "number of test sample")
    ("exps,n",  value<int>(), "number of repetions per experiment")
    ("threshold_eout,o",  value<double>(), "enable saving weight matrices if value of out-of-sample error is below threshold.")
    ("scale_factor,c",  value<vector<double>>()->multitoken(), "range scale factors for test data. Eg. 1 1e+2")
    ("learning_rate,l", value<vector<double>>()->multitoken(), "learning rates. Eg. 1e-2 1e-3")
    ("reg_param,r", value<vector<double>>()->multitoken(), "regularization parameters. Eg. 1e-2 1e-3")
    ;
}



int main(int argc, char* argv[])
{
    ///----------------------------------------------------------------------//
    ///----------------------------------------------------------------------//

    string data_series_path;
    string comment;
    vector<int> matrix_dimensions {2,2,2}; /// matrix matrix_dimensionsensions
    int rank_estimate = 7;
    int epochs = 5e+3;

    int training_size = 1e+4;
    int test_size = 1e+3;

    int num_experiments = 5;

    double threshold_eout = 1e-8; /// threshold E_out to save weight matrices

    int seed_num = 0; /// control the seed for each experiment

    vector<double> regularization_parameters = {0};
    vector<double> range_scale_factors;
    vector<double> learning_rates;


    /**
        general & combined option definitions
    */
    options_description general("Options");
    general_define_options(general);

    /**
        build options map
    */
    variables_map vm;
    store(command_line_parser(argc, argv).options(general).run(), vm);

     try {

        if ( vm.count("help") )
        {
            cout << endl << general;
            cout << endl;
            exit(EXIT_SUCCESS);
        }
        /// only call here, in order to allow error-free call to --help
        notify(vm);

        /// select weight update method
        if ( vm.count("update-mode") ) {

            NotImplemented(); /// !!!!!

            string update_method = vm["update-mode"].as<string>();

            if (update_method == "sgd") {

            } else if (update_method == "sgdm") {

            } else if (update_method == "adam") {

            } else { /// set default method

            }

        }

        if (vm.count("matrix_dimensions"))
        {
        /// matrix_dimensionsensions of matrices A and B such that A: m*n, and B: n*k
            matrix_dimensions = vm["matrix_dimensions"].as<vector<int>>();

            /// ensure three are given, if only 1, then make all the same,
            if ( matrix_dimensions.size() > 3 ) {
                cerr << "too many input arguments for matrix_dimensions" << endl;
                exit(EXIT_FAILURE);
            }
        }

        if (vm.count("rank")) /// estimated rank of matrix product
        {
            rank_estimate = vm["rank"].as<int>();

            /// SANITY CHECK
            const double lower_limit = std::pow(std::max(matrix_dimensions[0], matrix_dimensions[1]), 2.0);
            //const double upper_limit = std::pow(std::max(matrix_dimensions[0], matrix_dimensions[1]), 2.8074);

            //if ( ( rank_estimate < lower_limit ) || (rank_estimate > upper_limit) )
            if ( rank_estimate < lower_limit  )
            {
                cout << "\nERROR:  Provided unreasonable rank estimate for matrix product." << endl;
                cout << "lower limit: " << lower_limit << endl;
                //cout << "upper limit: " << upper_limit << endl << endl;
                return 0;
            }
        }

        /// training/test data setup
        if (vm.count("epochs"))
        {
            epochs = vm["epochs"].as<int>();
        }

        if (vm.count("train"))
        {
            training_size = vm["train"].as<int>();
        }

        if (vm.count("test"))
        {
            test_size = vm["test"].as<int>();
        }

        ///----------------------------------------------------------------------//
        ///----------------------------------------------------------------------//

        if (vm.count("exps"))
        {
            num_experiments = vm["exps"].as<int>();
        }

        if (vm.count("threshold_eout"))
        {
            /// threshold E_out to save weight matrices
            threshold_eout = vm["threshold_eout"].as<double>();
        }

        if (vm.count("scale_factor"))
        {
            range_scale_factors = vm["scale_factor"].as<vector<double>>();
        }

        if (vm.count("learning_rate"))
        {
            /// learning parameters
            learning_rates = vm["learning_rate"].as<vector<double>>();
        }

        if (vm.count("reg_param"))
        {
            regularization_parameters = vm["reg_param"].as<vector<double>>();
        }

        if (vm.count("seed_init"))
        {
            seed_num = vm["seed_init"].as<int>();
        }

        if (vm.count("path"))
        {
            data_series_path = vm["path"].as<string>();
        } else {

            /// current time and date
            stringstream ss;
            auto now = chrono::system_clock::now();
            auto in_time_t = chrono::system_clock::to_time_t(now);
            ss << put_time(localtime(&in_time_t), "%Y-%m-%d %X");

            /// for the entire series, create parent directory, if it doesnt exist yet
            data_series_path = "DATA matrix_dimensions_" + to_string(matrix_dimensions[0]) + "_" + to_string(matrix_dimensions[1]) + "_" + to_string(matrix_dimensions[2]) +
                            " rank_" + to_string(rank_estimate) +
                            " epochs_" + to_string(epochs) +
                            " " + ss.str() +
                            "/";
        }

        if (vm.count("comment"))
        {
            comment = vm["comment"].as<string>();
        }
    }
    catch(std::exception& e)
    {
        cout << e.what() << endl;
        help();
        exit(EXIT_FAILURE);
    }


    ///----------------------------------------------------------------------//
    ///----------------------------------------------------------------------//

    ///for the entire series, create parent directory
    fs::create_directory(data_series_path);

    for (auto rsf : range_scale_factors) {
        for (auto lr : learning_rates) {
            for (auto rp : regularization_parameters) {

                /// run multiple repetitions for the combination of experimental parameters {rsf, lr, rp}
                for (int exp_id = 0; exp_id < num_experiments; ++exp_id) {


                    seed_num++; /// control different starts

                    /// initialize the neural network
                    Strassen_NN snn(matrix_dimensions,
                                    rank_estimate,
                                    training_size,
                                    test_size,
                                    seed_num,
                                    epochs,
                                    lr,
                                    rp,
                                    rsf,
                                    exp_id,
                                    threshold_eout,
                                    data_series_path,
                                    comment);

                    /// train the network
                    snn.run();
                }
            }
        }
    }

    return 0;
}
