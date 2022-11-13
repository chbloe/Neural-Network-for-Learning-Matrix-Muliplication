#ifndef STRASSEN_NN_H
#define STRASSEN_NN_H

#include <vector>
#include <armadillo>


class Strassen_NN
{
    public:
        Strassen_NN(std::vector<int>& matrix_dimensions,
                    int rank_estimate,
                    size_t training_size,
                    size_t test_size,
                    int seed_num=0,
                    size_t epochs=1e+3,
                    double learning_rate=1e-2,
                    double regularization_parameter=0.0,
                    double range_scale_factor=1.0,
                    int exp_id=0,
                    double threshold_error_out=1e-4,
                    std::string data_series_path="data",
                    std::string comment="");

        ~Strassen_NN(){}

        double forward_propagation(const arma::mat&, const arma::mat&);
        void backward_propagation();

        void update_weight_matrices();
        void adam_optimization(const arma::vec& delta, const arma::vec& x, arma::mat& W, arma::mat& v_dW, arma::mat& S_dW);
        void momentum(const arma::vec& delta, const arma::vec& x, arma::mat& W, arma::mat& v_dW);

        void initialize_weight_matrices();
        void set_optimal_weights_2_2_2();
        void set_near_optimal_weights_2_2_2();

        void set_epochs(int); /// to try a second, warm start

        void run();
        double test_out_of_sample();

        /// utilities
        void expand_data_range(arma::cube& A, arma::cube& B, double scale, double upper_matrix_element_magnitude_boundary);
        void expand_data_range(arma::cube& A, arma::cube& B, double scale=1.0);

        void display_weight_matrices() const;

        void save_info() const;

        void save_data(int);
        void save_errors() const;
        void save_weights(int);

    private:

        ///dimensions
        std::vector<int> matrix_dimensions;
        int rank_estimate;
        int seed_num;

        size_t epochs;
        size_t training_size;
        size_t test_size;

        static constexpr double epsilon = 1e-8;
        static constexpr double beta_1 = 0.9;
        static constexpr double beta_2 = 0.999;

        double beta_1_t = 1.0;
        double beta_2_t = 1.0;

        /// learning parameters
        double learning_rate;
        double regularization_parameter;
        double weight_decay_factor;
        double threshold_error_out;
        double range_scale_factor;


        /// data locations
        std::string data_series_path;
        std::string instance_path;
        /// comment on experiment
        std::string comment;


        /// input layer
        arma::vec x_0A;
        arma::vec x_0B;
        /// hidden layer
        arma::vec x_1;
        /// hidden and output layer
        arma::vec x_2;

        /// signals
        arma::vec s_1A;
        arma::vec s_1B;

        /// sensitivities
        arma::vec temp;
        arma::vec delta_1A;
        arma::vec delta_1B;
        arma::vec delta_2;

        /// weight matrices
        arma::mat W_1A;
        arma::mat W_1B;
        arma::mat W_2;

        arma::mat v_dW_2;
        arma::mat v_dW_1A;
        arma::mat v_dW_1B;

        arma::mat S_dW_2;
        arma::mat S_dW_1A;
        arma::mat S_dW_1B;

        /// errors
        arma::vec in_sample_error;
        arma::vec out_sample_error;
};

#endif // STRASSEN_NN_H
