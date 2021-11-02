from .CPD_tuning import main as CPD_main
from .pretrain_RBN import main as RBN_main
from .evaluate import main as eval_main


def generate_data(plot_data=False, n_samples=5):
    return CPD_main(generate_data=True,
                compute_results=False,
                plot_results=False,
                plot_data=plot_data,
                parallel=False,
                n_cpu=None,
                detailed_progress=False,
                timestamp=None,
                l_sequence=50,
                n_samples=n_samples,
                max_l_sequence=None,
                multi_terminal_mean=5,
                file_suffix=None)


def train_CPD(n_samples=5, compute_results=True, parallel=True):
    return CPD_main(generate_data=False,
                compute_results=compute_results,
                plot_results=True,
                plot_data=False,
                parallel=parallel,
                n_cpu=None,
                detailed_progress=False,
                timestamp=None,
                l_sequence=50,
                n_samples=n_samples,
                max_l_sequence=None,
                multi_terminal_mean=5,
                file_suffix=None)


def train_RBN(terminal_std=None, n_samples=5, max_epochs=10, pre_trained=False, print_params=False, plot_progress=False):
    return RBN_main(l_sequence=50,
                    n_samples=n_samples,
                    multi_terminal_lambda=5,
                    file_suffix=None,
                    terminal_std=terminal_std,  # noise levels: 0.01, 0.05, 0.1, 0.15, 0.2, 0.25
                    pick_first=None,
                    max_epochs=max_epochs,
                    pre_trained=pre_trained,
                    print_params=print_params,
                    plot_progress=plot_progress)


def evaluate(n_samples=5,
             compute_cpd_results=True,
             compute_rbn_results=True,
             RBN_marginal=True,
             rbn_parallel=True,
             plot_data=False,
             n_data_points=None):
    return eval_main(l_sequence=50,
                     n_samples=n_samples,
                     multi_terminal_lambda=5,
                     file_suffix=None,
                     compute_cpd_results=compute_cpd_results,
                     compute_rbn_results=compute_rbn_results,
                     RBN_marginal=RBN_marginal,
                     rbn_parallel=rbn_parallel,
                     n_cpu=None,
                     plot_data=plot_data,
                     n_data_points=n_data_points,
                     detailed_progress=False)