#include <cstring>
#include "numpy.h"
// #include "coupled_states.cu"
#include "coupled_states_dev.cu"

void set_ham(std::string &ham_file) {
    npy_cpp::NumpyCpp np;
    np.loadz(ham_file);
    printf("Load ham file from %s\n", ham_file.c_str());
    int32 n_qubits = np["n_qubits"].getDataPtr<int32>()[0];
    int64 K = np["K"].getDataPtr<int64>()[0];
    int64 NK = np["NK"].getDataPtr<int64>()[0];
 
    int64 *idxs = np["idxs"].getDataPtr<int64>();
    coeff_dtype *coeffs = np["coeffs"].getDataPtr<coeff_dtype>();
    int8 *pauli_mat12 = np["pauli_mat12"].getDataPtr<int8>();
    int8 *pauli_mat23 = np["pauli_mat23"].getDataPtr<int8>();

    set_indices_ham_int_opt(
        n_qubits,
        K,
        NK,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23);
}

/*void test(std::string &ham_file, std::string &data_file) {
    set_ham(ham_file);

    npy_cpp::NumpyCpp np;
    np.loadz(data_file);
    printf("Load input data file from %s\n", data_file.c_str());

    int64 batch_size = np["batch_size"].getDataPtr<int64>()[0];
    int64 *_states = np["_states"].getDataPtr<int64>();
    int64 ist = np["ist"].getDataPtr<int64>()[0];
    int64 ied = np["ied"].getDataPtr<int64>()[0];
    uint64 *ks = np["ks"].getDataPtr<uint64>();
    psi_dtype *vs = np["vs"].getDataPtr<psi_dtype>();
    int64 rank = np["rank"].getDataPtr<int64>()[0];
    float64 eps = np["eps"].getDataPtr<float64>()[0];
    size_t size_res_eloc_batch = np["res_eloc_batch"].getLength();
    psi_dtype *res_eloc_batch = (psi_dtype *)malloc(size_res_eloc_batch);
    // printf("size_res_eloc_batch: %d\n", size_res_eloc_batch);
    // printf("batch_size: %d ist: %d ied: %d rank: %d eps: %.13lf\n", batch_size, ist, ied, rank, eps);
    // npy_cpp::printArray<int64>(np["_states"], "_states");

    calculate_local_energy(
        batch_size,
        _states,
        ist,
        ied,
        nullptr,
        ks,
        vs,
        rank,
        eps,
        res_eloc_batch);

    npy_cpp::check(np["res_eloc_batch"], res_eloc_batch, size_res_eloc_batch, "res_eloc_batch");
    free(res_eloc_batch);
}*/

void test_bigInt(std::string &ham_file, std::string &data_file) {
    set_ham(ham_file);

    npy_cpp::NumpyCpp np;
    np.loadz(data_file);
    printf("Load input data file from %s\n", data_file.c_str());

    int64 batch_size = np["batch_size"].getDataPtr<int64>()[0];
    int64 *_states = np["_states"].getDataPtr<int64>();
    int64 ist = np["ist"].getDataPtr<int64>()[0];
    int64 ied = np["ied"].getDataPtr<int64>()[0];
    uint64 *ks = np["ks"].getDataPtr<uint64>();
    psi_dtype *vs = np["vs"].getDataPtr<psi_dtype>();
    int64 rank = np["rank"].getDataPtr<int64>()[0];
    float64 eps = np["eps"].getDataPtr<float64>()[0];
    size_t size_res_eloc_batch = np["res_eloc_batch"].getLength();
    psi_dtype *res_eloc_batch = (psi_dtype *)malloc(size_res_eloc_batch);
    // printf("size_res_eloc_batch: %d\n", size_res_eloc_batch);
    // printf("batch_size: %d ist: %d ied: %d rank: %d eps: %.13lf\n", batch_size, ist, ied, rank, eps);
    // npy_cpp::printArray<int64>(np["_states"], "_states");

    int64 all_batch_size = np["all_batch_size"].getDataPtr<int64>()[0];
    int64 id_width = np["id_width"].getDataPtr<int64>()[0];
    int64 ks_disp_idx = np["ks_disp_idx"].getDataPtr<int64>()[0];

    calculate_local_energy_sampling_parallel_bigInt(
        all_batch_size,
        batch_size,
        _states,
        ist,
        ied,
        ks_disp_idx,
        ks,
        id_width,
        vs,
        rank,
        eps,
        res_eloc_batch);

    npy_cpp::check(np["res_eloc_batch"], res_eloc_batch, size_res_eloc_batch, "res_eloc_batch");
    free(res_eloc_batch);
}

int main() {
    // std::string molecule_name = "testcases/h2";
    // std::string molecule_name = "testcases/lih";
    // std::string molecule_name = "testcases/li2o";
    // std::string molecule_name = "testcases/c2h4o";
    // std::string molecule_name = "testcases/c3h6";
    // std::string molecule_name = "testcases/c2h4o2";
    // std::string molecule_name = "testcases/h2o4s";
    std::string molecule_name = "testcases/cna2o3_bigInt";
    std::string ham_file = molecule_name + ".ham";
    std::string data_file = molecule_name + ".indata";
    // test(ham_file, data_file);
    test_bigInt(ham_file, data_file);
    return 0;
}