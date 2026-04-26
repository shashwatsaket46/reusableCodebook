#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
#include <array>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "defines.hpp"
#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"

static constexpr std::array<size_t, 6> EVAL_KS = {1, 2, 4, 8, 16, 32};
static constexpr size_t EVAL_KMAX = 32;

size_t TOPK = 32;
size_t ROUND = 3;

int main(int argc, char* argv[]) {
    assert(argc == 3);
    char* DATASET = argv[1];
    int B = atoi(argv[2]);
    assert(B == 9 || B == 5 || B == 7 || B == 3 || B == 4 || B == 8 || B == 2);

    char data_file[500];
    char query_file[500];
    char gt_file[500];
    char ivf_file[500];
    char result_file[500];

    sprintf(data_file, "../data/%s/%s_base.fvecs", DATASET, DATASET);
    sprintf(query_file, "../data/%s/%s_query.fvecs", DATASET, DATASET);
    sprintf(gt_file, "../data/%s/%s_groundtruth.ivecs", DATASET, DATASET);
    sprintf(ivf_file, "../data/%s/ivf_exhaf%d.index", DATASET, B);
    sprintf(result_file, "../results/exrabitq/%s_exhaf%d.csv", DATASET, B);

    FloatRowMat data;
    FloatRowMat query;
    UintRowMat gt;

    load_vecs<float, FloatRowMat>(data_file, data);
    load_vecs<float, FloatRowMat>(query_file, query);
    load_vecs<PID, UintRowMat>(gt_file, gt);

    size_t N = data.rows();
    size_t DIM = data.cols();
    size_t NQ = query.rows();

    std::cout << "data loaded\n";
    std::cout << "\tN: " << N << '\n' << "\tDIM: " << DIM << '\n';
    std::cout << "query loaded\n";
    std::cout << "\tNQ: " << NQ << '\n';

    IVF ivf;
    ivf.load(ivf_file);

    // Exhaustive: probe every cluster (nprobe = nlist)
    std::vector<size_t> nprobes = {ivf.k()};
    size_t length = nprobes.size();

    StopW stopw;

    FloatRowMat padded_query(NQ, ivf.padded_dim());
    padded_query.setZero();
    FloatRowMat rotated_query(NQ, ivf.padded_dim());
    for (size_t i = 0; i < NQ; ++i) {
        std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
    }
    Rotator& rp = ivf.rotator();
    stopw.reset();
    rp.rotate(padded_query, rotated_query);
    float rotate_time = stopw.getElapsedTimeMicro();

    size_t total_count = NQ * TOPK;

    std::vector<std::vector<float>> all_qps(ROUND, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(ROUND, std::vector<float>(length));
    std::vector<std::vector<float>> all_ratio(ROUND, std::vector<float>(length));

    // per-k hit counters [ROUND][length][EVAL_KS.size()]
    std::vector<std::vector<std::vector<size_t>>> all_hits(
        ROUND, std::vector<std::vector<size_t>>(
            length, std::vector<size_t>(EVAL_KS.size(), 0)));

    for (size_t r = 0; r < ROUND; r++) {
        for (size_t i = 0; i < length; ++i) {
            size_t nprobe = nprobes[i];
            size_t total_correct = 0;
            double total_ratio = 0;
            float total_time = 0;
            PID results[TOPK];

            std::fill(all_hits[r][i].begin(), all_hits[r][i].end(), 0);

            for (size_t qi = 0; qi < NQ; qi++) {
                stopw.reset();
                ivf.search(&rotated_query(qi, 0), data.data(), TOPK, nprobe, results);
                total_time += stopw.getElapsedTimeMicro();
                total_ratio += get_ratio(qi, query, data, gt, results, TOPK, L2Sqr);

                // standard recall (top-TOPK in top-TOPK GT)
                for (size_t j = 0; j < TOPK; j++) {
                    for (size_t kk = 0; kk < TOPK; kk++) {
                        if (gt(qi, kk) == results[j]) {
                            total_correct++;
                            break;
                        }
                    }
                }

                // per-k recall: is top-1 GT in top-k results?
                PID top1_gt = gt(qi, 0);
                for (size_t ki = 0; ki < EVAL_KS.size(); ki++) {
                    size_t k = EVAL_KS[ki];
                    for (size_t j = 0; j < k && j < TOPK; j++) {
                        if (results[j] == top1_gt) {
                            all_hits[r][i][ki]++;
                            break;
                        }
                    }
                }
            }

            float qps = NQ / ((total_time + rotate_time) / 1e6);
            float recall = static_cast<float>(total_correct) / total_count;
            float ratio = total_ratio / total_count;

            all_qps[r][i] = qps;
            all_recall[r][i] = recall;
            all_ratio[r][i] = ratio;
        }
    }

    auto avg_qps = horizontal_avg(all_qps);
    auto avg_recall = horizontal_avg(all_recall);
    auto avg_ratio = horizontal_avg(all_ratio);

    std::ofstream csv_data(result_file, std::ios::out);
    csv_data << "nprobe,QPS,recall,ratio" << std::endl;

    for (size_t i = 0; i < length; ++i) {
        size_t nprobe = nprobes[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];
        float ratio = avg_ratio[i];

        csv_data << nprobe << ',';
        csv_data << qps << ',';
        csv_data << recall << ',';
        csv_data << ratio << '\n';

        // Emit per-k recall lines (averaged over rounds)
        for (size_t ki = 0; ki < EVAL_KS.size(); ki++) {
            size_t k = EVAL_KS[ki];
            double hit_sum = 0;
            for (size_t r = 0; r < ROUND; r++) {
                hit_sum += all_hits[r][i][ki];
            }
            float rec_k = static_cast<float>(hit_sum) / (ROUND * NQ);
            std::cout << "EVAL bits=" << B
                      << " nprobe=" << nprobe
                      << " k=" << k
                      << " recall=" << rec_k
                      << std::endl;
        }
    }
    csv_data.close();

    return 0;
}