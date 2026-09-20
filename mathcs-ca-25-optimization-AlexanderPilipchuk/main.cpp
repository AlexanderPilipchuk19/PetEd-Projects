#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <omp.h>

#include "hit.h"

namespace {

constexpr int REALIZATION_SINGLE = 1;
constexpr int REALIZATION_OPENMP_AUTO = 2;
constexpr int REALIZATION_OPENMP_MANUAL = 3;
constexpr int DEFAULT_DYNAMIC_CHUNK = 100;

struct Args {
    const char* input_file = nullptr;
    const char* output_file = nullptr;
    long long N = 0;                  // local fallback / backward compatibility
    bool has_explicit_n = false;
    int realization = 0;
    int threads = 0;                  // 0 -> OpenMP default
    const char* kind = nullptr;       // static | dynamic
    int chunk_size = 0;               // 0 -> choose default
};

int parse_positive_long_long(const char* text, long long& value) {
    if (text == nullptr || *text == '\0') {
        return 1;
    }

    errno = 0;
    char* endptr = nullptr;
    const long long parsed = strtoll(text, &endptr, 10);
    if (errno != 0 || endptr == nullptr || *endptr != '\0' || parsed <= 0) {
        return 1;
    }

    value = parsed;
    return 0;
}

int parse_positive_int(const char* text, int& value) {
    long long parsed = 0;
    if (parse_positive_long_long(text, parsed) != 0 || parsed > static_cast<long long>(INT_MAX)) {
        return 1;
    }

    value = static_cast<int>(parsed);
    return 0;
}

int parse_realization(const char* text, int& value) {
    long long parsed = 0;
    if (parse_positive_long_long(text, parsed) != 0) {
        return 1;
    }
    if (parsed < REALIZATION_SINGLE || parsed > REALIZATION_OPENMP_MANUAL) {
        return 1;
    }

    value = static_cast<int>(parsed);
    return 0;
}

int parse_args(int argc, char* argv[], Args& args) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --input requires a file path\n");
                return 1;
            }
            args.input_file = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --output requires a file path\n");
                return 1;
            }
            args.output_file = argv[++i];
        } else if (strcmp(argv[i], "--realization") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --realization requires a value\n");
                return 1;
            }
            if (parse_realization(argv[++i], args.realization) != 0) {
                fprintf(stderr, "Error: --realization must be 1, 2 or 3\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --threads requires a value\n");
                return 1;
            }
            if (parse_positive_int(argv[++i], args.threads) != 0) {
                fprintf(stderr, "Error: threads must be a positive integer\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--kind") == 0 || strcmp(argv[i], "--schedule") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires a value\n", argv[i]);
                return 1;
            }
            args.kind = argv[++i];
            if (strcmp(args.kind, "static") != 0 && strcmp(args.kind, "dynamic") != 0) {
                fprintf(stderr, "Error: kind must be 'static' or 'dynamic'\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--chunk_size") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --chunk_size requires a value\n");
                return 1;
            }
            if (parse_positive_int(argv[++i], args.chunk_size) != 0) {
                fprintf(stderr, "Error: chunk_size must be a positive integer\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--N") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --N requires a value\n");
                return 1;
            }
            if (parse_positive_long_long(argv[++i], args.N) != 0) {
                fprintf(stderr, "Error: N must be a positive integer\n");
                return 1;
            }
            args.has_explicit_n = true;
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (args.output_file == nullptr) {
        fprintf(stderr, "Error: --output is required\n");
        return 1;
    }
    if (args.realization == 0) {
        fprintf(stderr, "Error: --realization is required\n");
        return 1;
    }
    if (args.input_file == nullptr && !args.has_explicit_n) {
        fprintf(stderr, "Error: --input is required\n");
        return 1;
    }

    return 0;
}

int read_n_from_input(const Args& args, long long& n_value) {
    if (args.input_file != nullptr) {
        FILE* input = fopen(args.input_file, "r");
        if (input == nullptr) {
            fprintf(stderr, "Error: Cannot open input file\n");
            return 1;
        }

        char buffer[256] = {0};
        if (fgets(buffer, sizeof(buffer), input) == nullptr) {
            fprintf(stderr, "Error: Cannot read input file\n");
            fclose(input);
            return 1;
        }

        if (fclose(input) != 0) {
            fprintf(stderr, "Error: Cannot close input file\n");
            return 1;
        }

        if (parse_positive_long_long(buffer, n_value) == 0) {
            return 0;
        }

        char* endptr = nullptr;
        errno = 0;
        const long long parsed = strtoll(buffer, &endptr, 10);
        while (endptr != nullptr && (*endptr == ' ' || *endptr == '\t' || *endptr == '\n' || *endptr == '\r')) {
            ++endptr;
        }
        if (errno == 0 && endptr != nullptr && *endptr == '\0' && parsed > 0) {
            n_value = parsed;
            return 0;
        }

        fprintf(stderr, "Error: Input file must contain a positive integer\n");
        return 1;
    }

    n_value = args.N;
    return 0;
}

int write_result(const char* output_file, double volume) {
    FILE* out = fopen(output_file, "w");
    if (out == nullptr) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 1;
    }

    if (fprintf(out, "%g\n", volume) < 0) {
        fprintf(stderr, "Error: Cannot write to output file\n");
        fclose(out);
        return 1;
    }

    if (fclose(out) != 0) {
        fprintf(stderr, "Error: Cannot close output file\n");
        return 1;
    }

    return 0;
}

int get_effective_thread_count(const Args& args) {
    if (args.threads > 0) {
        return args.threads;
    }

    const int max_threads = omp_get_max_threads();
    return max_threads > 0 ? max_threads : 1;
}

omp_sched_t get_schedule_kind(const Args& args) {
    if (args.kind != nullptr && strcmp(args.kind, "dynamic") == 0) {
        return omp_sched_dynamic;
    }
    return omp_sched_static;
}

int get_chunk_size(const Args& args, long long n_value, int thread_count) {
    if (args.chunk_size > 0) {
        return args.chunk_size;
    }

    const omp_sched_t schedule_kind = get_schedule_kind(args);
    if (schedule_kind == omp_sched_dynamic) {
        return DEFAULT_DYNAMIC_CHUNK;
    }

    if (thread_count <= 0) {
        thread_count = 1;
    }
    long long chunk = (n_value + thread_count - 1) / thread_count;
    if (chunk <= 0) {
        chunk = 1;
    }
    if (chunk > static_cast<long long>(INT_MAX)) {
        chunk = INT_MAX;
    }
    return static_cast<int>(chunk);
}

void init_rng_for_thread(int thread_id,
                         float x_min,
                         float x_max,
                         float y_min,
                         float y_max,
                         float z_min,
                         float z_max,
                         std::mt19937& gen,
                         std::uniform_real_distribution<float>& x_dist,
                         std::uniform_real_distribution<float>& y_dist,
                         std::uniform_real_distribution<float>& z_dist) {
    std::random_device rd;
    const unsigned int seed = static_cast<unsigned int>(rd()) ^
                              (0x9E3779B9u + static_cast<unsigned int>(thread_id * 2654435761u));
    gen.seed(seed);
    x_dist = std::uniform_real_distribution<float>(x_min, x_max);
    y_dist = std::uniform_real_distribution<float>(y_min, y_max);
    z_dist = std::uniform_real_distribution<float>(z_min, z_max);
}

long long run_single(long long n_value,
                     float x_min,
                     float x_max,
                     float y_min,
                     float y_max,
                     float z_min,
                     float z_max) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> x_dist;
    std::uniform_real_distribution<float> y_dist;
    std::uniform_real_distribution<float> z_dist;
    init_rng_for_thread(0, x_min, x_max, y_min, y_max, z_min, z_max, gen, x_dist, y_dist, z_dist);

    long long hits = 0;
    for (long long i = 0; i < n_value; ++i) {
        const float x = x_dist(gen);
        const float y = y_dist(gen);
        const float z = z_dist(gen);
        hits += hit_test(x, y, z) ? 1 : 0;
    }
    return hits;
}

long long run_openmp_auto(long long n_value,
                          float x_min,
                          float x_max,
                          float y_min,
                          float y_max,
                          float z_min,
                          float z_max,
                          const Args& args,
                          int& used_threads) {
    if (args.threads > 0) {
        omp_set_num_threads(args.threads);
    }

    const int thread_count = get_effective_thread_count(args);
    const omp_sched_t schedule_kind = get_schedule_kind(args);
    const int chunk_size = get_chunk_size(args, n_value, thread_count);
    omp_set_schedule(schedule_kind, chunk_size);

    long long hits = 0;
    used_threads = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            used_threads = omp_get_num_threads();
        }

        const int thread_id = omp_get_thread_num();
        std::mt19937 gen;
        std::uniform_real_distribution<float> x_dist;
        std::uniform_real_distribution<float> y_dist;
        std::uniform_real_distribution<float> z_dist;
        init_rng_for_thread(thread_id, x_min, x_max, y_min, y_max, z_min, z_max, gen, x_dist, y_dist, z_dist);

        long long local_hits = 0;

        #pragma omp for schedule(runtime) nowait
        for (long long i = 0; i < n_value; ++i) {
            const float x = x_dist(gen);
            const float y = y_dist(gen);
            const float z = z_dist(gen);
            local_hits += hit_test(x, y, z) ? 1 : 0;
        }

        #pragma omp atomic
        hits += local_hits;
    }

    return hits;
}

long long run_openmp_manual(long long n_value,
                            float x_min,
                            float x_max,
                            float y_min,
                            float y_max,
                            float z_min,
                            float z_max,
                            const Args& args,
                            int& used_threads) {
    if (args.threads > 0) {
        omp_set_num_threads(args.threads);
    }

    long long hits = 0;
    long long next_index = 0;
    used_threads = 0;

    #pragma omp parallel shared(next_index)
    {
        #pragma omp single
        {
            used_threads = omp_get_num_threads();
        }

        int num_threads = used_threads;
        if (num_threads <= 0) {
            num_threads = 1;
        }

        const omp_sched_t schedule_kind = get_schedule_kind(args);
        const int chunk_size = get_chunk_size(args, n_value, num_threads);
        const int thread_id = omp_get_thread_num();

        std::mt19937 gen;
        std::uniform_real_distribution<float> x_dist;
        std::uniform_real_distribution<float> y_dist;
        std::uniform_real_distribution<float> z_dist;
        init_rng_for_thread(thread_id, x_min, x_max, y_min, y_max, z_min, z_max, gen, x_dist, y_dist, z_dist);

        long long local_hits = 0;

        if (schedule_kind == omp_sched_static) {
            for (long long block_begin = static_cast<long long>(thread_id) * chunk_size;
                 block_begin < n_value;
                 block_begin += static_cast<long long>(num_threads) * chunk_size) {
                const long long block_end = std::min(block_begin + static_cast<long long>(chunk_size), n_value);
                for (long long i = block_begin; i < block_end; ++i) {
                    const float x = x_dist(gen);
                    const float y = y_dist(gen);
                    const float z = z_dist(gen);
                    local_hits += hit_test(x, y, z) ? 1 : 0;
                }
            }
        } else {
            while (true) {
                long long block_begin = 0;
                #pragma omp atomic capture
                {
                    block_begin = next_index;
                    next_index += chunk_size;
                }

                if (block_begin >= n_value) {
                    break;
                }

                const long long block_end = std::min(block_begin + static_cast<long long>(chunk_size), n_value);
                for (long long i = block_begin; i < block_end; ++i) {
                    const float x = x_dist(gen);
                    const float y = y_dist(gen);
                    const float z = z_dist(gen);
                    local_hits += hit_test(x, y, z) ? 1 : 0;
                }
            }
        }

        #pragma omp atomic
        hits += local_hits;
    }

    return hits;
}

} // namespace

int main(int argc, char* argv[]) {
    Args args;
    if (parse_args(argc, argv, args) != 0) {
        return 1;
    }

    long long n_value = 0;
    if (read_n_from_input(args, n_value) != 0) {
        return 1;
    }

    const float* range = get_axis_range();
    const float x_min = range[0];
    const float x_max = range[1];
    const float y_min = range[2];
    const float y_max = range[3];
    const float z_min = range[4];
    const float z_max = range[5];

    const double x_range = static_cast<double>(x_max) - static_cast<double>(x_min);
    const double y_range = static_cast<double>(y_max) - static_cast<double>(y_min);
    const double z_range = static_cast<double>(z_max) - static_cast<double>(z_min);
    const double volume_box = x_range * y_range * z_range;

    int used_threads = 0;
    long long hits = 0;

    const double start_time = omp_get_wtime();

    if (args.realization == REALIZATION_SINGLE) {
        hits = run_single(n_value, x_min, x_max, y_min, y_max, z_min, z_max);
        used_threads = 0;
    } else if (args.realization == REALIZATION_OPENMP_AUTO) {
        hits = run_openmp_auto(n_value, x_min, x_max, y_min, y_max, z_min, z_max, args, used_threads);
    } else {
        hits = run_openmp_manual(n_value, x_min, x_max, y_min, y_max, z_min, z_max, args, used_threads);
    }

    const double end_time = omp_get_wtime();
    const double elapsed_ms = (end_time - start_time) * 1000.0;

    const double volume = (static_cast<double>(hits) / static_cast<double>(n_value)) * volume_box;

    if (write_result(args.output_file, volume) != 0) {
        return 1;
    }

    printf("Time (%i thread(s)): %g ms\n", used_threads, elapsed_ms);
    return 0;
}
