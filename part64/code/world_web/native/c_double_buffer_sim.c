#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CDB_FLAG_NEXUS 0x1u
#define CDB_FLAG_CHAOS 0x2u
#define CDB_FLAG_ACTIVE 0x4u

typedef struct Vec2DoubleBuffer {
    float *x[2];
    float *y[2];
    _Atomic int readable;
} Vec2DoubleBuffer;

typedef struct CDBEngine {
    uint32_t particle_count;
    uint32_t seed;
    _Atomic int running;
    _Atomic uint64_t frame_id;
    _Atomic uint64_t force_frame;
    _Atomic uint64_t chaos_frame;
    _Atomic uint64_t semantic_frame;

    Vec2DoubleBuffer position;
    Vec2DoubleBuffer velocity;
    Vec2DoubleBuffer acceleration;
    Vec2DoubleBuffer noise;
    Vec2DoubleBuffer action_prob;
    Vec2DoubleBuffer particle_metrics;

    uint32_t *owner_id;
    uint32_t *group_id;
    uint32_t *flags;
    float *mass;
    float *radius;

    uint32_t force_sleep_us;
    uint32_t chaos_sleep_us;
    uint32_t integrate_sleep_us;
    uint32_t semantic_sleep_us;

    pthread_t force_thread;
    pthread_t chaos_thread;
    pthread_t semantic_thread;
    pthread_t integrate_thread;
} CDBEngine;

void cdb_engine_destroy(CDBEngine *engine);

static uint32_t lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float lcg_unit(uint32_t *state) {
    return (float)(lcg_next(state) & 0x00FFFFFFu) / 16777216.0f;
}

static uint32_t env_u32(const char *name, uint32_t fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    if (parsed > 10000000ul) {
        return 10000000u;
    }
    return (uint32_t)parsed;
}

static float clamp01(float value) {
    if (value <= 0.0f) {
        return 0.0f;
    }
    if (value >= 1.0f) {
        return 1.0f;
    }
    return value;
}

uint32_t cdb_growth_guard_scores(
    const float *importance,
    const uint32_t *layer_counts,
    const uint8_t *has_collection,
    const uint8_t *recent_hit,
    uint32_t count,
    float *out_scores
) {
    if (out_scores == NULL || count == 0u) {
        return 0u;
    }

    for (uint32_t i = 0; i < count; i += 1) {
        float importance_value = 0.25f;
        float layer_ratio = 0.0f;
        float collection_bonus = 0.0f;
        float recent_bonus = 0.0f;

        if (importance != NULL) {
            importance_value = clamp01(importance[i]);
        }
        if (layer_counts != NULL) {
            layer_ratio = clamp01(((float)layer_counts[i]) / 4.0f);
        }
        if (has_collection != NULL && has_collection[i] != 0u) {
            collection_bonus = 0.08f;
        }
        if (recent_hit != NULL && recent_hit[i] != 0u) {
            recent_bonus = 0.34f;
        }

        out_scores[i] = clamp01(
            (importance_value * 0.56f)
            + (layer_ratio * 0.2f)
            + collection_bonus
            + recent_bonus
        );
    }
    return count;
}

int cdb_growth_guard_pressure(
    uint32_t file_count,
    uint32_t edge_count,
    uint32_t crawler_count,
    uint32_t item_count,
    uint32_t sim_point_budget,
    uint32_t queue_pending_count,
    uint32_t queue_event_count,
    float cpu_utilization,
    float weaver_graph_node_limit,
    float watch_threshold,
    float critical_threshold,
    float *out_blend,
    float *out_point_ratio,
    float *out_file_ratio,
    float *out_edge_ratio,
    float *out_crawler_ratio,
    float *out_queue_ratio,
    float *out_resource_ratio,
    uint32_t *out_target_file_nodes,
    uint32_t *out_target_edge_count,
    uint32_t *out_mode
) {
    float budget = (sim_point_budget == 0u) ? 1.0f : (float)sim_point_budget;
    float target_file_nodes_f = fmaxf(96.0f, budget * 0.36f);
    uint32_t target_file_nodes = (uint32_t)target_file_nodes_f;
    if (target_file_nodes < 96u) {
        target_file_nodes = 96u;
    }
    if (target_file_nodes > 256u) {
        target_file_nodes = 256u;
    }

    if (cpu_utilization >= 88.0f) {
        uint32_t scaled = (uint32_t)(((float)target_file_nodes) * 0.72f);
        target_file_nodes = (scaled < 72u) ? 72u : scaled;
    } else if (cpu_utilization >= 78.0f) {
        uint32_t scaled = (uint32_t)(((float)target_file_nodes) * 0.84f);
        target_file_nodes = (scaled < 84u) ? 84u : scaled;
    }

    uint32_t target_edge_count = (uint32_t)(((float)target_file_nodes) * 3.0f);
    if (target_edge_count < 240u) {
        target_edge_count = 240u;
    }

    float queue_ratio = clamp01(
        (((float)queue_pending_count) + (((float)queue_event_count) * 0.25f)) / 16.0f
    );
    float resource_ratio = clamp01(cpu_utilization / 100.0f);
    float point_ratio = clamp01(((float)item_count) / fmaxf(1.0f, budget));
    float file_ratio = clamp01(((float)file_count) / fmaxf(1.0f, (float)target_file_nodes));
    float edge_ratio = clamp01(((float)edge_count) / fmaxf(1.0f, (float)target_edge_count));
    float crawler_ratio = clamp01(
        ((float)crawler_count) / fmaxf(1.0f, weaver_graph_node_limit)
    );
    float blend = clamp01(
        (file_ratio * 0.48f)
        + (edge_ratio * 0.24f)
        + (point_ratio * 0.14f)
        + (queue_ratio * 0.08f)
        + (resource_ratio * 0.06f)
    );

    uint32_t mode = 0u;
    if (blend >= critical_threshold) {
        mode = 2u;
    } else if (blend >= watch_threshold) {
        mode = 1u;
    }

    if (out_blend != NULL) {
        *out_blend = blend;
    }
    if (out_point_ratio != NULL) {
        *out_point_ratio = point_ratio;
    }
    if (out_file_ratio != NULL) {
        *out_file_ratio = file_ratio;
    }
    if (out_edge_ratio != NULL) {
        *out_edge_ratio = edge_ratio;
    }
    if (out_crawler_ratio != NULL) {
        *out_crawler_ratio = crawler_ratio;
    }
    if (out_queue_ratio != NULL) {
        *out_queue_ratio = queue_ratio;
    }
    if (out_resource_ratio != NULL) {
        *out_resource_ratio = resource_ratio;
    }
    if (out_target_file_nodes != NULL) {
        *out_target_file_nodes = target_file_nodes;
    }
    if (out_target_edge_count != NULL) {
        *out_target_edge_count = target_edge_count;
    }
    if (out_mode != NULL) {
        *out_mode = mode;
    }
    return 0;
}

int cdb_graph_runtime_maps(
    uint32_t node_count,
    uint32_t edge_count,
    const uint32_t *edge_src,
    const uint32_t *edge_dst,
    const float *edge_affinity,
    float queue_ratio,
    float cpu_ratio,
    float cost_w_l,
    float cost_w_c,
    float cost_w_s,
    const uint32_t *source_nodes,
    const float *source_mass,
    const float *source_need,
    uint32_t source_count,
    float radius_cost,
    float gravity_const,
    float epsilon,
    float *out_min_dist,
    float *out_gravity,
    float *out_edge_cost,
    float *out_node_saturation,
    float *out_node_price
) {
    if (node_count == 0u) {
        return -1;
    }
    if (edge_count > 0u && (edge_src == NULL || edge_dst == NULL)) {
        return -1;
    }
    if (source_count > 0u && source_nodes == NULL) {
        return -1;
    }

    float wl = (cost_w_l > 0.0f) ? cost_w_l : 1.0f;
    float wc = (cost_w_c >= 0.0f) ? cost_w_c : 2.0f;
    float ws = (cost_w_s >= 0.0f) ? cost_w_s : 1.0f;
    float bounded_radius = (radius_cost > 0.0f) ? radius_cost : 6.0f;
    float grav_const = (gravity_const > 0.0f) ? gravity_const : 1.0f;
    float grav_eps = (epsilon > 0.0f) ? epsilon : 0.001f;
    float queue_01 = clamp01(queue_ratio);
    float cpu_01 = clamp01(cpu_ratio);
    float global_sat = clamp01((queue_01 * 0.62f) + (cpu_01 * 0.38f));

    uint32_t *out_degree = (uint32_t *)calloc((size_t)node_count, sizeof(uint32_t));
    uint32_t *in_degree = (uint32_t *)calloc((size_t)node_count, sizeof(uint32_t));
    float *node_sat_sum = (float *)calloc((size_t)node_count, sizeof(float));
    uint32_t *node_sat_count = (uint32_t *)calloc((size_t)node_count, sizeof(uint32_t));
    if (
        out_degree == NULL
        || in_degree == NULL
        || node_sat_sum == NULL
        || node_sat_count == NULL
    ) {
        free(out_degree);
        free(in_degree);
        free(node_sat_sum);
        free(node_sat_count);
        return -1;
    }

    float *edge_cost_cache = out_edge_cost;
    if (edge_cost_cache == NULL && edge_count > 0u) {
        edge_cost_cache = (float *)calloc((size_t)edge_count, sizeof(float));
        if (edge_cost_cache == NULL) {
            free(out_degree);
            free(in_degree);
            free(node_sat_sum);
            free(node_sat_count);
            return -1;
        }
    }

    for (uint32_t node = 0u; node < node_count; node += 1u) {
        if (out_min_dist != NULL) {
            out_min_dist[node] = FLT_MAX;
        }
        if (out_gravity != NULL) {
            out_gravity[node] = 0.0f;
        }
        if (out_node_saturation != NULL) {
            out_node_saturation[node] = global_sat;
        }
        if (out_node_price != NULL) {
            out_node_price[node] = 1.0f;
        }
    }

    for (uint32_t edge = 0u; edge < edge_count; edge += 1u) {
        uint32_t src = edge_src[edge];
        uint32_t dst = edge_dst[edge];
        if (src >= node_count || dst >= node_count) {
            continue;
        }
        out_degree[src] += 1u;
        in_degree[dst] += 1u;
    }

    float mean_degree = sqrtf((((float)edge_count) / fmaxf(1.0f, (float)node_count)) + 1.0f);
    float degree_norm = fmaxf(1.0f, mean_degree * 2.0f);

    for (uint32_t edge = 0u; edge < edge_count; edge += 1u) {
        float affinity = 0.5f;
        float sat = global_sat;
        float edge_cost = fmaxf(0.0001f, wl + (wc * sat) + (ws * (1.0f - affinity)));

        uint32_t src = edge_src[edge];
        uint32_t dst = edge_dst[edge];
        if (src < node_count && dst < node_count) {
            if (edge_affinity != NULL) {
                affinity = clamp01(edge_affinity[edge]);
            }
            float degree_pressure = clamp01(
                (((float)out_degree[src]) + ((float)in_degree[dst])) / degree_norm
            );
            sat = clamp01((global_sat * 0.58f) + (degree_pressure * 0.42f));
            edge_cost = fmaxf(0.0001f, wl + (wc * sat) + (ws * (1.0f - affinity)));

            node_sat_sum[src] += sat;
            node_sat_count[src] += 1u;
            node_sat_sum[dst] += sat;
            node_sat_count[dst] += 1u;
        }

        if (edge_cost_cache != NULL) {
            edge_cost_cache[edge] = edge_cost;
        }
    }

    if ((out_gravity != NULL || out_min_dist != NULL) && source_count > 0u) {
        float *dist = (float *)calloc((size_t)node_count, sizeof(float));
        uint8_t *visited = (uint8_t *)calloc((size_t)node_count, sizeof(uint8_t));
        if (dist == NULL || visited == NULL) {
            free(dist);
            free(visited);
            if (out_edge_cost == NULL) {
                free(edge_cost_cache);
            }
            free(out_degree);
            free(in_degree);
            free(node_sat_sum);
            free(node_sat_count);
            return -1;
        }

        for (uint32_t source_index = 0u; source_index < source_count; source_index += 1u) {
            uint32_t source = source_nodes[source_index];
            if (source >= node_count) {
                continue;
            }
            float need = source_need != NULL ? clamp01(source_need[source_index]) : 1.0f;
            float mass = source_mass != NULL ? fmaxf(0.0f, source_mass[source_index]) : 1.0f;
            if (need <= 1e-6f || mass <= 1e-6f) {
                continue;
            }

            for (uint32_t node = 0u; node < node_count; node += 1u) {
                dist[node] = FLT_MAX;
                visited[node] = 0u;
            }
            dist[source] = 0.0f;

            for (uint32_t iter = 0u; iter < node_count; iter += 1u) {
                uint32_t best_node = node_count;
                float best_dist = bounded_radius + 1.0f;

                for (uint32_t node = 0u; node < node_count; node += 1u) {
                    if (visited[node] != 0u) {
                        continue;
                    }
                    float candidate = dist[node];
                    if (!isfinite(candidate)) {
                        continue;
                    }
                    if (candidate < best_dist) {
                        best_dist = candidate;
                        best_node = node;
                    }
                }

                if (best_node >= node_count || best_dist > bounded_radius) {
                    break;
                }

                visited[best_node] = 1u;
                for (uint32_t edge = 0u; edge < edge_count; edge += 1u) {
                    if (edge_src[edge] != best_node) {
                        continue;
                    }
                    uint32_t dst = edge_dst[edge];
                    if (dst >= node_count) {
                        continue;
                    }
                    float step = (edge_cost_cache != NULL)
                        ? edge_cost_cache[edge]
                        : fmaxf(0.0001f, wl + (wc * global_sat) + (ws * 0.5f));
                    float next_dist = best_dist + step;
                    if (next_dist > bounded_radius || next_dist >= dist[dst]) {
                        continue;
                    }
                    dist[dst] = next_dist;
                }
            }

            for (uint32_t node = 0u; node < node_count; node += 1u) {
                float distance = dist[node];
                if (!isfinite(distance) || distance < 0.0f || distance > bounded_radius) {
                    continue;
                }
                if (out_min_dist != NULL && distance < out_min_dist[node]) {
                    out_min_dist[node] = distance;
                }
                if (out_gravity != NULL) {
                    float potential = (grav_const * mass) / ((distance * distance) + grav_eps);
                    if (isfinite(potential) && potential > 0.0f) {
                        out_gravity[node] += need * potential;
                    }
                }
            }
        }

        free(dist);
        free(visited);
    }

    float max_gravity = 0.0f;
    if (out_gravity != NULL) {
        for (uint32_t node = 0u; node < node_count; node += 1u) {
            float gravity = out_gravity[node];
            if (isfinite(gravity) && gravity > max_gravity) {
                max_gravity = gravity;
            }
        }
    }
    if (max_gravity <= 1e-6f) {
        max_gravity = 1.0f;
    }

    for (uint32_t node = 0u; node < node_count; node += 1u) {
        if (out_min_dist != NULL) {
            if (!isfinite(out_min_dist[node]) || out_min_dist[node] >= FLT_MAX * 0.5f) {
                out_min_dist[node] = -1.0f;
            }
        }

        float sat = global_sat;
        if (node_sat_count[node] > 0u) {
            sat = clamp01(node_sat_sum[node] / (float)node_sat_count[node]);
        }
        if (out_node_saturation != NULL) {
            out_node_saturation[node] = sat;
        }

        if (out_node_price != NULL) {
            float gravity = out_gravity != NULL ? out_gravity[node] : 0.0f;
            if (!isfinite(gravity) || gravity < 0.0f) {
                gravity = 0.0f;
            }
            float pressure_hat = clamp01(gravity / max_gravity);
            float price = expf(1.2f * pressure_hat) * (1.0f + (0.35f * sat));
            if (!isfinite(price) || price <= 0.0f) {
                price = 1.0f;
            }
            if (price > 32.0f) {
                price = 32.0f;
            }
            out_node_price[node] = price;
        }
    }

    if (out_edge_cost == NULL) {
        free(edge_cost_cache);
    }
    free(out_degree);
    free(in_degree);
    free(node_sat_sum);
    free(node_sat_count);
    return 0;
}

int cdb_graph_route_step(
    uint32_t node_count,
    uint32_t edge_count,
    const uint32_t *edge_src,
    const uint32_t *edge_dst,
    const float *edge_cost,
    const float *node_gravity,
    const uint32_t *particle_current_node,
    uint32_t particle_count,
    float eta,
    float upsilon,
    float temperature,
    uint32_t step_seed,
    uint32_t *out_next_node,
    float *out_drift_score,
    float *out_route_probability
) {
    if (
        node_count == 0u
        || particle_count == 0u
        || particle_current_node == NULL
        || out_next_node == NULL
    ) {
        return -1;
    }
    if (edge_count > 0u && (edge_src == NULL || edge_dst == NULL)) {
        return -1;
    }

    float eta_gain = (eta > 0.0f) ? eta : 1.0f;
    float cost_gain = (upsilon > 0.0f) ? upsilon : 0.8f;
    float temp = (temperature > 0.05f) ? temperature : 0.35f;

    for (uint32_t particle = 0u; particle < particle_count; particle += 1u) {
        uint32_t current = particle_current_node[particle];
        if (current >= node_count) {
            current = current % node_count;
        }

        float gravity_here = 0.0f;
        if (node_gravity != NULL) {
            gravity_here = fmaxf(0.0f, node_gravity[current]);
        }

        uint32_t best_node = current;
        float best_score = -FLT_MAX;
        float second_score = -FLT_MAX;
        uint32_t candidate_count = 0u;

        for (uint32_t edge = 0u; edge < edge_count; edge += 1u) {
            if (edge_src[edge] != current) {
                continue;
            }
            uint32_t dst = edge_dst[edge];
            if (dst >= node_count) {
                continue;
            }

            float gravity_next = 0.0f;
            if (node_gravity != NULL) {
                gravity_next = fmaxf(0.0f, node_gravity[dst]);
            }
            float cost = 1.0f;
            if (edge_cost != NULL) {
                cost = fmaxf(0.0001f, edge_cost[edge]);
            }

            float score = (eta_gain * (gravity_next - gravity_here)) - (cost_gain * cost);
            candidate_count += 1u;

            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_node = dst;
            } else if (score > second_score) {
                second_score = score;
            }
        }

        uint32_t mix = step_seed
            ^ ((particle + 1u) * 747796405u)
            ^ ((current + 1u) * 2891336453u);
        mix ^= mix >> 16;
        mix *= 2246822519u;
        mix ^= mix >> 13;
        mix *= 3266489917u;
        mix ^= mix >> 16;
        float random_01 = (float)(mix & 0x00FFFFFFu) / 16777215.0f;

        if (candidate_count == 0u) {
            uint32_t fallback_node = current;
            float fallback_prob = 0.12f;
            if (step_seed != 0u && node_count > 1u && random_01 > 0.86f) {
                uint32_t hop = 1u + (mix % (node_count - 1u));
                fallback_node = (current + hop) % node_count;
                fallback_prob = 0.24f;
            }
            out_next_node[particle] = fallback_node;
            if (out_drift_score != NULL) {
                out_drift_score[particle] = 0.0f;
            }
            if (out_route_probability != NULL) {
                out_route_probability[particle] = fallback_prob;
            }
            continue;
        }

        if (second_score <= -FLT_MAX * 0.5f) {
            second_score = best_score - 1.0f;
        }
        float margin = best_score - second_score;

        float explore_chance = 0.0f;
        if (best_score < -0.12f) {
            explore_chance += 0.28f;
        }
        if (margin < 0.10f) {
            explore_chance += 0.18f;
        }
        if (step_seed != 0u && candidate_count > 1u && random_01 < fminf(0.65f, explore_chance)) {
            uint32_t target_rank = (mix >> 8) % candidate_count;
            uint32_t seen_rank = 0u;
            uint32_t explore_node = best_node;
            float explore_score = best_score;

            for (uint32_t edge = 0u; edge < edge_count; edge += 1u) {
                if (edge_src[edge] != current) {
                    continue;
                }
                uint32_t dst = edge_dst[edge];
                if (dst >= node_count) {
                    continue;
                }
                float gravity_next = 0.0f;
                if (node_gravity != NULL) {
                    gravity_next = fmaxf(0.0f, node_gravity[dst]);
                }
                float cost = 1.0f;
                if (edge_cost != NULL) {
                    cost = fmaxf(0.0001f, edge_cost[edge]);
                }
                float score = (eta_gain * (gravity_next - gravity_here)) - (cost_gain * cost);
                if (seen_rank == target_rank) {
                    explore_node = dst;
                    explore_score = score;
                    break;
                }
                seen_rank += 1u;
            }
            best_node = explore_node;
            best_score = explore_score;
        }

        float route_probability = 1.0f / (1.0f + expf(-(margin / temp)));
        route_probability = clamp01(route_probability);
        float drift_score = tanhf(best_score * 0.45f);

        out_next_node[particle] = best_node;
        if (out_drift_score != NULL) {
            out_drift_score[particle] = drift_score;
        }
        if (out_route_probability != NULL) {
            out_route_probability[particle] = route_probability;
        }
    }
    return 0;
}

static void sleep_microseconds(uint32_t microseconds) {
    if (microseconds == 0u) {
        return;
    }

    struct timespec req;
    req.tv_sec = (time_t)(microseconds / 1000000u);
    req.tv_nsec = (long)((microseconds % 1000000u) * 1000u);

    while (nanosleep(&req, &req) != 0) {
        if (errno != EINTR) {
            break;
        }
    }
}

static int vec2_buffer_init(Vec2DoubleBuffer *buffer, uint32_t count) {
    if (buffer == NULL || count == 0) {
        return -1;
    }
    memset(buffer, 0, sizeof(*buffer));
    for (int index = 0; index < 2; index += 1) {
        buffer->x[index] = (float *)calloc((size_t)count, sizeof(float));
        buffer->y[index] = (float *)calloc((size_t)count, sizeof(float));
        if (buffer->x[index] == NULL || buffer->y[index] == NULL) {
            return -1;
        }
    }
    atomic_store_explicit(&buffer->readable, 0, memory_order_release);
    return 0;
}

static void vec2_buffer_destroy(Vec2DoubleBuffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    for (int index = 0; index < 2; index += 1) {
        free(buffer->x[index]);
        free(buffer->y[index]);
        buffer->x[index] = NULL;
        buffer->y[index] = NULL;
    }
}

static void *force_system_worker(void *arg) {
    CDBEngine *engine = (CDBEngine *)arg;
    if (engine == NULL) {
        return NULL;
    }

    while (atomic_load_explicit(&engine->running, memory_order_acquire) != 0) {
        int pos_index = atomic_load_explicit(&engine->position.readable, memory_order_acquire);
        int vel_index = atomic_load_explicit(&engine->velocity.readable, memory_order_acquire);
        int write_index =
            1 - atomic_load_explicit(&engine->acceleration.readable, memory_order_relaxed);

        const float *pos_x = engine->position.x[pos_index];
        const float *pos_y = engine->position.y[pos_index];
        const float *vel_x = engine->velocity.x[vel_index];
        const float *vel_y = engine->velocity.y[vel_index];
        float *acc_x = engine->acceleration.x[write_index];
        float *acc_y = engine->acceleration.y[write_index];

        uint64_t frame = atomic_load_explicit(&engine->frame_id, memory_order_relaxed);
        uint32_t count = engine->particle_count;

        for (uint32_t i = 0; i < count; i += 1) {
            float xi = pos_x[i];
            float yi = pos_y[i];
            float vxi = vel_x[i];
            float vyi = vel_y[i];

            float fx = (0.5f - xi) * 0.12f;
            float fy = (0.5f - yi) * 0.12f;
            fx -= vxi * 0.06f;
            fy -= vyi * 0.06f;

            uint32_t owner = engine->owner_id[i];
            uint32_t group = engine->group_id[i];
            for (uint32_t sample = 1; sample <= 6; sample += 1) {
                uint32_t j = (uint32_t)((i + (sample * 97u) + (owner * 13u) + (uint32_t)frame) % count);
                if (j == i) {
                    continue;
                }
                float dx = xi - pos_x[j];
                float dy = yi - pos_y[j];
                float dist_sq = (dx * dx) + (dy * dy) + 0.00009f;
                float inv_dist = 1.0f / dist_sq;
                float group_factor = (group == engine->group_id[j]) ? -0.28f : 1.0f;
                fx += dx * inv_dist * 0.0026f * group_factor;
                fy += dy * inv_dist * 0.0026f * group_factor;
            }

            if ((engine->flags[i] & CDB_FLAG_NEXUS) != 0u) {
                fx *= 0.24f;
                fy *= 0.24f;
            }

            acc_x[i] = fx;
            acc_y[i] = fy;
        }

        atomic_store_explicit(&engine->acceleration.readable, write_index, memory_order_release);
        atomic_fetch_add_explicit(&engine->force_frame, 1u, memory_order_relaxed);
        if (engine->force_sleep_us > 0u) {
            sleep_microseconds(engine->force_sleep_us);
        }
    }

    return NULL;
}

static void *chaos_system_worker(void *arg) {
    CDBEngine *engine = (CDBEngine *)arg;
    if (engine == NULL) {
        return NULL;
    }

    float phase = 0.0f;
    while (atomic_load_explicit(&engine->running, memory_order_acquire) != 0) {
        int write_index = 1 - atomic_load_explicit(&engine->noise.readable, memory_order_relaxed);
        float *noise_x = engine->noise.x[write_index];
        float *noise_y = engine->noise.y[write_index];
        uint32_t count = engine->particle_count;

        phase += 0.0175f;
        if (phase > 1000000.0f) {
            phase = 0.0f;
        }

        for (uint32_t i = 0; i < count; i += 1) {
            float amp = ((engine->flags[i] & CDB_FLAG_CHAOS) != 0u) ? 0.08f : 0.013f;
            float seed = (float)((i * 37u) % 8192u) * 0.0008f;
            noise_x[i] = sinf(phase + seed) * amp;
            noise_y[i] = cosf((phase * 0.91f) + seed) * amp;
        }

        atomic_store_explicit(&engine->noise.readable, write_index, memory_order_release);
        atomic_fetch_add_explicit(&engine->chaos_frame, 1u, memory_order_relaxed);
        if (engine->chaos_sleep_us > 0u) {
            sleep_microseconds(engine->chaos_sleep_us);
        }
    }

    return NULL;
}

static void *semantic_system_worker(void *arg) {
    CDBEngine *engine = (CDBEngine *)arg;
    if (engine == NULL) {
        return NULL;
    }

    while (atomic_load_explicit(&engine->running, memory_order_acquire) != 0) {
        int pos_index = atomic_load_explicit(&engine->position.readable, memory_order_acquire);
        int vel_index = atomic_load_explicit(&engine->velocity.readable, memory_order_acquire);
        int noise_index = atomic_load_explicit(&engine->noise.readable, memory_order_acquire);

        int write_index =
            1 - atomic_load_explicit(&engine->action_prob.readable, memory_order_relaxed);
        const float *pos_x = engine->position.x[pos_index];
        const float *pos_y = engine->position.y[pos_index];
        const float *vel_x = engine->velocity.x[vel_index];
        const float *vel_y = engine->velocity.y[vel_index];
        const float *noise_x = engine->noise.x[noise_index];
        const float *noise_y = engine->noise.y[noise_index];

        float *deflect = engine->action_prob.x[write_index];
        float *diffuse = engine->action_prob.y[write_index];
        float *message_prob = engine->particle_metrics.x[write_index];
        float *entropy = engine->particle_metrics.y[write_index];

        uint32_t count = engine->particle_count;
        for (uint32_t i = 0; i < count; i += 1) {
            uint32_t flags = engine->flags[i];
            float vx = vel_x[i];
            float vy = vel_y[i];
            float speed = sqrtf((vx * vx) + (vy * vy));
            float noise_mag = fabsf(noise_x[i]) + fabsf(noise_y[i]);
            float edge = fabsf(pos_x[i] - 0.5f) + fabsf(pos_y[i] - 0.5f);

            float deflect_value = clamp01(0.44f + (speed * 4.2f) + (edge * 0.16f));
            float message_value = clamp01(0.16f + (noise_mag * 1.6f) + (speed * 2.2f));
            float entropy_value = clamp01(0.34f + (edge * 0.8f) + (noise_mag * 0.7f));

            if ((flags & CDB_FLAG_NEXUS) != 0u) {
                deflect_value = 0.92f;
                message_value = 0.0f;
                entropy_value = clamp01(entropy_value * 0.7f);
            }
            if ((flags & CDB_FLAG_CHAOS) != 0u) {
                deflect_value = clamp01(deflect_value * 0.84f);
                message_value = clamp01(message_value * 1.25f);
                entropy_value = clamp01(entropy_value * 1.2f);
            }

            deflect[i] = deflect_value;
            diffuse[i] = 1.0f - deflect_value;
            message_prob[i] = message_value;
            entropy[i] = entropy_value;
        }

        atomic_store_explicit(&engine->action_prob.readable, write_index, memory_order_release);
        atomic_store_explicit(
            &engine->particle_metrics.readable,
            write_index,
            memory_order_release
        );
        atomic_fetch_add_explicit(&engine->semantic_frame, 1u, memory_order_relaxed);
        if (engine->semantic_sleep_us > 0u) {
            sleep_microseconds(engine->semantic_sleep_us);
        }
    }

    return NULL;
}

static void *integrate_system_worker(void *arg) {
    CDBEngine *engine = (CDBEngine *)arg;
    if (engine == NULL) {
        return NULL;
    }

    while (atomic_load_explicit(&engine->running, memory_order_acquire) != 0) {
        int pos_read = atomic_load_explicit(&engine->position.readable, memory_order_acquire);
        int vel_read = atomic_load_explicit(&engine->velocity.readable, memory_order_acquire);
        int acc_read = atomic_load_explicit(&engine->acceleration.readable, memory_order_acquire);
        int noise_read = atomic_load_explicit(&engine->noise.readable, memory_order_acquire);

        int write_index = 1 - pos_read;
        const float *pos_x = engine->position.x[pos_read];
        const float *pos_y = engine->position.y[pos_read];
        const float *vel_x = engine->velocity.x[vel_read];
        const float *vel_y = engine->velocity.y[vel_read];
        const float *acc_x = engine->acceleration.x[acc_read];
        const float *acc_y = engine->acceleration.y[acc_read];
        const float *noise_x = engine->noise.x[noise_read];
        const float *noise_y = engine->noise.y[noise_read];

        float *write_pos_x = engine->position.x[write_index];
        float *write_pos_y = engine->position.y[write_index];
        float *write_vel_x = engine->velocity.x[write_index];
        float *write_vel_y = engine->velocity.y[write_index];

        uint32_t count = engine->particle_count;
        const float dt = 0.016f;
        for (uint32_t i = 0; i < count; i += 1) {
            float vx = vel_x[i] + ((acc_x[i] + noise_x[i]) * dt);
            float vy = vel_y[i] + ((acc_y[i] + noise_y[i]) * dt);

            float speed_sq = (vx * vx) + (vy * vy);
            float speed_cap = ((engine->flags[i] & CDB_FLAG_NEXUS) != 0u) ? 0.022f : 0.06f;
            if (speed_sq > (speed_cap * speed_cap)) {
                float inv = speed_cap / sqrtf(speed_sq + 0.000001f);
                vx *= inv;
                vy *= inv;
            }

            float x = pos_x[i] + (vx * dt);
            float y = pos_y[i] + (vy * dt);
            if (x < 0.0f) {
                x = 0.0f;
                vx = -vx * 0.52f;
            } else if (x > 1.0f) {
                x = 1.0f;
                vx = -vx * 0.52f;
            }
            if (y < 0.0f) {
                y = 0.0f;
                vy = -vy * 0.52f;
            } else if (y > 1.0f) {
                y = 1.0f;
                vy = -vy * 0.52f;
            }

            write_pos_x[i] = x;
            write_pos_y[i] = y;
            write_vel_x[i] = vx;
            write_vel_y[i] = vy;
        }

        atomic_store_explicit(&engine->position.readable, write_index, memory_order_release);
        atomic_store_explicit(&engine->velocity.readable, write_index, memory_order_release);
        atomic_fetch_add_explicit(&engine->frame_id, 1u, memory_order_relaxed);
        if (engine->integrate_sleep_us > 0u) {
            sleep_microseconds(engine->integrate_sleep_us);
        }
    }

    return NULL;
}

CDBEngine *cdb_engine_create(uint32_t particle_count, uint32_t seed) {
    if (particle_count == 0u) {
        return NULL;
    }

    CDBEngine *engine = (CDBEngine *)calloc(1u, sizeof(CDBEngine));
    if (engine == NULL) {
        return NULL;
    }

    engine->particle_count = particle_count;
    engine->seed = seed;
    atomic_store_explicit(&engine->running, 0, memory_order_release);
    atomic_store_explicit(&engine->frame_id, 0u, memory_order_release);
    atomic_store_explicit(&engine->force_frame, 0u, memory_order_release);
    atomic_store_explicit(&engine->chaos_frame, 0u, memory_order_release);
    atomic_store_explicit(&engine->semantic_frame, 0u, memory_order_release);

    engine->force_sleep_us = env_u32("CDB_FORCE_SLEEP_US", 1200u);
    engine->chaos_sleep_us = env_u32("CDB_CHAOS_SLEEP_US", 2600u);
    engine->integrate_sleep_us = env_u32("CDB_INTEGRATE_SLEEP_US", 1500u);
    engine->semantic_sleep_us = env_u32("CDB_SEMANTIC_SLEEP_US", 9000u);

    uint32_t nexus_stride = env_u32("CDB_NEXUS_STRIDE", 11u);
    if (nexus_stride == 0u) {
        nexus_stride = 11u;
    }
    uint32_t chaos_stride = env_u32("CDB_CHAOS_STRIDE", 17u);
    if (chaos_stride < 2u) {
        chaos_stride = 17u;
    }

    if (
        vec2_buffer_init(&engine->position, particle_count) != 0
        || vec2_buffer_init(&engine->velocity, particle_count) != 0
        || vec2_buffer_init(&engine->acceleration, particle_count) != 0
        || vec2_buffer_init(&engine->noise, particle_count) != 0
        || vec2_buffer_init(&engine->action_prob, particle_count) != 0
        || vec2_buffer_init(&engine->particle_metrics, particle_count) != 0
    ) {
        vec2_buffer_destroy(&engine->position);
        vec2_buffer_destroy(&engine->velocity);
        vec2_buffer_destroy(&engine->acceleration);
        vec2_buffer_destroy(&engine->noise);
        vec2_buffer_destroy(&engine->action_prob);
        vec2_buffer_destroy(&engine->particle_metrics);
        free(engine);
        return NULL;
    }

    engine->owner_id = (uint32_t *)calloc((size_t)particle_count, sizeof(uint32_t));
    engine->group_id = (uint32_t *)calloc((size_t)particle_count, sizeof(uint32_t));
    engine->flags = (uint32_t *)calloc((size_t)particle_count, sizeof(uint32_t));
    engine->mass = (float *)calloc((size_t)particle_count, sizeof(float));
    engine->radius = (float *)calloc((size_t)particle_count, sizeof(float));
    if (
        engine->owner_id == NULL
        || engine->group_id == NULL
        || engine->flags == NULL
        || engine->mass == NULL
        || engine->radius == NULL
    ) {
        cdb_engine_destroy(engine);
        return NULL;
    }

    uint32_t rng = (seed == 0u) ? 17u : seed;
    for (uint32_t i = 0; i < particle_count; i += 1) {
        float x = lcg_unit(&rng);
        float y = lcg_unit(&rng);
        float vx = (lcg_unit(&rng) - 0.5f) * 0.02f;
        float vy = (lcg_unit(&rng) - 0.5f) * 0.02f;

        engine->position.x[0][i] = x;
        engine->position.y[0][i] = y;
        engine->position.x[1][i] = x;
        engine->position.y[1][i] = y;

        engine->velocity.x[0][i] = vx;
        engine->velocity.y[0][i] = vy;
        engine->velocity.x[1][i] = vx;
        engine->velocity.y[1][i] = vy;

        engine->acceleration.x[0][i] = 0.0f;
        engine->acceleration.y[0][i] = 0.0f;
        engine->acceleration.x[1][i] = 0.0f;
        engine->acceleration.y[1][i] = 0.0f;

        engine->noise.x[0][i] = 0.0f;
        engine->noise.y[0][i] = 0.0f;
        engine->noise.x[1][i] = 0.0f;
        engine->noise.y[1][i] = 0.0f;

        engine->action_prob.x[0][i] = 0.66f;
        engine->action_prob.y[0][i] = 0.34f;
        engine->action_prob.x[1][i] = 0.66f;
        engine->action_prob.y[1][i] = 0.34f;

        engine->particle_metrics.x[0][i] = 0.24f;
        engine->particle_metrics.y[0][i] = 0.45f;
        engine->particle_metrics.x[1][i] = 0.24f;
        engine->particle_metrics.y[1][i] = 0.45f;

        engine->owner_id[i] = i % 64u;
        engine->group_id[i] = (i / 7u) % 16u;
        engine->mass[i] = 0.7f + (lcg_unit(&rng) * 0.8f);
        engine->radius[i] = 0.008f + (lcg_unit(&rng) * 0.01f);

        uint32_t flags = CDB_FLAG_ACTIVE;
        if ((i % nexus_stride) == 0u) {
            flags |= CDB_FLAG_NEXUS;
        }
        if ((i % chaos_stride) == 0u) {
            flags |= CDB_FLAG_CHAOS;
        }
        engine->flags[i] = flags;
    }

    return engine;
}

int cdb_engine_start(CDBEngine *engine) {
    if (engine == NULL) {
        return -1;
    }
    int was_running = atomic_exchange_explicit(&engine->running, 1, memory_order_acq_rel);
    if (was_running != 0) {
        return 0;
    }

    if (pthread_create(&engine->force_thread, NULL, force_system_worker, engine) != 0) {
        atomic_store_explicit(&engine->running, 0, memory_order_release);
        return -1;
    }
    if (pthread_create(&engine->chaos_thread, NULL, chaos_system_worker, engine) != 0) {
        atomic_store_explicit(&engine->running, 0, memory_order_release);
        pthread_join(engine->force_thread, NULL);
        return -1;
    }
    if (pthread_create(&engine->semantic_thread, NULL, semantic_system_worker, engine) != 0) {
        atomic_store_explicit(&engine->running, 0, memory_order_release);
        pthread_join(engine->force_thread, NULL);
        pthread_join(engine->chaos_thread, NULL);
        return -1;
    }
    if (pthread_create(&engine->integrate_thread, NULL, integrate_system_worker, engine) != 0) {
        atomic_store_explicit(&engine->running, 0, memory_order_release);
        pthread_join(engine->force_thread, NULL);
        pthread_join(engine->chaos_thread, NULL);
        pthread_join(engine->semantic_thread, NULL);
        return -1;
    }

    return 0;
}

int cdb_engine_stop(CDBEngine *engine) {
    if (engine == NULL) {
        return -1;
    }
    int was_running = atomic_exchange_explicit(&engine->running, 0, memory_order_acq_rel);
    if (was_running == 0) {
        return 0;
    }

    pthread_join(engine->force_thread, NULL);
    pthread_join(engine->chaos_thread, NULL);
    pthread_join(engine->semantic_thread, NULL);
    pthread_join(engine->integrate_thread, NULL);
    return 0;
}

void cdb_engine_destroy(CDBEngine *engine) {
    if (engine == NULL) {
        return;
    }

    (void)cdb_engine_stop(engine);
    vec2_buffer_destroy(&engine->position);
    vec2_buffer_destroy(&engine->velocity);
    vec2_buffer_destroy(&engine->acceleration);
    vec2_buffer_destroy(&engine->noise);
    vec2_buffer_destroy(&engine->action_prob);
    vec2_buffer_destroy(&engine->particle_metrics);

    free(engine->owner_id);
    free(engine->group_id);
    free(engine->flags);
    free(engine->mass);
    free(engine->radius);

    free(engine);
}

uint32_t cdb_engine_particle_count(CDBEngine *engine) {
    return (engine == NULL) ? 0u : engine->particle_count;
}

uint32_t cdb_engine_snapshot(
    CDBEngine *engine,
    float *out_x,
    float *out_y,
    float *out_vx,
    float *out_vy,
    float *out_deflect,
    float *out_message,
    float *out_entropy,
    uint32_t *out_owner,
    uint32_t *out_flags,
    uint32_t capacity,
    uint64_t *out_frame,
    uint64_t *out_force_frame,
    uint64_t *out_chaos_frame,
    uint64_t *out_semantic_frame
) {
    if (engine == NULL || capacity == 0u) {
        if (out_frame != NULL) {
            *out_frame = 0u;
        }
        if (out_force_frame != NULL) {
            *out_force_frame = 0u;
        }
        if (out_chaos_frame != NULL) {
            *out_chaos_frame = 0u;
        }
        if (out_semantic_frame != NULL) {
            *out_semantic_frame = 0u;
        }
        return 0u;
    }

    uint32_t count = engine->particle_count;
    if (capacity < count) {
        count = capacity;
    }

    int pos_index = atomic_load_explicit(&engine->position.readable, memory_order_acquire);
    int vel_index = atomic_load_explicit(&engine->velocity.readable, memory_order_acquire);
    int action_index = atomic_load_explicit(&engine->action_prob.readable, memory_order_acquire);
    int metrics_index = atomic_load_explicit(
        &engine->particle_metrics.readable,
        memory_order_acquire
    );
    const float *src_x = engine->position.x[pos_index];
    const float *src_y = engine->position.y[pos_index];
    const float *src_vx = engine->velocity.x[vel_index];
    const float *src_vy = engine->velocity.y[vel_index];
    const float *src_deflect = engine->action_prob.x[action_index];
    const float *src_message = engine->particle_metrics.x[metrics_index];
    const float *src_entropy = engine->particle_metrics.y[metrics_index];

    for (uint32_t i = 0; i < count; i += 1) {
        if (out_x != NULL) {
            out_x[i] = src_x[i];
        }
        if (out_y != NULL) {
            out_y[i] = src_y[i];
        }
        if (out_vx != NULL) {
            out_vx[i] = src_vx[i];
        }
        if (out_vy != NULL) {
            out_vy[i] = src_vy[i];
        }
        if (out_deflect != NULL) {
            out_deflect[i] = src_deflect[i];
        }
        if (out_message != NULL) {
            out_message[i] = src_message[i];
        }
        if (out_entropy != NULL) {
            out_entropy[i] = src_entropy[i];
        }
        if (out_owner != NULL) {
            out_owner[i] = engine->owner_id[i];
        }
        if (out_flags != NULL) {
            out_flags[i] = engine->flags[i];
        }
    }

    if (out_frame != NULL) {
        *out_frame = atomic_load_explicit(&engine->frame_id, memory_order_relaxed);
    }
    if (out_force_frame != NULL) {
        *out_force_frame = atomic_load_explicit(&engine->force_frame, memory_order_relaxed);
    }
    if (out_chaos_frame != NULL) {
        *out_chaos_frame = atomic_load_explicit(&engine->chaos_frame, memory_order_relaxed);
    }
    if (out_semantic_frame != NULL) {
        *out_semantic_frame = atomic_load_explicit(&engine->semantic_frame, memory_order_relaxed);
    }
    return count;
}
