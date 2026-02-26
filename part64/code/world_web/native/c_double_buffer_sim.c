#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

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
#include <unistd.h>

#define CDB_FLAG_NEXUS 0x1u
#define CDB_FLAG_CHAOS 0x2u
#define CDB_FLAG_ACTIVE 0x4u
#define CDB_WORLD_EDGE_BAND 0.12f
#define CDB_WORLD_EDGE_PRESSURE 0.18f
#define CDB_WORLD_EDGE_BOUNCE 0.74f

#define CDB_NOOI_COLS 64
#define CDB_NOOI_ROWS 64
#define CDB_NOOI_LAYERS 8
#define CDB_NOOI_SIZE (CDB_NOOI_COLS * CDB_NOOI_ROWS * CDB_NOOI_LAYERS * 2)

#define CDB_SIM_FLAG_COLLISION 0x1u
#define CDB_SIM_FLAG_MEAN_FIELD 0x2u

typedef struct Vec2DoubleBuffer {
    float *x[2];
    float *y[2];
    _Atomic int readable;
} Vec2DoubleBuffer;

typedef struct CDBQuadNode {
    float min_x;
    float min_y;
    float max_x;
    float max_y;
    uint32_t child_base;
    int32_t particle_head;
    uint16_t particle_count;
    uint16_t depth;
    uint32_t total_count;
    uint32_t nexus_count;
    float max_radius;
    float mass;
    float com_x;
    float com_y;
    float emb_sum[24];
    float emb_norm;
    uint64_t group_mask;
} CDBQuadNode;

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
    float *embeddings; // 24-dim per particle

    float *nooi_field;
    _Atomic int nooi_index; 
    float *nooi_buffer[2];
    _Atomic int nooi_readable;

    uint32_t force_sleep_us;
    uint32_t chaos_sleep_us;
    uint32_t integrate_sleep_us;
    uint32_t semantic_sleep_us;

    float daimon_friction;
    float nexus_friction;
    float grav_const;
    float grav_eps;

    uint32_t sim_flags;
    int32_t *grid_head;
    int32_t *grid_next;

    CDBQuadNode *quad_nodes;
    uint32_t quad_capacity;
    float bh_theta;
    uint32_t bh_leaf_capacity;
    uint32_t bh_max_depth;
    float collision_spring;
    float cluster_theta;
    float cluster_rest_length;
    float cluster_stiffness;
    uint32_t force_worker_count;

    pthread_t force_thread;
    pthread_t chaos_thread;
    pthread_t semantic_thread;
    pthread_t integrate_thread;
} CDBEngine;

void cdb_engine_destroy(CDBEngine *engine);
void cdb_release_thread_scratch(void);

void cdb_engine_set_flags(CDBEngine *engine, uint32_t flags) {
    if (engine != NULL) {
        atomic_store_explicit((_Atomic uint32_t *)&engine->sim_flags, flags, memory_order_release);
    }
}

int cdb_engine_update_nooi(CDBEngine *engine, const float *data);

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

static float env_f32(const char *name, float fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    float parsed = strtof(value, &end);
    if (end == value || !isfinite(parsed)) {
        return fallback;
    }
    return parsed;
}

static uint32_t detect_cpu_count(void) {
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count <= 0) {
        return 1u;
    }
    if (count > 64) {
        return 64u;
    }
    return (uint32_t)count;
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

static _Thread_local float *g_scratch_dist = NULL;
static _Thread_local uint32_t g_scratch_dist_cap = 0u;
static _Thread_local uint8_t *g_scratch_visited = NULL;
static _Thread_local uint32_t g_scratch_visited_cap = 0u;
static _Thread_local uint32_t *g_scratch_u32_a = NULL;
static _Thread_local uint32_t g_scratch_u32_a_cap = 0u;
static _Thread_local uint32_t *g_scratch_u32_b = NULL;
static _Thread_local uint32_t g_scratch_u32_b_cap = 0u;
static _Thread_local float *g_scratch_float_a = NULL;
static _Thread_local uint32_t g_scratch_float_a_cap = 0u;
static _Thread_local uint32_t *g_scratch_u32_c = NULL;
static _Thread_local uint32_t g_scratch_u32_c_cap = 0u;
static _Thread_local float *g_scratch_edge_cost = NULL;
static _Thread_local uint32_t g_scratch_edge_cost_cap = 0u;
static _Thread_local int32_t *g_collision_cell_head = NULL;
static _Thread_local uint32_t g_collision_cell_head_cap = 0u;
static _Thread_local int32_t *g_collision_next = NULL;
static _Thread_local uint32_t g_collision_next_cap = 0u;
static _Thread_local float *g_collision_dx = NULL;
static _Thread_local uint32_t g_collision_dx_cap = 0u;
static _Thread_local float *g_collision_dy = NULL;
static _Thread_local uint32_t g_collision_dy_cap = 0u;
static _Thread_local float *g_collision_dvx = NULL;
static _Thread_local uint32_t g_collision_dvx_cap = 0u;
static _Thread_local float *g_collision_dvy = NULL;
static _Thread_local uint32_t g_collision_dvy_cap = 0u;
static _Thread_local uint32_t *g_collision_hits = NULL;
static _Thread_local uint32_t g_collision_hits_cap = 0u;

void cdb_release_thread_scratch(void) {
    free(g_scratch_dist);
    g_scratch_dist = NULL;
    g_scratch_dist_cap = 0u;

    free(g_scratch_visited);
    g_scratch_visited = NULL;
    g_scratch_visited_cap = 0u;

    free(g_scratch_u32_a);
    g_scratch_u32_a = NULL;
    g_scratch_u32_a_cap = 0u;

    free(g_scratch_u32_b);
    g_scratch_u32_b = NULL;
    g_scratch_u32_b_cap = 0u;

    free(g_scratch_float_a);
    g_scratch_float_a = NULL;
    g_scratch_float_a_cap = 0u;

    free(g_scratch_u32_c);
    g_scratch_u32_c = NULL;
    g_scratch_u32_c_cap = 0u;

    free(g_scratch_edge_cost);
    g_scratch_edge_cost = NULL;
    g_scratch_edge_cost_cap = 0u;

    free(g_collision_cell_head);
    g_collision_cell_head = NULL;
    g_collision_cell_head_cap = 0u;

    free(g_collision_next);
    g_collision_next = NULL;
    g_collision_next_cap = 0u;

    free(g_collision_dx);
    g_collision_dx = NULL;
    g_collision_dx_cap = 0u;

    free(g_collision_dy);
    g_collision_dy = NULL;
    g_collision_dy_cap = 0u;

    free(g_collision_dvx);
    g_collision_dvx = NULL;
    g_collision_dvx_cap = 0u;

    free(g_collision_dvy);
    g_collision_dvy = NULL;
    g_collision_dvy_cap = 0u;

    free(g_collision_hits);
    g_collision_hits = NULL;
    g_collision_hits_cap = 0u;
}

static int cdb_ensure_scratch_float(float **buffer, uint32_t *capacity, uint32_t count) {
    if (count == 0u) {
        return 0;
    }
    if (*capacity >= count && *buffer != NULL) {
        return 0;
    }
    float *next = (float *)realloc(*buffer, (size_t)count * sizeof(float));
    if (next == NULL) {
        return -1;
    }
    *buffer = next;
    *capacity = count;
    return 0;
}

static int cdb_ensure_scratch_u32(uint32_t **buffer, uint32_t *capacity, uint32_t count) {
    if (count == 0u) {
        return 0;
    }
    if (*capacity >= count && *buffer != NULL) {
        return 0;
    }
    uint32_t *next = (uint32_t *)realloc(*buffer, (size_t)count * sizeof(uint32_t));
    if (next == NULL) {
        return -1;
    }
    *buffer = next;
    *capacity = count;
    return 0;
}

static int cdb_ensure_scratch_u8(uint8_t **buffer, uint32_t *capacity, uint32_t count) {
    if (count == 0u) {
        return 0;
    }
    if (*capacity >= count && *buffer != NULL) {
        return 0;
    }
    uint8_t *next = (uint8_t *)realloc(*buffer, (size_t)count * sizeof(uint8_t));
    if (next == NULL) {
        return -1;
    }
    *buffer = next;
    *capacity = count;
    return 0;
}

static int cdb_ensure_scratch_i32(int32_t **buffer, uint32_t *capacity, uint32_t count) {
    if (count == 0u) {
        return 0;
    }
    if (*capacity >= count && *buffer != NULL) {
        return 0;
    }
    int32_t *next = (int32_t *)realloc(*buffer, (size_t)count * sizeof(int32_t));
    if (next == NULL) {
        return -1;
    }
    *buffer = next;
    *capacity = count;
    return 0;
}

static int fast_floor_to_int(float value) {
    int i = (int)value;
    return (value < (float)i) ? (i - 1) : i;
}

static uint32_t mix_hash_u32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

static uint32_t simplex_hash2(int i, int j, uint32_t seed) {
    uint32_t mix =
        (((uint32_t)i + 1u) * 0x9E3779B1u)
        ^ (((uint32_t)j + 1u) * 0x85EBCA77u)
        ^ ((seed + 1u) * 0xC2B2AE3Du);
    return mix_hash_u32(mix) % 12u;
}

static float simplex_grad2(uint32_t hash, float x, float y) {
    switch (hash % 12u) {
        case 0u:
            return x + y;
        case 1u:
            return -x + y;
        case 2u:
            return x - y;
        case 3u:
            return -x - y;
        case 4u:
            return x;
        case 5u:
            return -x;
        case 6u:
            return x;
        case 7u:
            return -x;
        case 8u:
            return y;
        case 9u:
            return -y;
        case 10u:
            return y;
        default:
            return -y;
    }
}

static float simplex_noise_2d(float x, float y, uint32_t seed) {
    const float F2 = 0.3660254037844386f;
    const float G2 = 0.21132486540518713f;

    float s = (x + y) * F2;
    int i = fast_floor_to_int(x + s);
    int j = fast_floor_to_int(y + s);
    float t = ((float)(i + j)) * G2;

    float x0 = x - (((float)i) - t);
    float y0 = y - (((float)j) - t);

    int i1;
    int j1;
    if (x0 > y0) {
        i1 = 1;
        j1 = 0;
    } else {
        i1 = 0;
        j1 = 1;
    }

    float x1 = x0 - ((float)i1) + G2;
    float y1 = y0 - ((float)j1) + G2;
    float x2 = x0 - 1.0f + (2.0f * G2);
    float y2 = y0 - 1.0f + (2.0f * G2);

    float n0 = 0.0f;
    float n1 = 0.0f;
    float n2 = 0.0f;

    float t0 = 0.5f - (x0 * x0) - (y0 * y0);
    if (t0 >= 0.0f) {
        t0 *= t0;
        n0 = t0 * t0 * simplex_grad2(simplex_hash2(i, j, seed), x0, y0);
    }

    float t1 = 0.5f - (x1 * x1) - (y1 * y1);
    if (t1 >= 0.0f) {
        t1 *= t1;
        n1 = t1 * t1 * simplex_grad2(simplex_hash2(i + i1, j + j1, seed), x1, y1);
    }

    float t2 = 0.5f - (x2 * x2) - (y2 * y2);
    if (t2 >= 0.0f) {
        t2 *= t2;
        n2 = t2 * t2 * simplex_grad2(simplex_hash2(i + 1, j + 1, seed), x2, y2);
    }

    return 70.0f * (n0 + n1 + n2);
}

static int cdb_graph_runtime_accumulate_gravity(
    uint32_t node_count,
    uint32_t edge_count,
    const uint32_t *edge_src,
    const uint32_t *edge_dst,
    const float *edge_cost_cache,
    float wl,
    float wc,
    float ws,
    float global_sat,
    const uint32_t *source_nodes,
    const float *source_mass,
    const float *source_need,
    uint32_t source_count,
    float bounded_radius,
    float grav_const,
    float grav_eps,
    float *out_min_dist,
    float *out_gravity
) {
    if ((out_gravity == NULL && out_min_dist == NULL) || source_count == 0u) {
        return 0;
    }

    if (
        cdb_ensure_scratch_float(&g_scratch_dist, &g_scratch_dist_cap, node_count) != 0
        || cdb_ensure_scratch_u8(&g_scratch_visited, &g_scratch_visited_cap, node_count) != 0
    ) {
        return -1;
    }
    float *dist = g_scratch_dist;
    uint8_t *visited = g_scratch_visited;

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

    return 0;
}

static void cdb_graph_runtime_populate_edge_metrics(
    uint32_t node_count,
    uint32_t edge_count,
    const uint32_t *edge_src,
    const uint32_t *edge_dst,
    const float *edge_affinity,
    float global_sat,
    float wl,
    float wc,
    float ws,
    uint32_t *out_degree,
    uint32_t *in_degree,
    float *node_sat_sum,
    uint32_t *node_sat_count,
    float *edge_cost_cache
) {
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

            node_sat_sum[src] += sat;
            node_sat_count[src] += 1u;
            node_sat_sum[dst] += sat;
            node_sat_count[dst] += 1u;
        }

        if (edge_cost_cache != NULL) {
            edge_cost_cache[edge] = fmaxf(0.0001f, wl + (wc * sat) + (ws * (1.0f - affinity)));
        }
    }
}

static void cdb_graph_runtime_finalize_node_outputs(
    uint32_t node_count,
    float global_sat,
    const uint32_t *node_sat_count,
    const float *node_sat_sum,
    float *out_min_dist,
    const float *out_gravity,
    float *out_node_saturation,
    float *out_node_price
) {
    float max_gravity = 1.0f;
    if (out_gravity != NULL) {
        max_gravity = 0.0f;
        for (uint32_t node = 0u; node < node_count; node += 1u) {
            float gravity = out_gravity[node];
            if (isfinite(gravity) && gravity > max_gravity) {
                max_gravity = gravity;
            }
        }
        if (max_gravity <= 1e-6f) {
            max_gravity = 1.0f;
        }
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

    if (
        cdb_ensure_scratch_u32(&g_scratch_u32_a, &g_scratch_u32_a_cap, node_count) != 0
        || cdb_ensure_scratch_u32(&g_scratch_u32_b, &g_scratch_u32_b_cap, node_count) != 0
        || cdb_ensure_scratch_float(&g_scratch_float_a, &g_scratch_float_a_cap, node_count) != 0
        || cdb_ensure_scratch_u32(&g_scratch_u32_c, &g_scratch_u32_c_cap, node_count) != 0
    ) {
        return -1;
    }
    uint32_t *out_degree = g_scratch_u32_a;
    uint32_t *in_degree = g_scratch_u32_b;
    float *node_sat_sum = g_scratch_float_a;
    uint32_t *node_sat_count = g_scratch_u32_c;
    memset(out_degree, 0, (size_t)node_count * sizeof(uint32_t));
    memset(in_degree, 0, (size_t)node_count * sizeof(uint32_t));
    memset(node_sat_sum, 0, (size_t)node_count * sizeof(float));
    memset(node_sat_count, 0, (size_t)node_count * sizeof(uint32_t));

    float *edge_cost_cache = out_edge_cost;
    if (edge_cost_cache == NULL && edge_count > 0u) {
        if (
            cdb_ensure_scratch_float(
                &g_scratch_edge_cost,
                &g_scratch_edge_cost_cap,
                edge_count
            ) != 0
        ) {
            return -1;
        }
        edge_cost_cache = g_scratch_edge_cost;
        memset(edge_cost_cache, 0, (size_t)edge_count * sizeof(float));
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

    cdb_graph_runtime_populate_edge_metrics(
        node_count,
        edge_count,
        edge_src,
        edge_dst,
        edge_affinity,
        global_sat,
        wl,
        wc,
        ws,
        out_degree,
        in_degree,
        node_sat_sum,
        node_sat_count,
        edge_cost_cache
    );

    if (
        cdb_graph_runtime_accumulate_gravity(
            node_count,
            edge_count,
            edge_src,
            edge_dst,
            edge_cost_cache,
            wl,
            wc,
            ws,
            global_sat,
            source_nodes,
            source_mass,
            source_need,
            source_count,
            bounded_radius,
            grav_const,
            grav_eps,
            out_min_dist,
            out_gravity
        ) != 0
    ) {
        return -1;
    }

    cdb_graph_runtime_finalize_node_outputs(
        node_count,
        global_sat,
        node_sat_count,
        node_sat_sum,
        out_min_dist,
        out_gravity,
        out_node_saturation,
        out_node_price
    );

    return 0;
}

static float cdb_route_edge_score(
    uint32_t edge,
    uint32_t dst,
    const float *edge_cost,
    const float *node_gravity,
    float gravity_here,
    float eta_gain,
    float cost_gain
) {
    float gravity_next = 0.0f;
    if (node_gravity != NULL) {
        gravity_next = fmaxf(0.0f, node_gravity[dst]);
    }

    float cost = 1.0f;
    if (edge_cost != NULL) {
        cost = fmaxf(0.0001f, edge_cost[edge]);
    }

    return (eta_gain * (gravity_next - gravity_here)) - (cost_gain * cost);
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

            float score = cdb_route_edge_score(
                edge,
                dst,
                edge_cost,
                node_gravity,
                gravity_here,
                eta_gain,
                cost_gain
            );
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
                float score = cdb_route_edge_score(
                    edge,
                    dst,
                    edge_cost,
                    node_gravity,
                    gravity_here,
                    eta_gain,
                    cost_gain
                );
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

// ============================================================================
// CSR (Compressed Sparse Row) Edge Storage for O(degree) Route Lookups
// ============================================================================

int cdb_build_csr_edges(
    uint32_t node_count,
    uint32_t edge_count,
    const uint32_t *edge_src,
    const uint32_t *edge_dst,
    uint32_t *csr_node_offsets,
    uint32_t *csr_edge_indices
) {
    if (node_count == 0u || edge_count == 0u) {
        return -1;
    }
    if (edge_src == NULL || edge_dst == NULL || csr_node_offsets == NULL || csr_edge_indices == NULL) {
        return -1;
    }

    // Count edges per node
    if (cdb_ensure_scratch_u32(&g_scratch_u32_a, &g_scratch_u32_a_cap, node_count) != 0) {
        return -1;
    }
    uint32_t *degree = g_scratch_u32_a;
    memset(degree, 0, (size_t)node_count * sizeof(uint32_t));

    for (uint32_t e = 0u; e < edge_count; e += 1u) {
        uint32_t src = edge_src[e];
        if (src < node_count) {
            degree[src] += 1u;
        }
    }

    // Build offsets (prefix sum)
    csr_node_offsets[0] = 0u;
    for (uint32_t n = 0u; n < node_count; n += 1u) {
        csr_node_offsets[n + 1u] = csr_node_offsets[n] + degree[n];
    }

    // Reset degree for use as insertion cursor
    memset(degree, 0, (size_t)node_count * sizeof(uint32_t));

    // Fill edge indices
    for (uint32_t e = 0u; e < edge_count; e += 1u) {
        uint32_t src = edge_src[e];
        if (src < node_count) {
            uint32_t pos = csr_node_offsets[src] + degree[src];
            csr_edge_indices[pos] = e;
            degree[src] += 1u;
        }
    }

    return 0;
}

int cdb_graph_route_step_csr(
    uint32_t node_count,
    uint32_t edge_count,
    const uint32_t *edge_src,
    const uint32_t *edge_dst,
    const float *edge_cost,
    const float *node_gravity,
    const uint32_t *csr_node_offsets,
    const uint32_t *csr_edge_indices,
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
        || csr_node_offsets == NULL
        || csr_edge_indices == NULL
    ) {
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

        // CSR range for current node
        uint32_t edge_start = csr_node_offsets[current];
        uint32_t edge_end = csr_node_offsets[current + 1u];

        for (uint32_t idx = edge_start; idx < edge_end; idx += 1u) {
            uint32_t edge = csr_edge_indices[idx];
            uint32_t dst = edge_dst[edge];
            if (dst >= node_count) {
                continue;
            }

            float score = cdb_route_edge_score(
                edge,
                dst,
                edge_cost,
                node_gravity,
                gravity_here,
                eta_gain,
                cost_gain
            );
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

            // CSR range exploration
            for (uint32_t idx = edge_start; idx < edge_end; idx += 1u) {
                uint32_t edge = csr_edge_indices[idx];
                uint32_t dst = edge_dst[edge];
                if (dst >= node_count) {
                    continue;
                }
                float score = cdb_route_edge_score(
                    edge,
                    dst,
                    edge_cost,
                    node_gravity,
                    gravity_here,
                    eta_gain,
                    cost_gain
                );
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

typedef struct CDBCollisionResolveContext {
    uint32_t count;
    uint32_t cols;
    uint32_t rows;
    uint32_t cell_count;
    const int32_t *cell_head;
    const int32_t *next;
    const float *x;
    const float *y;
    const float *vx;
    const float *vy;
    const float *radius;
    const float *mass;
    float restitution;
    float separation_percent;
} CDBCollisionResolveContext;

typedef struct CDBCollisionResolveTask {
    const CDBCollisionResolveContext *ctx;
    uint32_t start_cell;
    uint32_t end_cell;
    float *dx;
    float *dy;
    float *dvx;
    float *dvy;
    uint32_t *hit_count;
    uint32_t collision_pairs;
} CDBCollisionResolveTask;

static void cdb_collision_apply_pair(
    CDBCollisionResolveTask *task,
    uint32_t i,
    uint32_t j
) {
    if (task == NULL || task->ctx == NULL) {
        return;
    }
    const CDBCollisionResolveContext *ctx = task->ctx;
    if (i >= ctx->count || j >= ctx->count || i == j) {
        return;
    }

    float xi = ctx->x[i];
    float yi = ctx->y[i];
    float xj = ctx->x[j];
    float yj = ctx->y[j];
    if (!isfinite(xi) || !isfinite(yi) || !isfinite(xj) || !isfinite(yj)) {
        return;
    }

    float radius_i = ctx->radius != NULL ? ctx->radius[i] : 0.0f;
    float radius_j = ctx->radius != NULL ? ctx->radius[j] : 0.0f;
    if (!isfinite(radius_i) || radius_i < 0.0f) {
        radius_i = 0.0f;
    }
    if (!isfinite(radius_j) || radius_j < 0.0f) {
        radius_j = 0.0f;
    }
    float min_distance = radius_i + radius_j;
    if (min_distance <= 0.0f) {
        return;
    }

    float dx = xj - xi;
    float dy = yj - yi;
    float dist_sq = (dx * dx) + (dy * dy);
    float min_dist_sq = min_distance * min_distance;
    if (dist_sq >= min_dist_sq) {
        return;
    }

    float nx = 0.0f;
    float ny = 0.0f;
    float distance = 0.0f;
    if (dist_sq < 1e-12f) {
        uint32_t jitter_seed = mix_hash_u32(
            ((i + 1u) * 73856093u)
            ^ ((j + 1u) * 19349663u)
            ^ 0x9E3779B9u
        );
        float theta = ((float)(jitter_seed % 6283u)) * 0.001f;
        nx = cosf(theta);
        ny = sinf(theta);
        distance = 1e-6f;
    } else {
        distance = sqrtf(dist_sq);
        nx = dx / distance;
        ny = dy / distance;
    }

    float mass_i = ctx->mass != NULL ? ctx->mass[i] : 1.0f;
    float mass_j = ctx->mass != NULL ? ctx->mass[j] : 1.0f;
    if (!isfinite(mass_i) || mass_i < 0.2f) {
        mass_i = 0.2f;
    }
    if (!isfinite(mass_j) || mass_j < 0.2f) {
        mass_j = 0.2f;
    }
    float inv_mass_i = 1.0f / mass_i;
    float inv_mass_j = 1.0f / mass_j;

    float vx_i = ctx->vx[i];
    float vy_i = ctx->vy[i];
    float vx_j = ctx->vx[j];
    float vy_j = ctx->vy[j];

    float rel_vx = vx_i - vx_j;
    float rel_vy = vy_i - vy_j;
    float vel_normal = (rel_vx * nx) + (rel_vy * ny);

    if (vel_normal < 0.0f) {
        float impulse = (-(1.0f + ctx->restitution) * vel_normal)
            / fmaxf(1e-6f, inv_mass_i + inv_mass_j);
        float impulse_x = impulse * nx;
        float impulse_y = impulse * ny;
        task->dvx[i] += impulse_x * inv_mass_i;
        task->dvy[i] += impulse_y * inv_mass_i;
        task->dvx[j] -= impulse_x * inv_mass_j;
        task->dvy[j] -= impulse_y * inv_mass_j;

        float tangent_x = rel_vx - (vel_normal * nx);
        float tangent_y = rel_vy - (vel_normal * ny);
        float tangent_norm = sqrtf((tangent_x * tangent_x) + (tangent_y * tangent_y));
        if (tangent_norm > 1e-6f) {
            tangent_x /= tangent_norm;
            tangent_y /= tangent_norm;
            float tangent_velocity = (rel_vx * tangent_x) + (rel_vy * tangent_y);
            float tangent_impulse = fminf(
                fabsf(impulse) * 0.18f,
                fabsf(tangent_velocity)
            );
            task->dvx[i] -= tangent_impulse * tangent_x * inv_mass_i;
            task->dvy[i] -= tangent_impulse * tangent_y * inv_mass_i;
            task->dvx[j] += tangent_impulse * tangent_x * inv_mass_j;
            task->dvy[j] += tangent_impulse * tangent_y * inv_mass_j;
        }
    }

    float penetration = min_distance - distance;
    float correction = (
        fmaxf(0.0f, penetration) / fmaxf(1e-6f, inv_mass_i + inv_mass_j)
    ) * ctx->separation_percent;
    float correction_x = correction * nx;
    float correction_y = correction * ny;
    task->dx[i] -= correction_x * inv_mass_i;
    task->dy[i] -= correction_y * inv_mass_i;
    task->dx[j] += correction_x * inv_mass_j;
    task->dy[j] += correction_y * inv_mass_j;

    task->hit_count[i] += 1u;
    task->hit_count[j] += 1u;
    task->collision_pairs += 1u;
}

static void cdb_collision_process_cells(CDBCollisionResolveTask *task) {
    if (task == NULL || task->ctx == NULL) {
        return;
    }
    const CDBCollisionResolveContext *ctx = task->ctx;
    if (ctx->cell_head == NULL || ctx->next == NULL) {
        return;
    }

    for (uint32_t cell = task->start_cell; cell < task->end_cell; cell += 1u) {
        int32_t head_i = ctx->cell_head[cell];
        if (head_i < 0) {
            continue;
        }

        uint32_t gx = cell % ctx->cols;
        uint32_t gy = cell / ctx->cols;
        int32_t x_min = (gx > 0u) ? (int32_t)(gx - 1u) : (int32_t)gx;
        int32_t x_max = (gx + 1u < ctx->cols) ? (int32_t)(gx + 1u) : (int32_t)gx;
        int32_t y_min = (gy > 0u) ? (int32_t)(gy - 1u) : (int32_t)gy;
        int32_t y_max = (gy + 1u < ctx->rows) ? (int32_t)(gy + 1u) : (int32_t)gy;

        for (int32_t ny = y_min; ny <= y_max; ny += 1) {
            for (int32_t nx = x_min; nx <= x_max; nx += 1) {
                uint32_t neighbor = ((uint32_t)ny * ctx->cols) + (uint32_t)nx;
                if (neighbor < cell) {
                    continue;
                }
                int32_t head_j = ctx->cell_head[neighbor];
                if (head_j < 0) {
                    continue;
                }

                if (neighbor == cell) {
                    for (int32_t i = head_i; i >= 0; i = ctx->next[(uint32_t)i]) {
                        for (
                            int32_t j = ctx->next[(uint32_t)i];
                            j >= 0;
                            j = ctx->next[(uint32_t)j]
                        ) {
                            cdb_collision_apply_pair(task, (uint32_t)i, (uint32_t)j);
                        }
                    }
                } else {
                    for (int32_t i = head_i; i >= 0; i = ctx->next[(uint32_t)i]) {
                        for (int32_t j = head_j; j >= 0; j = ctx->next[(uint32_t)j]) {
                            cdb_collision_apply_pair(task, (uint32_t)i, (uint32_t)j);
                        }
                    }
                }
            }
        }
    }
}

static void *cdb_collision_worker_entry(void *arg) {
    CDBCollisionResolveTask *task = (CDBCollisionResolveTask *)arg;
    if (task == NULL) {
        return NULL;
    }
    cdb_collision_process_cells(task);
    return NULL;
}

uint32_t cdb_resolve_semantic_collisions(
    uint32_t count,
    float *x,
    float *y,
    float *vx,
    float *vy,
    const float *radius,
    const float *mass,
    float restitution,
    float separation_percent,
    float cell_size,
    uint32_t worker_count,
    uint32_t *out_collision_count
) {
    if (
        count == 0u
        || x == NULL
        || y == NULL
        || vx == NULL
        || vy == NULL
        || radius == NULL
        || mass == NULL
    ) {
        return 0u;
    }

    if (out_collision_count != NULL) {
        memset(out_collision_count, 0, (size_t)count * sizeof(uint32_t));
    }
    if (count < 2u) {
        return 0u;
    }

    float bounded_restitution = isfinite(restitution) ? restitution : 0.88f;
    if (bounded_restitution < 0.0f) {
        bounded_restitution = 0.0f;
    } else if (bounded_restitution > 1.0f) {
        bounded_restitution = 1.0f;
    }

    float bounded_separation = isfinite(separation_percent) ? separation_percent : 0.72f;
    if (bounded_separation < 0.0f) {
        bounded_separation = 0.0f;
    } else if (bounded_separation > 1.2f) {
        bounded_separation = 1.2f;
    }

    float bounded_cell = isfinite(cell_size) ? cell_size : 0.04f;
    if (bounded_cell < 0.005f) {
        bounded_cell = 0.005f;
    } else if (bounded_cell > 0.25f) {
        bounded_cell = 0.25f;
    }

    uint32_t cols = (uint32_t)floorf(1.0f / bounded_cell) + 1u;
    if (cols < 4u) {
        cols = 4u;
    } else if (cols > 512u) {
        cols = 512u;
    }
    uint32_t rows = cols;
    uint32_t cell_count = cols * rows;

    if (
        cdb_ensure_scratch_i32(
            &g_collision_cell_head,
            &g_collision_cell_head_cap,
            cell_count
        ) != 0
        || cdb_ensure_scratch_i32(
            &g_collision_next,
            &g_collision_next_cap,
            count
        ) != 0
    ) {
        return 0u;
    }
    int32_t *cell_head = g_collision_cell_head;
    int32_t *next = g_collision_next;

    for (uint32_t cell = 0u; cell < cell_count; cell += 1u) {
        cell_head[cell] = -1;
    }

    for (uint32_t i = 0u; i < count; i += 1u) {
        float xi = x[i];
        float yi = y[i];
        if (!isfinite(xi)) {
            xi = 0.5f;
        }
        if (!isfinite(yi)) {
            yi = 0.5f;
        }
        xi = clamp01(xi);
        yi = clamp01(yi);
        x[i] = xi;
        y[i] = yi;

        uint32_t cx = (uint32_t)floorf(xi / bounded_cell);
        uint32_t cy = (uint32_t)floorf(yi / bounded_cell);
        if (cx >= cols) {
            cx = cols - 1u;
        }
        if (cy >= rows) {
            cy = rows - 1u;
        }
        uint32_t cell = (cy * cols) + cx;
        next[i] = cell_head[cell];
        cell_head[cell] = (int32_t)i;
    }

    uint32_t workers = worker_count;
    if (workers < 1u) {
        workers = 1u;
    } else if (workers > 32u) {
        workers = 32u;
    }
    if (workers > cell_count) {
        workers = cell_count;
    }
    if (workers < 1u) {
        workers = 1u;
    }

    if (count > 0u && workers > (UINT32_MAX / count)) {
        return 0u;
    }
    uint32_t work_slots = count * workers;
    if (
        cdb_ensure_scratch_float(&g_collision_dx, &g_collision_dx_cap, work_slots) != 0
        || cdb_ensure_scratch_float(&g_collision_dy, &g_collision_dy_cap, work_slots) != 0
        || cdb_ensure_scratch_float(
            &g_collision_dvx,
            &g_collision_dvx_cap,
            work_slots
        ) != 0
        || cdb_ensure_scratch_float(
            &g_collision_dvy,
            &g_collision_dvy_cap,
            work_slots
        ) != 0
        || cdb_ensure_scratch_u32(
            &g_collision_hits,
            &g_collision_hits_cap,
            work_slots
        ) != 0
    ) {
        return 0u;
    }

    memset(g_collision_dx, 0, (size_t)work_slots * sizeof(float));
    memset(g_collision_dy, 0, (size_t)work_slots * sizeof(float));
    memset(g_collision_dvx, 0, (size_t)work_slots * sizeof(float));
    memset(g_collision_dvy, 0, (size_t)work_slots * sizeof(float));
    memset(g_collision_hits, 0, (size_t)work_slots * sizeof(uint32_t));

    CDBCollisionResolveContext ctx;
    ctx.count = count;
    ctx.cols = cols;
    ctx.rows = rows;
    ctx.cell_count = cell_count;
    ctx.cell_head = cell_head;
    ctx.next = next;
    ctx.x = x;
    ctx.y = y;
    ctx.vx = vx;
    ctx.vy = vy;
    ctx.radius = radius;
    ctx.mass = mass;
    ctx.restitution = bounded_restitution;
    ctx.separation_percent = bounded_separation;

    CDBCollisionResolveTask tasks[32];
    pthread_t threads[32];
    uint8_t thread_started[32];
    memset(tasks, 0, sizeof(tasks));
    memset(threads, 0, sizeof(threads));
    memset(thread_started, 0, sizeof(thread_started));

    for (uint32_t w = 0u; w < workers; w += 1u) {
        size_t base = (size_t)w * (size_t)count;
        tasks[w].ctx = &ctx;
        tasks[w].dx = g_collision_dx + base;
        tasks[w].dy = g_collision_dy + base;
        tasks[w].dvx = g_collision_dvx + base;
        tasks[w].dvy = g_collision_dvy + base;
        tasks[w].hit_count = g_collision_hits + base;
    }

    uint32_t chunk = (cell_count + workers - 1u) / workers;
    for (uint32_t w = 0u; w < workers; w += 1u) {
        tasks[w].ctx = &ctx;
        uint32_t start = w * chunk;
        uint32_t end = start + chunk;
        if (start > cell_count) {
            start = cell_count;
        }
        if (end > cell_count) {
            end = cell_count;
        }
        tasks[w].start_cell = start;
        tasks[w].end_cell = end;
    }

    for (uint32_t w = 1u; w < workers; w += 1u) {
        if (tasks[w].start_cell >= tasks[w].end_cell) {
            continue;
        }
        if (pthread_create(&threads[w], NULL, cdb_collision_worker_entry, &tasks[w]) == 0) {
            thread_started[w] = 1u;
        } else {
            cdb_collision_process_cells(&tasks[w]);
        }
    }
    if (tasks[0].start_cell < tasks[0].end_cell) {
        cdb_collision_process_cells(&tasks[0]);
    }
    for (uint32_t w = 1u; w < workers; w += 1u) {
        if (thread_started[w] != 0u) {
            (void)pthread_join(threads[w], NULL);
        }
    }

    uint32_t collision_pairs = 0u;
    for (uint32_t w = 0u; w < workers; w += 1u) {
        collision_pairs += tasks[w].collision_pairs;
    }

    for (uint32_t i = 0u; i < count; i += 1u) {
        float sum_dx = 0.0f;
        float sum_dy = 0.0f;
        float sum_dvx = 0.0f;
        float sum_dvy = 0.0f;
        uint32_t sum_hits = 0u;
        for (uint32_t w = 0u; w < workers; w += 1u) {
            sum_dx += tasks[w].dx[i];
            sum_dy += tasks[w].dy[i];
            sum_dvx += tasks[w].dvx[i];
            sum_dvy += tasks[w].dvy[i];
            sum_hits += tasks[w].hit_count[i];
        }
        x[i] = clamp01(x[i] + sum_dx);
        y[i] = clamp01(y[i] + sum_dy);
        vx[i] += sum_dvx;
        vy[i] += sum_dvy;
        if (out_collision_count != NULL) {
            out_collision_count[i] = sum_hits;
        }
    }

    return collision_pairs;
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

static float dot_product_24(const float *a, const float *b) {
    float dot = 0.0f;
    for (int i = 0; i < 24; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

static void mix_vectors_24(float *target, const float *source, float alpha) {
    float inv = 1.0f - alpha;
    for (int i = 0; i < 24; i++) {
        target[i] = (target[i] * inv) + (source[i] * alpha);
    }
    // Normalize
    float mag_sq = 0.0f;
    for (int i = 0; i < 24; i++) {
        mag_sq += target[i] * target[i];
    }
    if (mag_sq > 1e-9f) {
        float scale = 1.0f / sqrtf(mag_sq);
        for (int i = 0; i < 24; i++) {
            target[i] *= scale;
        }
    }
}

static void cdb_quadtree_reset_node(
    CDBQuadNode *node,
    float min_x,
    float min_y,
    float max_x,
    float max_y,
    uint16_t depth
) {
    if (node == NULL) {
        return;
    }
    node->min_x = min_x;
    node->min_y = min_y;
    node->max_x = max_x;
    node->max_y = max_y;
    node->child_base = UINT32_MAX;
    node->particle_head = -1;
    node->particle_count = 0u;
    node->depth = depth;
    node->total_count = 0u;
    node->nexus_count = 0u;
    node->max_radius = 0.0f;
    node->mass = 0.0f;
    node->com_x = 0.5f * (min_x + max_x);
    node->com_y = 0.5f * (min_y + max_y);
    memset(node->emb_sum, 0, sizeof(node->emb_sum));
    node->emb_norm = 0.0f;
    node->group_mask = 0u;
}

static uint64_t cdb_group_mask_bit(uint32_t group_id) {
    return 1ull << (uint64_t)(group_id & 63u);
}

static uint32_t cdb_quadtree_choose_child(const CDBQuadNode *node, float x, float y) {
    float mid_x = 0.5f * (node->min_x + node->max_x);
    float mid_y = 0.5f * (node->min_y + node->max_y);
    uint32_t bit_x = (x >= mid_x) ? 1u : 0u;
    uint32_t bit_y = (y >= mid_y) ? 1u : 0u;
    return node->child_base + (bit_y * 2u) + bit_x;
}

static int cdb_quadtree_subdivide(
    CDBEngine *engine,
    uint32_t node_index,
    uint32_t *io_node_count
) {
    if (engine == NULL || io_node_count == NULL || engine->quad_nodes == NULL) {
        return -1;
    }
    if (node_index >= engine->quad_capacity) {
        return -1;
    }
    CDBQuadNode *node = &engine->quad_nodes[node_index];
    if (node->child_base != UINT32_MAX) {
        return 0;
    }
    if ((*io_node_count + 4u) > engine->quad_capacity) {
        return -1;
    }

    uint32_t base = *io_node_count;
    *io_node_count += 4u;
    node->child_base = base;

    float mid_x = 0.5f * (node->min_x + node->max_x);
    float mid_y = 0.5f * (node->min_y + node->max_y);
    uint16_t child_depth = (uint16_t)(node->depth + 1u);

    cdb_quadtree_reset_node(
        &engine->quad_nodes[base + 0u],
        node->min_x,
        node->min_y,
        mid_x,
        mid_y,
        child_depth
    );
    cdb_quadtree_reset_node(
        &engine->quad_nodes[base + 1u],
        mid_x,
        node->min_y,
        node->max_x,
        mid_y,
        child_depth
    );
    cdb_quadtree_reset_node(
        &engine->quad_nodes[base + 2u],
        node->min_x,
        mid_y,
        mid_x,
        node->max_y,
        child_depth
    );
    cdb_quadtree_reset_node(
        &engine->quad_nodes[base + 3u],
        mid_x,
        mid_y,
        node->max_x,
        node->max_y,
        child_depth
    );
    return 0;
}

static int cdb_quadtree_insert_particle(
    CDBEngine *engine,
    uint32_t node_index,
    uint32_t particle_index,
    const float *pos_x,
    const float *pos_y,
    uint32_t *io_node_count
) {
    if (
        engine == NULL
        || engine->quad_nodes == NULL
        || pos_x == NULL
        || pos_y == NULL
        || io_node_count == NULL
    ) {
        return -1;
    }
    if (node_index >= engine->quad_capacity || particle_index >= engine->particle_count) {
        return -1;
    }

    CDBQuadNode *node = &engine->quad_nodes[node_index];
    if (node->child_base == UINT32_MAX) {
        if (
            node->particle_count < engine->bh_leaf_capacity
            || node->depth >= engine->bh_max_depth
        ) {
            engine->grid_next[particle_index] = node->particle_head;
            node->particle_head = (int32_t)particle_index;
            node->particle_count = (uint16_t)(node->particle_count + 1u);
            return 0;
        }

        if (cdb_quadtree_subdivide(engine, node_index, io_node_count) != 0) {
            return -1;
        }

        int32_t cursor = node->particle_head;
        node->particle_head = -1;
        node->particle_count = 0u;

        while (cursor != -1) {
            uint32_t existing = (uint32_t)cursor;
            int32_t next_cursor = engine->grid_next[existing];
            engine->grid_next[existing] = -1;
            if (
                cdb_quadtree_insert_particle(
                    engine,
                    node_index,
                    existing,
                    pos_x,
                    pos_y,
                    io_node_count
                )
                != 0
            ) {
                return -1;
            }
            cursor = next_cursor;
        }
    }

    float px = pos_x[particle_index];
    float py = pos_y[particle_index];
    uint32_t child_index = cdb_quadtree_choose_child(node, px, py);
    return cdb_quadtree_insert_particle(
        engine,
        child_index,
        particle_index,
        pos_x,
        pos_y,
        io_node_count
    );
}

static void cdb_quadtree_accumulate(
    CDBEngine *engine,
    uint32_t node_index,
    const float *pos_x,
    const float *pos_y
) {
    CDBQuadNode *node = &engine->quad_nodes[node_index];
    node->total_count = 0u;
    node->nexus_count = 0u;
    node->max_radius = 0.0f;
    node->mass = 0.0f;
    node->com_x = 0.0f;
    node->com_y = 0.0f;
    memset(node->emb_sum, 0, sizeof(node->emb_sum));
    node->emb_norm = 0.0f;
    node->group_mask = 0u;

    if (node->child_base != UINT32_MAX) {
        for (uint32_t child_offset = 0u; child_offset < 4u; child_offset += 1u) {
            uint32_t child_index = node->child_base + child_offset;
            cdb_quadtree_accumulate(engine, child_index, pos_x, pos_y);
            CDBQuadNode *child = &engine->quad_nodes[child_index];
            node->total_count += child->total_count;
            node->nexus_count += child->nexus_count;
            if (child->max_radius > node->max_radius) {
                node->max_radius = child->max_radius;
            }
            node->mass += child->mass;
            node->com_x += child->com_x * child->mass;
            node->com_y += child->com_y * child->mass;
            for (uint32_t k = 0u; k < 24u; k += 1u) {
                node->emb_sum[k] += child->emb_sum[k];
            }
            node->group_mask |= child->group_mask;
        }
    } else {
        int32_t cursor = node->particle_head;
        while (cursor != -1) {
            uint32_t particle = (uint32_t)cursor;
            cursor = engine->grid_next[particle];
            node->total_count += 1u;

            float radius = engine->radius[particle];
            if (!isfinite(radius) || radius < 0.0f) {
                radius = 0.0f;
            }
            if (radius > node->max_radius) {
                node->max_radius = radius;
            }

            if ((engine->flags[particle] & CDB_FLAG_NEXUS) != 0u) {
                float particle_mass = engine->mass[particle];
                if (!isfinite(particle_mass) || particle_mass <= 0.0f) {
                    particle_mass = 0.7f;
                }
                particle_mass = fmaxf(0.05f, particle_mass);
                node->nexus_count += 1u;
                node->mass += particle_mass;
                node->com_x += pos_x[particle] * particle_mass;
                node->com_y += pos_y[particle] * particle_mass;
                const float *embedding = &engine->embeddings[particle * 24u];
                for (uint32_t k = 0u; k < 24u; k += 1u) {
                    node->emb_sum[k] += embedding[k] * particle_mass;
                }
                node->group_mask |= cdb_group_mask_bit(engine->group_id[particle]);
            }
        }
    }

    if (node->mass > 1e-9f) {
        node->com_x /= node->mass;
        node->com_y /= node->mass;
    } else {
        node->com_x = 0.5f * (node->min_x + node->max_x);
        node->com_y = 0.5f * (node->min_y + node->max_y);
    }

    float emb_sq = 0.0f;
    for (uint32_t k = 0u; k < 24u; k += 1u) {
        emb_sq += node->emb_sum[k] * node->emb_sum[k];
    }
    node->emb_norm = (emb_sq > 1e-9f) ? sqrtf(emb_sq) : 0.0f;
}

static int cdb_build_quadtree_frame(
    CDBEngine *engine,
    const float *pos_x,
    const float *pos_y,
    uint32_t count,
    uint32_t *out_node_count
) {
    if (
        engine == NULL
        || engine->quad_nodes == NULL
        || pos_x == NULL
        || pos_y == NULL
        || out_node_count == NULL
        || count == 0u
    ) {
        if (out_node_count != NULL) {
            *out_node_count = 0u;
        }
        return -1;
    }

    for (uint32_t i = 0u; i < count; i += 1u) {
        engine->grid_next[i] = -1;
    }

    float min_x = pos_x[0];
    float max_x = pos_x[0];
    float min_y = pos_y[0];
    float max_y = pos_y[0];
    for (uint32_t i = 1u; i < count; i += 1u) {
        float x = pos_x[i];
        float y = pos_y[i];
        if (x < min_x) {
            min_x = x;
        }
        if (x > max_x) {
            max_x = x;
        }
        if (y < min_y) {
            min_y = y;
        }
        if (y > max_y) {
            max_y = y;
        }
    }

    float span = fmaxf(max_x - min_x, max_y - min_y);
    span = fmaxf(0.02f, span);
    float half = (span * 0.5f) + 0.0005f;
    float center_x = 0.5f * (min_x + max_x);
    float center_y = 0.5f * (min_y + max_y);

    cdb_quadtree_reset_node(
        &engine->quad_nodes[0],
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
        0u
    );

    uint32_t node_count = 1u;
    for (uint32_t i = 0u; i < count; i += 1u) {
        if (cdb_quadtree_insert_particle(engine, 0u, i, pos_x, pos_y, &node_count) != 0) {
            *out_node_count = 0u;
            return -1;
        }
    }

    cdb_quadtree_accumulate(engine, 0u, pos_x, pos_y);
    *out_node_count = node_count;
    return 0;
}

static float cdb_point_aabb_distance_sq(float x, float y, const CDBQuadNode *node) {
    float dx = 0.0f;
    float dy = 0.0f;
    if (x < node->min_x) {
        dx = node->min_x - x;
    } else if (x > node->max_x) {
        dx = x - node->max_x;
    }
    if (y < node->min_y) {
        dy = node->min_y - y;
    } else if (y > node->max_y) {
        dy = y - node->max_y;
    }
    return (dx * dx) + (dy * dy);
}

static void cdb_apply_collision_force_from_tree(
    const CDBEngine *engine,
    const CDBQuadNode *nodes,
    uint32_t node_index,
    const float *pos_x,
    const float *pos_y,
    uint32_t target,
    float xi,
    float yi,
    float radius_i,
    float *io_fx,
    float *io_fy
) {
    const CDBQuadNode *node = &nodes[node_index];
    if (node->total_count == 0u) {
        return;
    }

    float max_interaction = radius_i + node->max_radius;
    if (max_interaction <= 0.0f) {
        return;
    }
    float max_dist_sq = max_interaction * max_interaction;
    if (cdb_point_aabb_distance_sq(xi, yi, node) > max_dist_sq) {
        return;
    }

    if (node->child_base != UINT32_MAX) {
        for (uint32_t child_offset = 0u; child_offset < 4u; child_offset += 1u) {
            cdb_apply_collision_force_from_tree(
                engine,
                nodes,
                node->child_base + child_offset,
                pos_x,
                pos_y,
                target,
                xi,
                yi,
                radius_i,
                io_fx,
                io_fy
            );
        }
        return;
    }

    int32_t cursor = node->particle_head;
    while (cursor != -1) {
        uint32_t other = (uint32_t)cursor;
        cursor = engine->grid_next[other];
        if (other == target) {
            continue;
        }
        float radius_other = engine->radius[other];
        if (!isfinite(radius_other) || radius_other < 0.0f) {
            radius_other = 0.0f;
        }
        float r_sum = radius_i + radius_other;
        if (r_sum <= 0.0f) {
            continue;
        }

        float dx = pos_x[other] - xi;
        float dy = pos_y[other] - yi;
        float dist_sq = (dx * dx) + (dy * dy);
        float min_dist_sq = r_sum * r_sum;
        if (dist_sq >= min_dist_sq) {
            continue;
        }

        float nx = 0.0f;
        float ny = 0.0f;
        float dist = 0.0f;
        if (dist_sq > 1e-10f) {
            dist = sqrtf(dist_sq);
            nx = dx / dist;
            ny = dy / dist;
        } else {
            uint32_t jitter_seed = mix_hash_u32(
                ((target + 1u) * 73856093u)
                ^ ((other + 1u) * 19349663u)
                ^ engine->seed
            );
            float angle = ((float)(jitter_seed % 6283u)) * 0.001f;
            nx = cosf(angle);
            ny = sinf(angle);
            dist = 0.0f;
        }

        float overlap = fmaxf(0.0f, r_sum - dist);
        float force_mag = engine->collision_spring * overlap;
        *io_fx -= nx * force_mag;
        *io_fy -= ny * force_mag;
    }
}

static void cdb_apply_barnes_hut_force(
    const CDBEngine *engine,
    const CDBQuadNode *nodes,
    uint32_t node_index,
    const float *pos_x,
    const float *pos_y,
    uint32_t target,
    float xi,
    float yi,
    const float *embedding,
    float grav_const,
    float grav_eps,
    float theta,
    float *io_fx,
    float *io_fy
) {
    const CDBQuadNode *node = &nodes[node_index];
    if (node->nexus_count == 0u) {
        return;
    }

    if (node->child_base == UINT32_MAX) {
        int32_t cursor = node->particle_head;
        while (cursor != -1) {
            uint32_t other = (uint32_t)cursor;
            cursor = engine->grid_next[other];
            if (other == target || (engine->flags[other] & CDB_FLAG_NEXUS) == 0u) {
                continue;
            }

            float dx = pos_x[other] - xi;
            float dy = pos_y[other] - yi;
            float dist_sq = (dx * dx) + (dy * dy);
            const float *other_embedding = &engine->embeddings[other * 24u];
            float kappa = dot_product_24(embedding, other_embedding);
            float particle_mass = engine->mass[other];
            if (!isfinite(particle_mass) || particle_mass <= 0.0f) {
                particle_mass = 0.7f;
            }
            particle_mass = fmaxf(0.05f, particle_mass);
            float force = (grav_const * kappa * particle_mass) / (dist_sq + grav_eps);
            *io_fx += dx * force;
            *io_fy += dy * force;
        }
        return;
    }

    float dx = node->com_x - xi;
    float dy = node->com_y - yi;
    float dist_sq = (dx * dx) + (dy * dy);
    float size = fmaxf(node->max_x - node->min_x, node->max_y - node->min_y);
    int contains_target = (
        xi >= node->min_x && xi <= node->max_x
        && yi >= node->min_y && yi <= node->max_y
    );

    if (!contains_target && dist_sq > 1e-9f && (size * size) <= ((theta * theta) * dist_sq)) {
        if (node->emb_norm > 1e-9f && node->mass > 0.0f) {
            float dot = 0.0f;
            for (uint32_t k = 0u; k < 24u; k += 1u) {
                dot += embedding[k] * node->emb_sum[k];
            }
            float kappa = dot / node->emb_norm;
            float force = (grav_const * kappa * node->mass) / (dist_sq + grav_eps);
            *io_fx += dx * force;
            *io_fy += dy * force;
        }
        return;
    }

    for (uint32_t child_offset = 0u; child_offset < 4u; child_offset += 1u) {
        cdb_apply_barnes_hut_force(
            engine,
            nodes,
            node->child_base + child_offset,
            pos_x,
            pos_y,
            target,
            xi,
            yi,
            embedding,
            grav_const,
            grav_eps,
            theta,
            io_fx,
            io_fy
        );
    }
}

static void cdb_apply_group_spring_forces_exact(
    const CDBEngine *engine,
    const float *pos_x,
    const float *pos_y,
    uint32_t target,
    float xi,
    float yi,
    const float *embedding,
    float *io_fx,
    float *io_fy
) {
    if ((engine->flags[target] & CDB_FLAG_NEXUS) == 0u) {
        return;
    }
    float rest_length = engine->cluster_rest_length;
    float k0 = 0.05f * engine->cluster_stiffness;
    if (!isfinite(rest_length) || rest_length <= 0.0f) {
        rest_length = 0.08f;
    }
    if (!isfinite(k0) || k0 <= 0.0f) {
        k0 = 0.05f;
    }

    uint32_t group_i = engine->group_id[target];
    uint32_t count = engine->particle_count;
    for (uint32_t other = 0u; other < count; other += 1u) {
        if (other == target || (engine->flags[other] & CDB_FLAG_NEXUS) == 0u) {
            continue;
        }
        if (engine->group_id[other] != group_i) {
            continue;
        }

        float dx = pos_x[other] - xi;
        float dy = pos_y[other] - yi;
        float dist = sqrtf((dx * dx) + (dy * dy)) + 1e-9f;
        const float *other_embedding = &engine->embeddings[other * 24u];
        float sim = dot_product_24(embedding, other_embedding);
        float beta = 1.2f;
        float ke = k0 * powf(clamp01(1.0f - sim), beta);
        float f_edge = ke * (dist - rest_length);
        *io_fx += (dx / dist) * f_edge;
        *io_fy += (dy / dist) * f_edge;
    }
}

static void cdb_apply_group_cluster_force_bh(
    const CDBEngine *engine,
    const CDBQuadNode *nodes,
    uint32_t node_index,
    const float *pos_x,
    const float *pos_y,
    uint32_t target,
    float xi,
    float yi,
    uint32_t target_group,
    const float *embedding,
    float theta,
    float *io_fx,
    float *io_fy
) {
    if ((engine->flags[target] & CDB_FLAG_NEXUS) == 0u) {
        return;
    }

    const CDBQuadNode *node = &nodes[node_index];
    if (node->nexus_count == 0u) {
        return;
    }

    uint64_t target_mask = cdb_group_mask_bit(target_group);
    if ((node->group_mask & target_mask) == 0u) {
        return;
    }

    float rest_length = engine->cluster_rest_length;
    float k0 = 0.05f * engine->cluster_stiffness;
    if (!isfinite(rest_length) || rest_length <= 0.0f) {
        rest_length = 0.08f;
    }
    if (!isfinite(k0) || k0 <= 0.0f) {
        k0 = 0.05f;
    }

    if (node->child_base == UINT32_MAX) {
        int32_t cursor = node->particle_head;
        while (cursor != -1) {
            uint32_t other = (uint32_t)cursor;
            cursor = engine->grid_next[other];
            if (other == target || (engine->flags[other] & CDB_FLAG_NEXUS) == 0u) {
                continue;
            }
            if (engine->group_id[other] != target_group) {
                continue;
            }

            float dx = pos_x[other] - xi;
            float dy = pos_y[other] - yi;
            float dist_sq = (dx * dx) + (dy * dy);
            float dist = 0.0f;
            float nx = 0.0f;
            float ny = 0.0f;
            if (dist_sq > 1e-10f) {
                dist = sqrtf(dist_sq);
                nx = dx / dist;
                ny = dy / dist;
            } else {
                uint32_t jitter_seed = mix_hash_u32(
                    ((target + 1u) * 2654435761u)
                    ^ ((other + 1u) * 2246822519u)
                    ^ engine->seed
                );
                float angle = ((float)(jitter_seed % 6283u)) * 0.001f;
                nx = cosf(angle);
                ny = sinf(angle);
                dist = 0.0f;
            }

            const float *other_embedding = &engine->embeddings[other * 24u];
            float sim = dot_product_24(embedding, other_embedding);
            float ke = k0 * powf(clamp01(1.0f - sim), 1.2f);
            float f_edge = ke * (dist - rest_length);
            *io_fx += nx * f_edge;
            *io_fy += ny * f_edge;
        }
        return;
    }

    float dx = node->com_x - xi;
    float dy = node->com_y - yi;
    float dist_sq = (dx * dx) + (dy * dy);
    float size = fmaxf(node->max_x - node->min_x, node->max_y - node->min_y);
    int contains_target = (
        xi >= node->min_x && xi <= node->max_x
        && yi >= node->min_y && yi <= node->max_y
    );
    int node_is_single_group = (node->group_mask == target_mask);

    if (
        !contains_target
        && node_is_single_group
        && dist_sq > 1e-9f
        && (size * size) <= ((theta * theta) * dist_sq)
    ) {
        if (node->emb_norm > 1e-9f && node->nexus_count > 0u) {
            float dot = 0.0f;
            for (uint32_t k = 0u; k < 24u; k += 1u) {
                dot += embedding[k] * node->emb_sum[k];
            }
            float sim = dot / node->emb_norm;
            float dist = sqrtf(dist_sq) + 1e-9f;
            float ke = k0 * powf(clamp01(1.0f - sim), 1.2f);
            float node_scale = (float)node->nexus_count;
            float f_edge = ke * (dist - rest_length) * node_scale;
            *io_fx += (dx / dist) * f_edge;
            *io_fy += (dy / dist) * f_edge;
        }
        return;
    }

    for (uint32_t child_offset = 0u; child_offset < 4u; child_offset += 1u) {
        cdb_apply_group_cluster_force_bh(
            engine,
            nodes,
            node->child_base + child_offset,
            pos_x,
            pos_y,
            target,
            xi,
            yi,
            target_group,
            embedding,
            theta,
            io_fx,
            io_fy
        );
    }
}

typedef struct CDBForceEvalContext {
    CDBEngine *engine;
    const float *pos_x;
    const float *pos_y;
    float *acc_x;
    float *acc_y;
    const float *nooi;
    uint32_t count;
    uint32_t node_count;
    uint32_t sim_flags;
} CDBForceEvalContext;

typedef struct CDBForceWorkerTask {
    const CDBForceEvalContext *ctx;
    uint32_t start;
    uint32_t end;
} CDBForceWorkerTask;

static void cdb_force_eval_range(const CDBForceEvalContext *ctx, uint32_t start, uint32_t end) {
    if (ctx == NULL || ctx->engine == NULL) {
        return;
    }

    CDBEngine *engine = ctx->engine;
    const float *pos_x = ctx->pos_x;
    const float *pos_y = ctx->pos_y;
    float *acc_x = ctx->acc_x;
    float *acc_y = ctx->acc_y;
    const float *nooi = ctx->nooi;
    uint32_t count = ctx->count;
    float grav_const = engine->grav_const;
    float grav_eps = engine->grav_eps;
    float bh_theta = engine->bh_theta;
    float cluster_theta = engine->cluster_theta;
    if (!isfinite(bh_theta) || bh_theta <= 0.0f) {
        bh_theta = 0.62f;
    }
    if (!isfinite(cluster_theta) || cluster_theta <= 0.0f) {
        cluster_theta = 0.68f;
    }

    if (end > count) {
        end = count;
    }

    for (uint32_t i = start; i < end; i += 1u) {
        float xi = pos_x[i];
        float yi = pos_y[i];
        float fx = 0.0f;
        float fy = 0.0f;

        if (nooi != NULL) {
            int cx = fast_floor_to_int(xi * (float)CDB_NOOI_COLS);
            int cy = fast_floor_to_int(yi * (float)CDB_NOOI_ROWS);
            if (cx >= 0 && cx < CDB_NOOI_COLS && cy >= 0 && cy < CDB_NOOI_ROWS) {
                uint32_t owner = engine->owner_id[i];
                int layer = (int)(owner % CDB_NOOI_LAYERS);
                size_t layer_stride = (size_t)(CDB_NOOI_COLS * CDB_NOOI_ROWS * 2);
                size_t cell_offset = (size_t)((cy * CDB_NOOI_COLS + cx) * 2);
                size_t idx = (size_t)(layer * layer_stride) + cell_offset;
                if (idx + 1u < (size_t)CDB_NOOI_SIZE) {
                    float f_scale = 6.5f;
                    fx += nooi[idx] * f_scale;
                    fy += nooi[idx + 1u] * f_scale;
                }
            }
        }

        float radius_i = engine->radius[i];
        if (!isfinite(radius_i) || radius_i < 0.0f) {
            radius_i = 0.0f;
        }

        const float *embedding = &engine->embeddings[i * 24u];
        if (ctx->node_count > 0u) {
            if ((ctx->sim_flags & CDB_SIM_FLAG_COLLISION) != 0u) {
                cdb_apply_collision_force_from_tree(
                    engine,
                    engine->quad_nodes,
                    0u,
                    pos_x,
                    pos_y,
                    i,
                    xi,
                    yi,
                    radius_i,
                    &fx,
                    &fy
                );
            }
            cdb_apply_barnes_hut_force(
                engine,
                engine->quad_nodes,
                0u,
                pos_x,
                pos_y,
                i,
                xi,
                yi,
                embedding,
                grav_const,
                grav_eps,
                bh_theta,
                &fx,
                &fy
            );
        } else {
            if ((ctx->sim_flags & CDB_SIM_FLAG_COLLISION) != 0u) {
                for (uint32_t other = 0u; other < count; other += 1u) {
                    if (other == i) {
                        continue;
                    }
                    float radius_other = engine->radius[other];
                    if (!isfinite(radius_other) || radius_other < 0.0f) {
                        radius_other = 0.0f;
                    }
                    float r_sum = radius_i + radius_other;
                    float dx = pos_x[other] - xi;
                    float dy = pos_y[other] - yi;
                    float dist_sq = (dx * dx) + (dy * dy);
                    if (dist_sq >= (r_sum * r_sum) || r_sum <= 0.0f) {
                        continue;
                    }
                    float dist = sqrtf(fmaxf(dist_sq, 1e-10f));
                    float overlap = fmaxf(0.0f, r_sum - dist);
                    float nx = dx / dist;
                    float ny = dy / dist;
                    float force_mag = engine->collision_spring * overlap;
                    fx -= nx * force_mag;
                    fy -= ny * force_mag;
                }
            }
            for (uint32_t other = 0u; other < count; other += 1u) {
                if (other == i || (engine->flags[other] & CDB_FLAG_NEXUS) == 0u) {
                    continue;
                }
                float dx = pos_x[other] - xi;
                float dy = pos_y[other] - yi;
                float dist_sq = (dx * dx) + (dy * dy);
                const float *other_embedding = &engine->embeddings[other * 24u];
                float kappa = dot_product_24(embedding, other_embedding);
                float particle_mass = engine->mass[other];
                if (!isfinite(particle_mass) || particle_mass <= 0.0f) {
                    particle_mass = 0.7f;
                }
                particle_mass = fmaxf(0.05f, particle_mass);
                float force = (grav_const * kappa * particle_mass) / (dist_sq + grav_eps);
                fx += dx * force;
                fy += dy * force;
            }
        }

        if (ctx->node_count > 0u) {
            cdb_apply_group_cluster_force_bh(
                engine,
                engine->quad_nodes,
                0u,
                pos_x,
                pos_y,
                i,
                xi,
                yi,
                engine->group_id[i],
                embedding,
                cluster_theta,
                &fx,
                &fy
            );
        } else {
            cdb_apply_group_spring_forces_exact(
                engine,
                pos_x,
                pos_y,
                i,
                xi,
                yi,
                embedding,
                &fx,
                &fy
            );
        }

        acc_x[i] = fx;
        acc_y[i] = fy;
    }
}

static void *cdb_force_worker_entry(void *arg) {
    CDBForceWorkerTask *task = (CDBForceWorkerTask *)arg;
    if (task == NULL || task->ctx == NULL) {
        return NULL;
    }
    cdb_force_eval_range(task->ctx, task->start, task->end);
    return NULL;
}

static void cdb_force_parallel_apply(const CDBForceEvalContext *ctx) {
    if (ctx == NULL || ctx->engine == NULL || ctx->count == 0u) {
        return;
    }

    uint32_t worker_count = ctx->engine->force_worker_count;
    if (worker_count < 1u) {
        worker_count = 1u;
    }
    worker_count = (worker_count > 32u) ? 32u : worker_count;
    worker_count = (worker_count > ctx->count) ? ctx->count : worker_count;

    if (worker_count <= 1u || ctx->count < 96u) {
        cdb_force_eval_range(ctx, 0u, ctx->count);
        return;
    }

    CDBForceWorkerTask tasks[32];
    pthread_t threads[32];
    uint32_t active_workers[32];
    uint32_t active_count = 0u;
    uint32_t chunk = (ctx->count + worker_count - 1u) / worker_count;

    for (uint32_t worker = 1u; worker < worker_count; worker += 1u) {
        uint32_t start = worker * chunk;
        if (start >= ctx->count) {
            continue;
        }
        uint32_t end = start + chunk;
        if (end > ctx->count) {
            end = ctx->count;
        }
        tasks[worker].ctx = ctx;
        tasks[worker].start = start;
        tasks[worker].end = end;
        if (pthread_create(&threads[worker], NULL, cdb_force_worker_entry, &tasks[worker]) == 0) {
            active_workers[active_count] = worker;
            active_count += 1u;
        } else {
            cdb_force_eval_range(ctx, start, end);
        }
    }

    uint32_t main_end = chunk;
    if (main_end > ctx->count) {
        main_end = ctx->count;
    }
    cdb_force_eval_range(ctx, 0u, main_end);

    for (uint32_t i = 0u; i < active_count; i += 1u) {
        uint32_t worker = active_workers[i];
        pthread_join(threads[worker], NULL);
    }
}

static void *force_system_worker(void *arg) {
    CDBEngine *engine = (CDBEngine *)arg;
    if (engine == NULL) {
        return NULL;
    }

    while (atomic_load_explicit(&engine->running, memory_order_acquire) != 0) {
        int pos_index = atomic_load_explicit(&engine->position.readable, memory_order_acquire);
        int write_index =
            1 - atomic_load_explicit(&engine->acceleration.readable, memory_order_relaxed);

        const float *pos_x = engine->position.x[pos_index];
        const float *pos_y = engine->position.y[pos_index];
        float *acc_x = engine->acceleration.x[write_index];
        float *acc_y = engine->acceleration.y[write_index];
        uint32_t count = engine->particle_count;
        uint32_t flags = atomic_load_explicit(
            (_Atomic uint32_t *)&engine->sim_flags,
            memory_order_relaxed
        );

        uint32_t node_count = 0u;
        int tree_ok = cdb_build_quadtree_frame(engine, pos_x, pos_y, count, &node_count);

        int nooi_idx = atomic_load_explicit(&engine->nooi_readable, memory_order_acquire);
        const float *nooi = engine->nooi_buffer[nooi_idx];

        CDBForceEvalContext ctx;
        ctx.engine = engine;
        ctx.pos_x = pos_x;
        ctx.pos_y = pos_y;
        ctx.acc_x = acc_x;
        ctx.acc_y = acc_y;
        ctx.nooi = nooi;
        ctx.count = count;
        ctx.node_count = (tree_ok == 0) ? node_count : 0u;
        ctx.sim_flags = flags;

        cdb_force_parallel_apply(&ctx);

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
        int pos_index = atomic_load_explicit(&engine->position.readable, memory_order_acquire);
        int write_index = 1 - atomic_load_explicit(&engine->noise.readable, memory_order_relaxed);
        const float *pos_x = engine->position.x[pos_index];
        const float *pos_y = engine->position.y[pos_index];
        float *noise_x = engine->noise.x[write_index];
        float *noise_y = engine->noise.y[write_index];
        uint32_t count = engine->particle_count;

        phase += 0.0175f;
        if (phase > 1000000.0f) {
            phase = 0.0f;
        }

        for (uint32_t i = 0; i < count; i += 1) {
            float amp = ((engine->flags[i] & CDB_FLAG_CHAOS) != 0u) ? 0.12f : 0.035f;
            float harmonic_seed = (float)((i * 37u) % 8192u) * 0.0008f;
            float harmonic_x = sinf(phase + harmonic_seed);
            float harmonic_y = cosf((phase * 0.91f) + harmonic_seed);

            float px = clamp01(pos_x[i]);
            float py = clamp01(pos_y[i]);
            uint32_t simplex_seed = engine->seed ^ ((i + 1u) * 2246822519u);
            float simplex_phase = phase * 0.37f;
            float simplex_x = simplex_noise_2d(
                (px * 4.4f) + simplex_phase,
                (py * 4.4f) + (simplex_phase * 0.73f),
                simplex_seed
            );
            float simplex_y = simplex_noise_2d(
                (px * 4.4f) + 17.0f + (simplex_phase * 0.61f),
                (py * 4.4f) + 11.0f + simplex_phase,
                simplex_seed + 101u
            );

            float simplex_mix = ((engine->flags[i] & CDB_FLAG_CHAOS) != 0u) ? 0.62f : 0.36f;
            noise_x[i] = ((harmonic_x * (1.0f - simplex_mix)) + (simplex_x * simplex_mix)) * amp;
            noise_y[i] = ((harmonic_y * (1.0f - simplex_mix)) + (simplex_y * simplex_mix)) * amp;
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
        float daimon_friction = engine->daimon_friction;
        float nexus_friction = engine->nexus_friction;

        for (uint32_t i = 0; i < count; i += 1) {
            float friction = ((engine->flags[i] & CDB_FLAG_NEXUS) != 0u) ? nexus_friction : daimon_friction;
            
            // Spec: v = (1 - gamma * dt)v + a * dt
            float vx = (1.0f - friction * dt) * vel_x[i] + ((acc_x[i] + noise_x[i]) * dt);
            float vy = (1.0f - friction * dt) * vel_y[i] + ((acc_y[i] + noise_y[i]) * dt);

            float edge_fx = 0.0f;
            float edge_fy = 0.0f;
            if (pos_x[i] < CDB_WORLD_EDGE_BAND) {
                edge_fx += ((CDB_WORLD_EDGE_BAND - pos_x[i]) / CDB_WORLD_EDGE_BAND)
                    * CDB_WORLD_EDGE_PRESSURE;
            } else if (pos_x[i] > (1.0f - CDB_WORLD_EDGE_BAND)) {
                edge_fx -= ((pos_x[i] - (1.0f - CDB_WORLD_EDGE_BAND)) / CDB_WORLD_EDGE_BAND)
                    * CDB_WORLD_EDGE_PRESSURE;
            }
            if (pos_y[i] < CDB_WORLD_EDGE_BAND) {
                edge_fy += ((CDB_WORLD_EDGE_BAND - pos_y[i]) / CDB_WORLD_EDGE_BAND)
                    * CDB_WORLD_EDGE_PRESSURE;
            } else if (pos_y[i] > (1.0f - CDB_WORLD_EDGE_BAND)) {
                edge_fy -= ((pos_y[i] - (1.0f - CDB_WORLD_EDGE_BAND)) / CDB_WORLD_EDGE_BAND)
                    * CDB_WORLD_EDGE_PRESSURE;
            }
            vx += edge_fx * dt;
            vy += edge_fy * dt;

            float x = pos_x[i] + (vx * dt);
            float y = pos_y[i] + (vy * dt);
            if (x < 0.0f) {
                x = -x;
                vx = fabsf(vx) * CDB_WORLD_EDGE_BOUNCE;
            } else if (x > 1.0f) {
                x = 2.0f - x;
                vx = -fabsf(vx) * CDB_WORLD_EDGE_BOUNCE;
            }
            if (y < 0.0f) {
                y = -y;
                vy = fabsf(vy) * CDB_WORLD_EDGE_BOUNCE;
            } else if (y > 1.0f) {
                y = 2.0f - y;
                vy = -fabsf(vy) * CDB_WORLD_EDGE_BOUNCE;
            }

            x = clamp01(x);
            y = clamp01(y);

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

    engine->daimon_friction = 1.2f;
    engine->nexus_friction = 4.5f;
    engine->grav_const = 0.0025f;
    engine->grav_eps = 0.0015f;
    engine->bh_theta = env_f32("CDB_BH_THETA", 0.62f);
    if (!isfinite(engine->bh_theta) || engine->bh_theta < 0.2f) {
        engine->bh_theta = 0.2f;
    } else if (engine->bh_theta > 1.4f) {
        engine->bh_theta = 1.4f;
    }
    engine->bh_leaf_capacity = env_u32("CDB_BH_LEAF_CAP", 8u);
    if (engine->bh_leaf_capacity < 1u) {
        engine->bh_leaf_capacity = 1u;
    } else if (engine->bh_leaf_capacity > 64u) {
        engine->bh_leaf_capacity = 64u;
    }
    engine->bh_max_depth = env_u32("CDB_BH_MAX_DEPTH", 12u);
    if (engine->bh_max_depth < 4u) {
        engine->bh_max_depth = 4u;
    } else if (engine->bh_max_depth > 24u) {
        engine->bh_max_depth = 24u;
    }
    engine->collision_spring = env_f32("CDB_COLLISION_SPRING", 80.0f);
    if (!isfinite(engine->collision_spring) || engine->collision_spring < 1.0f) {
        engine->collision_spring = 1.0f;
    } else if (engine->collision_spring > 250.0f) {
        engine->collision_spring = 250.0f;
    }
    engine->cluster_theta = env_f32("CDB_CLUSTER_THETA", 0.68f);
    if (!isfinite(engine->cluster_theta) || engine->cluster_theta < 0.2f) {
        engine->cluster_theta = 0.2f;
    } else if (engine->cluster_theta > 1.6f) {
        engine->cluster_theta = 1.6f;
    }
    engine->cluster_rest_length = env_f32("CDB_CLUSTER_REST_LENGTH", 0.08f);
    if (!isfinite(engine->cluster_rest_length) || engine->cluster_rest_length < 0.01f) {
        engine->cluster_rest_length = 0.01f;
    } else if (engine->cluster_rest_length > 0.4f) {
        engine->cluster_rest_length = 0.4f;
    }
    engine->cluster_stiffness = env_f32("CDB_CLUSTER_STIFFNESS", 1.0f);
    if (!isfinite(engine->cluster_stiffness) || engine->cluster_stiffness < 0.1f) {
        engine->cluster_stiffness = 0.1f;
    } else if (engine->cluster_stiffness > 4.0f) {
        engine->cluster_stiffness = 4.0f;
    }

    uint32_t default_force_workers = detect_cpu_count();
    if (default_force_workers > 4u) {
        default_force_workers -= 3u;
    }
    if (default_force_workers < 1u) {
        default_force_workers = 1u;
    }
    engine->force_worker_count = env_u32("CDB_FORCE_WORKERS", default_force_workers);
    if (engine->force_worker_count < 1u) {
        engine->force_worker_count = 1u;
    } else if (engine->force_worker_count > 32u) {
        engine->force_worker_count = 32u;
    }

    engine->quad_capacity = (particle_count * 16u) + 128u;
    if (engine->quad_capacity < 256u) {
        engine->quad_capacity = 256u;
    }

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

    engine->nooi_buffer[0] = (float *)calloc(CDB_NOOI_SIZE, sizeof(float));
    engine->nooi_buffer[1] = (float *)calloc(CDB_NOOI_SIZE, sizeof(float));
    atomic_store_explicit(&engine->nooi_readable, 0, memory_order_release);

    if (engine->nooi_buffer[0] == NULL || engine->nooi_buffer[1] == NULL) {
        vec2_buffer_destroy(&engine->position);
        vec2_buffer_destroy(&engine->velocity);
        vec2_buffer_destroy(&engine->acceleration);
        vec2_buffer_destroy(&engine->noise);
        vec2_buffer_destroy(&engine->action_prob);
        vec2_buffer_destroy(&engine->particle_metrics);
        free(engine->nooi_buffer[0]);
        free(engine->nooi_buffer[1]);
        free(engine);
        return NULL;
    }

    engine->owner_id = (uint32_t *)calloc((size_t)particle_count, sizeof(uint32_t));
    engine->group_id = (uint32_t *)calloc((size_t)particle_count, sizeof(uint32_t));
    engine->flags = (uint32_t *)calloc((size_t)particle_count, sizeof(uint32_t));
    engine->mass = (float *)calloc((size_t)particle_count, sizeof(float));
    engine->radius = (float *)calloc((size_t)particle_count, sizeof(float));
    engine->embeddings = (float *)calloc((size_t)particle_count * 24, sizeof(float));
    engine->grid_head = (int32_t *)calloc(CDB_NOOI_COLS * CDB_NOOI_ROWS, sizeof(int32_t));
    engine->grid_next = (int32_t *)calloc((size_t)particle_count, sizeof(int32_t));
    engine->quad_nodes = (CDBQuadNode *)calloc((size_t)engine->quad_capacity, sizeof(CDBQuadNode));

    if (
        engine->owner_id == NULL
        || engine->group_id == NULL
        || engine->flags == NULL
        || engine->mass == NULL
        || engine->radius == NULL
        || engine->embeddings == NULL
        || engine->grid_head == NULL
        || engine->grid_next == NULL
        || engine->quad_nodes == NULL
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

    free(engine->nooi_buffer[0]);
    free(engine->nooi_buffer[1]);

    free(engine->owner_id);
    free(engine->group_id);
    free(engine->flags);
    free(engine->mass);
    free(engine->radius);
    free(engine->embeddings);
    free(engine->grid_head);
    free(engine->grid_next);
    free(engine->quad_nodes);

    free(engine);
}

int cdb_engine_update_nooi(CDBEngine *engine, const float *data) {
    if (engine == NULL || data == NULL) {
        return -1;
    }
    
    int current_read = atomic_load_explicit(&engine->nooi_readable, memory_order_relaxed);
    int write_idx = 1 - current_read;
    float *buffer = engine->nooi_buffer[write_idx];
    
    if (buffer == NULL) {
        return -1;
    }
    
    memcpy(buffer, data, CDB_NOOI_SIZE * sizeof(float));
    atomic_store_explicit(&engine->nooi_readable, write_idx, memory_order_release);
    return 0;
}

int cdb_engine_update_embeddings(CDBEngine *engine, const float *data) {
    if (engine == NULL || data == NULL) {
        return -1;
    }
    // Expected size: particle_count * 24
    memcpy(engine->embeddings, data, (size_t)engine->particle_count * 24 * sizeof(float));
    return 0;
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
