#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <stddef.h>

#if __has_include(<hiredis/hiredis.h>)
#include <hiredis/hiredis.h>
#else
typedef struct redisContext {
    int err;
    char errstr[128];
} redisContext;

typedef struct redisReply {
    int type;
    size_t elements;
    struct redisReply **element;
    char *str;
    long long integer;
} redisReply;

#define REDIS_REPLY_STRING 1
#define REDIS_REPLY_ARRAY 2
#define REDIS_REPLY_NIL 4
#define REDIS_REPLY_ERROR 6

redisContext *redisConnect(const char *ip, int port);
void redisFree(redisContext *c);
void *redisCommand(redisContext *c, const char *format, ...);
void freeReplyObject(void *reply);
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void sleep_microseconds(unsigned int microseconds) {
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

static int safe_int(const char *text, int fallback) {
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text) {
        return fallback;
    }
    return (int)value;
}

static double safe_double(const char *text, double fallback) {
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text) {
        return fallback;
    }
    return value;
}

static const char *env_or_default(const char *name, const char *fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return value;
}

static int compute_sim_point_budget(double cpu_utilization, int max_sim_points) {
    int max_points = max_sim_points > 0 ? max_sim_points : 800;
    if (cpu_utilization >= 90.0) {
        int scaled = (int)((double)max_points * 0.55);
        if (scaled < 256) {
            scaled = 256;
        }
        return scaled;
    }
    if (cpu_utilization >= 78.0) {
        int scaled = (int)((double)max_points * 0.74);
        if (scaled < 320) {
            scaled = 320;
        }
        return scaled;
    }
    return max_points;
}

static redisContext *connect_redis(const char *host, int port, const char *password) {
    redisContext *ctx = redisConnect(host, port);
    if (ctx == NULL || ctx->err) {
        if (ctx != NULL) {
            fprintf(stderr, "redis connect failed: %s\n", ctx->errstr);
            redisFree(ctx);
        } else {
            fprintf(stderr, "redis connect failed: context allocation\n");
        }
        return NULL;
    }

    if (password != NULL && password[0] != '\0') {
        redisReply *auth = redisCommand(ctx, "AUTH %s", password);
        if (auth == NULL || auth->type == REDIS_REPLY_ERROR) {
            fprintf(stderr, "redis auth failed\n");
            if (auth != NULL) {
                freeReplyObject(auth);
            }
            redisFree(ctx);
            return NULL;
        }
        freeReplyObject(auth);
    }

    return ctx;
}

static const char *field_value(redisReply *fields, const char *key) {
    if (fields == NULL || fields->type != REDIS_REPLY_ARRAY) {
        return NULL;
    }
    for (size_t index = 0; index + 1 < fields->elements; index += 2) {
        redisReply *field = fields->element[index];
        redisReply *value = fields->element[index + 1];
        if (field == NULL || value == NULL) {
            continue;
        }
        if (field->type != REDIS_REPLY_STRING || value->type != REDIS_REPLY_STRING) {
            continue;
        }
        if (strcmp(field->str, key) == 0) {
            return value->str;
        }
    }
    return NULL;
}

int main(void) {
    const char *host = env_or_default("SIM_SLICE_REDIS_HOST", "127.0.0.1");
    const int port = safe_int(env_or_default("SIM_SLICE_REDIS_PORT", "6379"), 6379);
    const char *password = env_or_default("SIM_SLICE_REDIS_PASSWORD", "");
    const char *jobs_stream = env_or_default(
        "SIM_SLICE_REDIS_JOBS_STREAM",
        "eta_mu:sim_slice_jobs"
    );
    const char *worker_name = env_or_default(
        "SIM_SLICE_WORKER_NAME",
        "c-budget-worker.v1"
    );
    const int reply_ttl_seconds = safe_int(
        env_or_default("SIM_SLICE_REPLY_TTL_SECONDS", "20"),
        20
    );

    char last_id[64];
    snprintf(last_id, sizeof(last_id), "%s", "$");

    fprintf(stdout, "[sim-slice-worker] listening on redis://%s:%d stream=%s\n", host, port, jobs_stream);
    fflush(stdout);

    redisContext *ctx = NULL;
    while (1) {
        if (ctx == NULL) {
            ctx = connect_redis(host, port, password);
            if (ctx == NULL) {
                sleep_microseconds(350000u);
                continue;
            }
        }

        redisReply *reply = redisCommand(
            ctx,
            "XREAD BLOCK 1500 COUNT 1 STREAMS %s %s",
            jobs_stream,
            last_id
        );
        if (reply == NULL) {
            fprintf(stderr, "redis xread failed, reconnecting\n");
            redisFree(ctx);
            ctx = NULL;
            sleep_microseconds(200000u);
            continue;
        }
        if (reply->type == REDIS_REPLY_NIL) {
            freeReplyObject(reply);
            continue;
        }
        if (reply->type != REDIS_REPLY_ARRAY || reply->elements == 0) {
            freeReplyObject(reply);
            continue;
        }

        redisReply *stream_row = reply->element[0];
        if (stream_row == NULL || stream_row->type != REDIS_REPLY_ARRAY || stream_row->elements < 2) {
            freeReplyObject(reply);
            continue;
        }
        redisReply *entries = stream_row->element[1];
        if (entries == NULL || entries->type != REDIS_REPLY_ARRAY || entries->elements == 0) {
            freeReplyObject(reply);
            continue;
        }
        redisReply *entry = entries->element[0];
        if (entry == NULL || entry->type != REDIS_REPLY_ARRAY || entry->elements < 2) {
            freeReplyObject(reply);
            continue;
        }
        redisReply *entry_id = entry->element[0];
        redisReply *fields = entry->element[1];
        if (entry_id == NULL || entry_id->type != REDIS_REPLY_STRING) {
            freeReplyObject(reply);
            continue;
        }
        snprintf(last_id, sizeof(last_id), "%s", entry_id->str);

        const char *slice = field_value(fields, "slice");
        const char *job_id = field_value(fields, "job_id");
        const char *reply_key = field_value(fields, "reply_key");
        const char *cpu_text = field_value(fields, "cpu_utilization");
        const char *max_points_text = field_value(fields, "max_sim_points");

        if (slice == NULL || strcmp(slice, "sim_point_budget.v1") != 0) {
            freeReplyObject(reply);
            continue;
        }
        if (reply_key == NULL || reply_key[0] == '\0') {
            freeReplyObject(reply);
            continue;
        }

        const double cpu_utilization = safe_double(cpu_text, 0.0);
        const int max_sim_points = safe_int(max_points_text, 800);
        const int budget = compute_sim_point_budget(cpu_utilization, max_sim_points);

        char payload[256];
        snprintf(
            payload,
            sizeof(payload),
            "{\"job_id\":\"%s\",\"sim_point_budget\":%d,\"source\":\"%s\"}",
            (job_id != NULL && job_id[0] != '\0') ? job_id : "",
            budget,
            worker_name
        );

        redisReply *set_reply = redisCommand(
            ctx,
            "SETEX %s %d %s",
            reply_key,
            reply_ttl_seconds,
            payload
        );
        if (set_reply != NULL) {
            freeReplyObject(set_reply);
        }

        freeReplyObject(reply);
    }

    return 0;
}
