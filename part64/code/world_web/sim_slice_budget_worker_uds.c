#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static const char *env_or_default(const char *name, const char *fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return value;
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

static int extract_json_string(
    const char *payload,
    const char *key,
    char *out,
    size_t out_size
) {
    if (payload == NULL || key == NULL || out == NULL || out_size == 0) {
        return 0;
    }

    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *pos = strstr(payload, needle);
    if (pos == NULL) {
        return 0;
    }
    pos = strchr(pos + strlen(needle), ':');
    if (pos == NULL) {
        return 0;
    }
    pos += 1;
    while (*pos == ' ' || *pos == '\t') {
        pos += 1;
    }
    if (*pos != '"') {
        return 0;
    }
    pos += 1;

    size_t index = 0;
    while (*pos != '\0' && *pos != '"' && index + 1 < out_size) {
        if (*pos == '\\' && pos[1] != '\0') {
            pos += 1;
        }
        out[index++] = *pos;
        pos += 1;
    }
    out[index] = '\0';
    return index > 0;
}

static int extract_json_double(const char *payload, const char *key, double *out) {
    if (payload == NULL || key == NULL || out == NULL) {
        return 0;
    }

    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *pos = strstr(payload, needle);
    if (pos == NULL) {
        return 0;
    }
    pos = strchr(pos + strlen(needle), ':');
    if (pos == NULL) {
        return 0;
    }
    pos += 1;

    while (*pos == ' ' || *pos == '\t') {
        pos += 1;
    }
    char *end = NULL;
    double value = strtod(pos, &end);
    if (end == pos) {
        return 0;
    }
    *out = value;
    return 1;
}

static int extract_json_int(const char *payload, const char *key, int *out) {
    double temp = 0.0;
    if (!extract_json_double(payload, key, &temp)) {
        return 0;
    }
    *out = (int)temp;
    return 1;
}

int main(void) {
    const char *socket_path = env_or_default(
        "SIM_SLICE_UDS_PATH",
        "/tmp/eta_mu_sim_slice.sock"
    );
    const char *worker_name = env_or_default(
        "SIM_SLICE_WORKER_NAME",
        "c-uds-worker.v1"
    );

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "[sim-slice-uds-worker] socket failed: %s\n", strerror(errno));
        return 1;
    }

    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(socket_path) >= sizeof(address.sun_path)) {
        fprintf(stderr, "[sim-slice-uds-worker] socket path too long\n");
        close(server_fd);
        return 1;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path);
    unlink(socket_path);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        fprintf(stderr, "[sim-slice-uds-worker] bind failed: %s\n", strerror(errno));
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 128) < 0) {
        fprintf(stderr, "[sim-slice-uds-worker] listen failed: %s\n", strerror(errno));
        close(server_fd);
        unlink(socket_path);
        return 1;
    }

    fprintf(
        stdout,
        "[sim-slice-uds-worker] listening socket=%s worker=%s\n",
        socket_path,
        worker_name
    );
    fflush(stdout);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            continue;
        }

        char request[4096];
        size_t used = 0;
        memset(request, 0, sizeof(request));
        while (used + 1 < sizeof(request)) {
            ssize_t got = recv(client_fd, request + used, sizeof(request) - used - 1, 0);
            if (got <= 0) {
                break;
            }
            used += (size_t)got;
            if (memchr(request, '\n', used) != NULL) {
                break;
            }
        }
        request[used] = '\0';

        char slice_id[64];
        char job_id[96];
        memset(slice_id, 0, sizeof(slice_id));
        memset(job_id, 0, sizeof(job_id));
        double cpu_utilization = 0.0;
        int max_sim_points = 800;

        int has_slice = extract_json_string(request, "slice", slice_id, sizeof(slice_id));
        int has_job_id = extract_json_string(request, "job_id", job_id, sizeof(job_id));
        int has_cpu = extract_json_double(request, "cpu_utilization", &cpu_utilization);
        int has_max_points = extract_json_int(request, "max_sim_points", &max_sim_points);

        int budget = 0;
        if (
            has_slice
            && has_job_id
            && has_cpu
            && has_max_points
            && strcmp(slice_id, "sim_point_budget.v1") == 0
        ) {
            budget = compute_sim_point_budget(cpu_utilization, max_sim_points);
        }

        char response[384];
        snprintf(
            response,
            sizeof(response),
            "{\"job_id\":\"%s\",\"sim_point_budget\":%d,\"source\":\"%s\"}\n",
            has_job_id ? job_id : "",
            budget,
            worker_name
        );
        (void)send(client_fd, response, strlen(response), 0);
        close(client_fd);
    }

    close(server_fd);
    unlink(socket_path);
    return 0;
}
