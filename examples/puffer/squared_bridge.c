#include "ocean/squared_continuous/squared_continuous.h"

Squared* squared_create(obs_t* observations, float* actions, float* reward, float* terminal, unsigned int seed) {
    Squared* env = calloc(1, sizeof(*env));
    if (!env) return NULL;
    DictItem size = {.key = "size", .value = 11};
    Dict options = {.items = &size, .size = 1, .cap = 1};
    env->rng = seed;
    env->agents[0] = (Agent){.observations = observations, .actions = actions,
                             .rewards = reward, .terminals = terminal};
    puf_init(env, &options);
    return env;
}

void squared_destroy(Squared* env) {
    free(env);
}
